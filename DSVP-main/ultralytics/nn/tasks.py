"""DSVP detection model construction and checkpoint loading."""

import ast
import contextlib
import re
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules import Bottleneck, C2f, Concat, Conv, Detect, SPPF, SelectiveStateSpaceModel
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, yaml_load
from ultralytics.utils.checks import check_suffix, check_yaml
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
)


class BaseModel(nn.Module):
    """Common forward, loss, fusion, and weight-loading behavior."""

    def forward(self, x, *args, **kwargs):
        return self.loss(x, *args, **kwargs) if isinstance(x, dict) else self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, visualize=visualize, embed=embed)

    def _predict_once(self, x, visualize=False, embed=None):
        outputs, embeddings = [], []
        for module in self.model:
            if module.f != -1:
                x = (
                    outputs[module.f]
                    if isinstance(module.f, int)
                    else [x if index == -1 else outputs[index] for index in module.f]
                )
            x = module(x)
            outputs.append(x if module.i in self.save else None)
            if visualize:
                feature_visualization(x, module.type, module.i, save_dir=visualize)
            if embed and module.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1))
                if module.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        LOGGER.warning("Augmented inference is disabled in the minimal runtime; using single-scale inference.")
        return self._predict_once(x)

    def fuse(self, verbose=True):
        if not self.is_fused():
            for module in self.model.modules():
                if isinstance(module, Conv) and hasattr(module, "bn"):
                    module.conv = fuse_conv_and_bn(module.conv, module.bn)
                    delattr(module, "bn")
                    module.forward = module.forward_fuse
            self.info(verbose=verbose)
        return self

    def is_fused(self, threshold=10):
        norm_types = tuple(value for key, value in nn.__dict__.items() if "Norm" in key)
        return sum(isinstance(value, norm_types) for value in self.modules()) < threshold

    def info(self, detailed=False, verbose=True, imgsz=640):
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        self = super()._apply(fn)
        head = self.model[-1]
        if isinstance(head, Detect):
            head.stride = fn(head.stride)
            head.anchors = fn(head.anchors)
            head.strides = fn(head.strides)
        return self

    def load(self, weights, verbose=True):
        source = weights["model"] if isinstance(weights, dict) else weights
        state = intersect_dicts(source.float().state_dict(), self.state_dict())
        self.load_state_dict(state, strict=False)
        if verbose:
            LOGGER.info(f"Transferred {len(state)}/{len(self.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        raise NotImplementedError


class DetectionModel(BaseModel):
    """Dynamic Streaming Vision Pipeline object detection network."""

    def __init__(self, cfg="dsvpn.yaml", ch=3, nc=None, verbose=True):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc is not None and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {index: str(index) for index in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)
        head = self.model[-1]
        if not isinstance(head, Detect):
            raise TypeError("A DSVP detection model must end with a Detect layer.")
        if self.yaml.get("stream", False):
            head.configure_stream()

        size = 256
        head.inplace = self.inplace
        head.stride = torch.tensor(
            [size / feature.shape[-2] for feature in self.forward(torch.zeros(1, ch, size, size))]
        )
        self.stride = head.stride
        head.bias_init()
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def set_stream(self, enabled=True):
        """Toggle causal backbone and detection-head memory."""
        for module in self.model.modules():
            if isinstance(module, SelectiveStateSpaceModel):
                module.set_stream(enabled)
        self.model[-1].set_stream(enabled)
        return self

    def reset_stream(self):
        """Reset all temporal states at a video/scene boundary."""
        for module in self.model.modules():
            if isinstance(module, SelectiveStateSpaceModel):
                module.reset_stream()
        self.model[-1].reset_stream()
        return self

    def _predict_augment(self, x):
        image_size = x.shape[-2:]
        predictions = []
        for scale, flip in zip((1.0, 0.83, 0.67), (None, 3, None)):
            image = scale_img(x.flip(flip) if flip else x, scale, gs=int(self.stride.max()))
            pred = super().predict(image)[0]
            pred[:, :4] /= scale
            x_coord, y_coord, wh, cls = pred.split((1, 1, 2, pred.shape[1] - 4), 1)
            if flip == 2:
                y_coord = image_size[0] - y_coord
            elif flip == 3:
                x_coord = image_size[1] - x_coord
            predictions.append(torch.cat((x_coord, y_coord, wh, cls), 1))
        return torch.cat(predictions, -1), None

    def init_criterion(self):
        return v8DetectionLoss(self)


def torch_safe_load(weight):
    """Load a DSVP-compatible PyTorch checkpoint."""
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)
    checkpoint = torch.load(file, map_location="cpu")
    if not isinstance(checkpoint, dict):
        checkpoint = {"model": checkpoint.model}
    return checkpoint, file


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Load one DSVP checkpoint and attach its training arguments."""
    checkpoint, weight = torch_safe_load(weight)
    args = {**DEFAULT_CFG_DICT, **checkpoint.get("train_args", {})}
    model = (checkpoint.get("ema") or checkpoint["model"]).to(device).float()
    model.args = {key: value for key, value in args.items() if key in DEFAULT_CFG_KEYS}
    model.args["task"] = "detect"
    model.pt_path = weight
    model.task = "detect"
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])
    model = model.fuse().eval() if fuse else model.eval()
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = inplace
        elif isinstance(module, nn.Upsample) and not hasattr(module, "recompute_scale_factor"):
            module.recompute_scale_factor = None
    return model, checkpoint


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Load a checkpoint; model ensembles are intentionally unsupported."""
    paths = weights if isinstance(weights, list) else [weights]
    if len(paths) != 1:
        raise ValueError("The minimal DSVP runtime does not support model ensembles.")
    return attempt_load_one_weight(paths[0], device=device, inplace=inplace, fuse=fuse)[0]


def parse_model(config, ch, verbose=True):
    """Build the DSVP detection graph from its YAML dictionary."""
    max_channels = float("inf")
    nc, activation, scales = (config.get(key) for key in ("nc", "activation", "scales"))
    depth = config.get("depth_multiple", 1.0)
    width = config.get("width_multiple", 1.0)
    if scales:
        scale = config.get("scale") or next(iter(scales))
        if not config.get("scale"):
            LOGGER.warning(f"No model scale supplied; using '{scale}'.")
        depth, width, max_channels = scales[scale]

    if activation:
        Conv.default_act = eval(activation)
    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<35}{'arguments':<30}")

    channels = [ch]
    layers, save = [], []
    allowed = {
        "Conv": Conv,
        "C2f": C2f,
        "SelectiveStateSpaceModel": SelectiveStateSpaceModel,
        "SPPF": SPPF,
        "Concat": Concat,
        "Detect": Detect,
    }
    for index, (source, repeats, module_name, args) in enumerate(config["backbone"] + config["head"]):
        if module_name.startswith("nn."):
            module = getattr(nn, module_name[3:])
        else:
            if module_name not in allowed:
                raise ValueError(f"Unsupported DSVP module in minimal runtime: {module_name}")
            module = allowed[module_name]
        for arg_index, value in enumerate(args):
            if isinstance(value, str):
                with contextlib.suppress(ValueError, SyntaxError):
                    args[arg_index] = locals()[value] if value in locals() else ast.literal_eval(value)

        count = count_scaled = max(round(repeats * depth), 1) if repeats > 1 else repeats
        if module in {Conv, C2f, SelectiveStateSpaceModel, SPPF}:
            c1, c2 = channels[source], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8) if c2 != nc else c2
            args = [c1, c2, *args[1:]]
            if module in {C2f, SelectiveStateSpaceModel}:
                args.insert(2, count)
                count = 1
        elif module is Concat:
            c2 = sum(channels[item] for item in source)
        elif module is Detect:
            args.append([channels[item] for item in source])
            c2 = nc
        else:
            c2 = channels[source]

        built = nn.Sequential(*(module(*args) for _ in range(count))) if count > 1 else module(*args)
        built.np = sum(parameter.numel() for parameter in built.parameters())
        built.i, built.f, built.type = index, source, f"{module.__module__}.{module.__name__}"
        if verbose:
            LOGGER.info(
                f"{index:>3}{str(source):>20}{count_scaled:>3}{built.np:10.0f}  {built.type:<35}{str(args):<30}"
            )
        save.extend(item % index for item in ([source] if isinstance(source, int) else source) if item != -1)
        layers.append(built)
        if index == 0:
            channels = []
        channels.append(c2)
    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """Resolve a scaled name such as dsvps.yaml to the shared dsvp.yaml definition."""
    path = Path(path)
    unified = re.sub(r"(dsvp|yolov8)([nslmx])(?=\.|-)", r"\1", str(path), flags=re.IGNORECASE)
    yaml_file = check_yaml(unified, hard=False) or check_yaml(path)
    config = yaml_load(yaml_file)
    config["scale"] = guess_model_scale(path) or config.get("scale", "n")
    config["yaml_file"] = str(path)
    return config


def guess_model_scale(model_path):
    """Extract the n/s/m/l/x compound scale from a DSVP model filename."""
    match = re.search(r"(?:dsvp|yolov8)([nslmx])", Path(model_path).stem, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def guess_model_task(model):
    """This package has one task."""
    return "detect"
