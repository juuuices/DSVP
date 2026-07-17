"""PyTorch-only inference backend for DSVP detection."""

from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER, yaml_load


def check_class_names(names):
    """Normalize class names to a zero-indexed integer dictionary."""
    if isinstance(names, list):
        names = dict(enumerate(names))
    if isinstance(names, dict):
        names = {int(key): str(value) for key, value in names.items()}
        if names and max(names) >= len(names):
            raise KeyError(f"Class indices must be contiguous from 0 to {len(names) - 1}: {names}")
    return names


def default_class_names(data=None):
    """Load names from a dataset YAML or return generic names."""
    if data:
        try:
            return check_class_names(yaml_load(data)["names"])
        except Exception:
            pass
    return {index: f"class{index}" for index in range(999)}


class AutoBackend(nn.Module):
    """Uniform wrapper around an in-memory model or DSVP `.pt` checkpoint."""

    def __init__(
        self,
        weights="yolov8n.pt",
        device=torch.device("cpu"),
        dnn=False,
        data=None,
        fp16=False,
        batch=1,
        fuse=True,
        verbose=True,
    ):
        super().__init__()
        self.nn_module = isinstance(weights, nn.Module)
        if self.nn_module:
            model = weights.to(device)
            if fuse:
                model = model.fuse(verbose=verbose)
        else:
            path = str(weights[0] if isinstance(weights, list) else weights)
            if Path(path).suffix.lower() != ".pt":
                raise ValueError("The minimal DSVP runtime supports only PyTorch .pt checkpoints.")
            from ultralytics.nn.tasks import attempt_load_weights

            model = attempt_load_weights(weights, device=device, inplace=True, fuse=fuse)

        self.model = model
        self.device = device
        self.fp16 = bool(fp16 and device.type != "cpu")
        self.model.half() if self.fp16 else self.model.float()
        self.names = check_class_names(model.module.names if hasattr(model, "module") else model.names)
        self.stride = max(int(model.stride.max()), 32)
        self.task = "detect"
        self.pt = True
        self.triton = False
        self.jit = self.onnx = self.engine = self.saved_model = self.pb = False
        self.batch = batch

    def forward(self, im, augment=False, visualize=False, embed=None):
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()
        return self.model(im, augment=augment, visualize=visualize, embed=embed)

    def warmup(self, imgsz=(1, 3, 640, 640)):
        if self.device.type != "cpu":
            dtype = torch.float16 if self.fp16 else torch.float32
            self.forward(torch.empty(*imgsz, dtype=dtype, device=self.device))

    @staticmethod
    def _model_type(path="model.pt"):
        """Compatibility helper returning only the PyTorch backend flag."""
        return [Path(str(path)).suffix.lower() == ".pt"] + [False] * 15
