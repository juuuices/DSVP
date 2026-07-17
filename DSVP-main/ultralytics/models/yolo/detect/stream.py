"""Scene-aware spatial and causal stream training for EA-UAV."""

from copy import copy

import torch.nn as nn

from ultralytics.data import build_scene_dataset, build_stream_dataloader
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import de_parallel

from .train import DetectionTrainer
from .val import DetectionValidator


def _unwrap_stream_model(model):
    """Find the DetectionModel inside a trainer model or AutoBackend."""
    target = de_parallel(model)
    if not hasattr(target, "set_stream") and hasattr(target, "model"):
        target = target.model
    return target if hasattr(target, "set_stream") else None


def _is_new_sequence(batch):
    value = batch.get("new_sequence", False)
    if isinstance(value, (tuple, list)):
        value = value[0]
    return bool(value)


class SceneDetectionValidator(DetectionValidator):
    """Validate the scene-folder LabelMe dataset without temporal state."""

    def build_dataset(self, img_path, mode="val", batch=None):
        return build_scene_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            stride=self.stride,
            stream=False,
        )


class SceneDetectionTrainer(DetectionTrainer):
    """YOLO spatial trainer for scene-folder LabelMe annotations."""

    def build_dataset(self, img_path, mode="train", batch=None):
        stride = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_scene_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            stride=stride,
            stream=False,
        )

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return SceneDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )


class StreamDetectionValidator(SceneDetectionValidator):
    """Causal validation in exact video order with state reset per scene."""

    def build_dataset(self, img_path, mode="val", batch=None):
        return build_scene_dataset(
            self.args,
            img_path,
            1,
            self.data,
            mode=mode,
            stride=self.stride,
            stream=True,
        )

    def get_dataloader(self, dataset_path, batch_size):
        return build_stream_dataloader(self.build_dataset(dataset_path, batch=1, mode="val"), workers=0)

    def init_metrics(self, model):
        stream_model = _unwrap_stream_model(model)
        if stream_model is None:
            raise TypeError("Stream validation requires the DSVP streaming model.")
        stream_model.set_stream(True)
        stream_model.reset_stream()
        self.stream_model = stream_model
        super().init_metrics(model)

    def preprocess(self, batch):
        if _is_new_sequence(batch):
            self.stream_model.reset_stream()
        return super().preprocess(batch)


class StreamDetectionTrainer(SceneDetectionTrainer):
    """Online batch-1 trainer whose feature state persists across a scene."""

    def set_model_attributes(self):
        super().set_model_attributes()
        stream_model = _unwrap_stream_model(self.model)
        if stream_model is None:
            raise TypeError("Stream training requires the DSVP streaming model.")
        stream_model.set_stream(True)
        stream_model.reset_stream()
        self.stream_model = stream_model

    def build_dataset(self, img_path, mode="train", batch=None):
        stride = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_scene_dataset(
            self.args,
            img_path,
            1,
            self.data,
            mode=mode,
            stride=stride,
            stream=True,
        )

    def get_dataloader(self, dataset_path, batch_size=1, rank=0, mode="train"):
        if rank not in {-1, 0}:
            raise ValueError("Stateful stream training supports one process/GPU only.")
        dataset = self.build_dataset(dataset_path, mode=mode, batch=1)
        return build_stream_dataloader(dataset, workers=0)

    def preprocess_batch(self, batch):
        # BaseTrainer calls model.train() every epoch. Batch-1 BN statistics
        # are unstable even when convolution weights are jointly fine-tuned.
        for module in de_parallel(self.model).modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        if _is_new_sequence(batch):
            self.stream_model.reset_stream()
        return super().preprocess_batch(batch)

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return StreamDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Use a small spatial LR while allowing newly initialized memory modules to learn faster."""
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        scale = float(self.args.stream_lr_scale)
        if scale == 1.0:
            return optimizer
        if scale <= 0:
            raise ValueError("stream_lr_scale must be positive.")

        stream_ids = {
            id(parameter)
            for parameter_name, parameter in de_parallel(model).named_parameters()
            if ".temporal." in parameter_name or ".box_memory." in parameter_name
        }
        backbone_ids = {
            id(parameter)
            for parameter_name, parameter in de_parallel(model).named_parameters()
            if ".temporal." in parameter_name and not parameter_name.startswith("model.22.")
        }
        backbone_scale = float(self.args.backbone_stream_lr_scale or scale)
        base_groups = list(optimizer.param_groups)
        for group in base_groups:
            template = {key: value for key, value in group.items() if key != "params"}
            backbone_parameters = [parameter for parameter in group["params"] if id(parameter) in backbone_ids]
            stream_parameters = [
                parameter
                for parameter in group["params"]
                if id(parameter) in stream_ids and id(parameter) not in backbone_ids
            ]
            spatial_parameters = [parameter for parameter in group["params"] if id(parameter) not in stream_ids]
            # Remove split parameters from their original group before adding
            # new groups; torch.optim rejects a parameter present twice.
            group["params"] = spatial_parameters
            if backbone_parameters:
                backbone_group = template.copy()
                backbone_group["params"] = backbone_parameters
                backbone_group["lr"] = template["lr"] * backbone_scale
                optimizer.add_param_group(backbone_group)
            if stream_parameters:
                stream_group = template.copy()
                stream_group["params"] = stream_parameters
                stream_group["lr"] = template["lr"] * scale
                optimizer.add_param_group(stream_group)

        stream_count = sum(parameter.numel() for parameter in de_parallel(model).parameters() if id(parameter) in stream_ids)
        LOGGER.info(
            f"Stream optimizer: spatial lr={lr:g}, temporal/box-memory lr={lr * scale:g} "
            f"and backbone-memory lr={lr * backbone_scale:g} for {stream_count:,} memory parameters."
        )
        return optimizer

    def _setup_train(self, world_size):
        if world_size > 1:
            raise ValueError("Stream training is sequential and cannot use DDP.")
        if self.batch_size != 1:
            LOGGER.warning("Stream training forces batch=1 to preserve causal scene order.")
            self.args.batch = self.batch_size = 1
        self.args.multi_scale = False
        self.args.mosaic = self.args.mixup = self.args.copy_paste = 0.0
        super()._setup_train(world_size)
        if self.args.temporal_only or self.args.stream_head_only or self.args.backbone_temporal_only:
            selected_backbone_layers = {
                int(index)
                for index in str(self.args.backbone_temporal_layers or "").split(",")
                if index.strip()
            }
            trainable = 0
            for name, parameter in de_parallel(self.model).named_parameters():
                if self.args.backbone_temporal_only:
                    is_backbone_temporal = ".temporal." in name and not name.startswith("model.22.")
                    layer = int(name.split(".")[1]) if name.startswith("model.") else -1
                    parameter.requires_grad = is_backbone_temporal and (
                        not selected_backbone_layers or layer in selected_backbone_layers
                    )
                elif self.args.temporal_only:
                    parameter.requires_grad = ".temporal." in name or ".box_memory." in name
                else:
                    parameter.requires_grad = name.startswith("model.22.")
                if parameter.requires_grad:
                    trainable += parameter.numel()
            if self.args.backbone_temporal_only:
                suffix = f" at layers {sorted(selected_backbone_layers)}" if selected_backbone_layers else ""
                scope = f"backbone C2f temporal modules{suffix}"
            elif self.args.temporal_only:
                scope = "temporal + box-memory modules"
            else:
                scope = "Detect head + stream modules"
            LOGGER.info(f"Stream stage: optimizing {trainable:,} parameters in {scope}.")
