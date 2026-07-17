"""Small public API for DSVP detection training, validation, and prediction."""

import inspect
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
from PIL import Image

from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
from ultralytics.nn.tasks import attempt_load_one_weight, yaml_model_load
from ultralytics.utils import ASSETS, DEFAULT_CFG_DICT, LOGGER, RANK, callbacks, checks, yaml_load


class Model(torch.nn.Module):
    """Detection-only DSVP model wrapper."""

    def __init__(self, model: Union[str, Path] = "dsvpn.yaml", task="detect", verbose=False):
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None
        self.trainer = None
        self.ckpt = {}
        self.ckpt_path = None
        self.cfg = None
        self.metrics = None
        self.task = "detect"
        self.overrides = {}

        model = str(model).strip()
        if Path(model).suffix.lower() in {".yaml", ".yml"}:
            self._new(model, verbose=verbose)
        else:
            self._load(model)
        self.model_name = model

    def __call__(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        stream=False,
        **kwargs: Any,
    ):
        return self.predict(source=source, stream=stream, **kwargs)

    def _new(self, cfg, verbose=False):
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.model = self._smart_load("model")(cfg_dict, verbose=verbose and RANK == -1)
        self.overrides = {"model": cfg, "task": "detect"}
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}
        self.model.task = "detect"

    def _load(self, weights):
        weights = checks.check_model_file_from_stem(weights)
        if Path(weights).suffix.lower() != ".pt":
            raise ValueError("The minimal runtime accepts only DSVP .yaml and .pt models.")
        self.model, self.ckpt = attempt_load_one_weight(weights)
        self.model.args["task"] = "detect"
        self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
        self.ckpt_path = self.model.pt_path
        self.overrides.update({"model": weights, "task": "detect"})

    def _check_is_pytorch_model(self):
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("This minimal runtime supports only PyTorch DSVP models.")

    def load(self, weights: Union[str, Path]):
        self._check_is_pytorch_model()
        self.overrides["pretrained"] = str(weights)
        weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    def save(self, filename: Union[str, Path]):
        self._check_is_pytorch_model()
        from ultralytics import __version__

        checkpoint = {
            **self.ckpt,
            "model": deepcopy(self.model).half(),
            "date": datetime.now().isoformat(),
            "version": __version__,
        }
        torch.save(checkpoint, filename)

    def info(self, detailed=False, verbose=True):
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self):
        self._check_is_pytorch_model()
        self.model.fuse()
        return self

    def predict(self, source=None, stream=False, predictor=None, **kwargs):
        if source is None:
            source = ASSETS
            LOGGER.warning(f"'source' was not provided; using {source}.")
        args = {**self.overrides, "conf": 0.25, "batch": 1, "save": False, "mode": "predict", **kwargs}
        if self.predictor is None:
            predictor_cls = predictor or self._smart_load("predictor")
            self.predictor = predictor_cls(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=False)
        else:
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        return self.predictor(source=source, stream=stream)

    def val(self, validator=None, **kwargs):
        args = {**self.overrides, "rect": True, **kwargs, "mode": "val", "task": "detect"}
        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return self.metrics

    def train(self, trainer=None, **kwargs):
        self._check_is_pytorch_model()
        overrides = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        custom = {
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA["detect"],
            "model": self.overrides["model"],
            "task": "detect",
        }
        args = {**overrides, **custom, **kwargs, "mode": "train"}
        if args.get("resume"):
            args["resume"] = self.ckpt_path

        self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
        if not args.get("resume"):
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.train()

        if RANK in {-1, 0}:
            checkpoint = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, self.ckpt = attempt_load_one_weight(checkpoint)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)
        return self.metrics

    def _apply(self, fn):
        self._check_is_pytorch_model()
        result = super()._apply(fn)
        self.predictor = None
        self.overrides["device"] = self.device
        return result

    @property
    def names(self) -> Dict[int, str]:
        from ultralytics.nn.autobackend import check_class_names

        return check_class_names(self.model.names)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def add_callback(self, event, func):
        self.callbacks[event].append(func)

    def clear_callback(self, event):
        self.callbacks[event] = []

    @staticmethod
    def _reset_ckpt_args(args):
        include = {"imgsz", "data", "task", "single_cls"}
        return {key: value for key, value in args.items() if key in include}

    def _smart_load(self, key):
        try:
            return self.task_map["detect"][key]
        except Exception as error:
            caller = inspect.stack()[1][3]
            raise NotImplementedError(f"DSVP detection does not support '{caller}' in this build.") from error

    @property
    def task_map(self):
        raise NotImplementedError
