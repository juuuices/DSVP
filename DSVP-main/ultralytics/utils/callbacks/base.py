"""Dependency-free lifecycle callbacks used by the local runtime."""

from collections import defaultdict
from copy import deepcopy


def _noop(instance):
    """Default callback."""


EVENTS = (
    "on_pretrain_routine_start",
    "on_pretrain_routine_end",
    "on_train_start",
    "on_train_epoch_start",
    "on_train_batch_start",
    "optimizer_step",
    "on_before_zero_grad",
    "on_train_batch_end",
    "on_train_epoch_end",
    "on_fit_epoch_end",
    "on_model_save",
    "on_train_end",
    "on_params_update",
    "teardown",
    "on_val_start",
    "on_val_batch_start",
    "on_val_batch_end",
    "on_val_end",
    "on_predict_start",
    "on_predict_batch_start",
    "on_predict_postprocess_end",
    "on_predict_batch_end",
    "on_predict_end",
)

default_callbacks = {event: [_noop] for event in EVENTS}


def get_default_callbacks():
    return defaultdict(list, deepcopy(default_callbacks))


def add_integration_callbacks(instance):
    """External experiment trackers are intentionally disabled."""
