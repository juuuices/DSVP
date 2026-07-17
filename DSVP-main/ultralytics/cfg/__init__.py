from pathlib import Path
from types import SimpleNamespace

from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, RANK, ROOT, RUNS_DIR, IterableSimpleNamespace, yaml_load

TASKS = {"detect"}
MODES = {"train", "val", "predict"}
TASK2DATA = {"detect": "EA-UAV.yaml"}
TASK2MODEL = {"detect": "dsvp-stream.yaml"}
TASK2METRIC = {"detect": "metrics/mAP50-95(B)"}


def cfg2dict(cfg):
    """Convert a YAML path, namespace, or mapping to a plain dictionary."""
    if cfg is None:
        return {}
    if isinstance(cfg, (str, Path)):
        return yaml_load(cfg)
    if isinstance(cfg, SimpleNamespace):
        return vars(cfg)
    return dict(cfg)


def get_cfg(cfg=DEFAULT_CFG_DICT, overrides=None):
    """Merge the default configuration with call-specific overrides."""
    values = cfg2dict(cfg).copy()
    if overrides:
        updates = cfg2dict(overrides)
        if "save_dir" not in values:
            updates.pop("save_dir", None)
        unknown = set(updates) - set(values)
        if unknown:
            raise SyntaxError(f"Unknown YOLO configuration option(s): {', '.join(sorted(unknown))}")
        values.update(updates)

    for key in ("project", "name"):
        if isinstance(values.get(key), (int, float)):
            values[key] = str(values[key])
    if values.get("task") not in {None, "detect"}:
        raise ValueError("This minimal project supports only task='detect'.")
    if values.get("mode") not in {None, *MODES}:
        raise ValueError(f"This minimal project supports only modes: {sorted(MODES)}.")
    return IterableSimpleNamespace(**values)


def get_save_dir(args, name=None):
    """Return an incremented output directory."""
    if getattr(args, "save_dir", None):
        return Path(args.save_dir)

    from ultralytics.utils.files import increment_path

    project = args.project or RUNS_DIR / args.task
    run_name = name or args.name or args.mode
    exist_ok = args.exist_ok if RANK in {-1, 0} else True
    return increment_path(Path(project) / run_name, exist_ok=exist_ok)
