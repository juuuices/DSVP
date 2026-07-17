"""Minimal Dynamic Streaming Vision Pipeline package."""

import os
from pathlib import Path

__version__ = "1.0.0-dsvp"

# Keep Ultralytics runtime settings inside this project instead of writing to
# the user's roaming profile.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".yolo"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from ultralytics.models import DSVP, YOLO

__all__ = ("DSVP", "__version__")
