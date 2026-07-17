"""DSVP object detection with legacy package-path compatibility."""

from . import detect
from .model import DSVP, YOLO

__all__ = ("DSVP", "YOLO", "detect")
