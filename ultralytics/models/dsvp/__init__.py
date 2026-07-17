"""Public Dynamic Streaming Vision Pipeline namespace."""

from ultralytics.models.yolo.model import DSVP

from . import detect

__all__ = ("DSVP", "detect")
