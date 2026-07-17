"""Dynamic Streaming Vision Pipeline public model API."""

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel


class DSVP(Model):
    """Dynamic Streaming Vision Pipeline object detector."""

    def __init__(self, model="dsvpn.yaml", verbose=False):
        super().__init__(model=model, task="detect", verbose=verbose)

    @property
    def task_map(self):
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }


# Backward-compatible symbol for loading historical Ultralytics checkpoints.
YOLO = DSVP
