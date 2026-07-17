import math

import torch
import torch.nn as nn

from ultralytics.utils.tal import dist2bbox, make_anchors
from .block import DFL
from .conv import Conv
from .temporal import SelectiveStateSpace2D, StreamingBoxCoordinationRegressor

__all__ = ("Detect",)


class Detect(nn.Module):
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = True

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.channels = tuple(ch)
        self.temporal = None
        self.box_memory = None
        self.stream_active = False
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        box_channels = max(16, ch[0] // 4, self.reg_max * 4)
        cls_channels = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, box_channels, 3), Conv(box_channels, box_channels, 3), nn.Conv2d(box_channels, 64, 1))
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, cls_channels, 3), Conv(cls_channels, cls_channels, 3), nn.Conv2d(cls_channels, nc, 1))
            for x in ch
        )
        self.dfl = DFL(self.reg_max)

    def configure_stream(self):
        if getattr(self, "temporal", None) is None:
            channels = getattr(
                self,
                "channels",
                tuple(branch[0].conv.in_channels for branch in self.cv2),
            )
            self.channels = tuple(channels)
            self.temporal = nn.ModuleList(SelectiveStateSpace2D(channels) for channels in self.channels)
        if getattr(self, "box_memory", None) is None:
            self.box_memory = nn.ModuleList(
                StreamingBoxCoordinationRegressor(self.reg_max) for _ in self.channels
            )
        self.stream_active = getattr(self, "stream_active", False)
        return self

    def set_stream(self, enabled=True):
        if enabled:
            self.configure_stream()
        self.stream_active = enabled
        if not enabled:
            self.reset_stream()
        return self

    def reset_stream(self):
        if getattr(self, "temporal", None) is not None:
            for module in self.temporal:
                module.reset_state()
        if getattr(self, "box_memory", None) is not None:
            for module in self.box_memory:
                module.reset_state()

    def forward(self, x):
        if getattr(self, "stream_active", False):
            x = [module(feature) for module, feature in zip(self.temporal, x)]
        for index in range(self.nl):
            raw = torch.cat((self.cv2[index](x[index]), self.cv3[index](x[index])), 1)
            if getattr(self, "stream_active", False):
                raw = self.box_memory[index](raw)
            x[index] = raw
        if self.training:
            return x
        predictions = self._inference(x)
        return predictions if self.export else (predictions, x)

    def _inference(self, x):
        shape = x[0].shape
        merged = torch.cat([feature.view(shape[0], self.no, -1) for feature in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (value.transpose(0, 1) for value in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = merged.split((self.reg_max * 4, self.nc), 1)
        decoded = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((decoded, cls.sigmoid()), 1)

    def bias_init(self):
        for box, cls, stride in zip(self.cv2, self.cv3, self.stride):
            box[-1].bias.data[:] = 1.0
            cls[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)

    @staticmethod
    def decode_bboxes(bboxes, anchors):
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)
