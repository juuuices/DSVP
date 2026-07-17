import torch
import torch.nn as nn

from .conv import Conv
from .temporal import SelectiveStateSpace2D

__all__ = ("DFL", "Bottleneck", "C2f", "SelectiveStateSpaceModel", "SPPF")


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, k[0], 1)
        self.cv2 = Conv(hidden, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(module(y[-1]) for module in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(module(y[-1]) for module in self.m)
        return self.cv2(torch.cat(y, 1))


class SelectiveStateSpaceModel(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        rng_state = torch.random.get_rng_state()
        self.temporal = SelectiveStateSpace2D(c2, history_only=True)
        torch.random.set_rng_state(rng_state)
        self.temporal.max_gain = 5e-2
        nn.init.constant_(self.temporal.residual_logit, -1.38629436112)  # sigmoid=0.2
        self.temporal.eval_calibration_scale = 6.0
        self.stream_active = False

    def set_stream(self, enabled=True):
        self.stream_active = bool(enabled)
        self.temporal.eval_calibration_scale = 6.0
        if not enabled:
            self.reset_stream()
        return self

    def reset_stream(self):
        self.temporal.reset_state()

    def forward(self, x):
        spatial = super().forward(x)
        return self.temporal(spatial) if self.stream_active else spatial


C2fTemporal = SelectiveStateSpaceModel


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        hidden = c1 // 2
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(hidden * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))
