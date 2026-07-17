import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveStateSpace2D(nn.Module):
    calibration_scale = 0.40

    def __init__(self, channels, min_rate=1e-3, max_rate=0.1, history_only=False):
        super().__init__()
        self.channels = channels
        self.history_only = history_only
        self.norm = nn.GroupNorm(1, channels)
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.input_gate = nn.Conv2d(channels, channels, 1)
        self.output_gate = nn.Conv2d(channels, channels, 1)
        self.dt = nn.Linear(channels, channels)
        rates = torch.logspace(math.log10(min_rate), math.log10(max_rate), channels)
        self.log_a = nn.Parameter(rates.log())
        self.transport = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.out = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        if history_only:

            self.history_gate = nn.Conv2d(2 * channels, channels, 1)
            self.history_out = nn.Conv2d(
                channels, channels, 1, groups=channels, bias=False
            )
            self.alignment_radius = 2
        self.residual_logit = nn.Parameter(torch.tensor(-3.0))
        self.max_gain = 1e-2
        self.register_buffer("_stream_state", None, persistent=False)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.dirac_(self.local.weight, groups=self.channels)
        nn.init.dirac_(self.transport.weight, groups=self.channels)
        nn.init.dirac_(self.value.weight)
        nn.init.zeros_(self.dt.weight)
        nn.init.constant_(self.dt.bias, 0.5413248546)  # softplus^-1(1)
        nn.init.zeros_(self.input_gate.weight)
        nn.init.zeros_(self.input_gate.bias)
        nn.init.zeros_(self.output_gate.weight)
        nn.init.zeros_(self.output_gate.bias)
        nn.init.zeros_(self.out.weight)
        if getattr(self, "history_only", False):
            nn.init.zeros_(self.history_gate.weight)
            nn.init.zeros_(self.history_gate.bias)
            nn.init.zeros_(self.history_out.weight)

    @staticmethod
    def _zero_wrapped_edges(x, dy, dx):
        x = x.clone()
        if dy > 0:
            x[..., :dy, :] = 0
        elif dy < 0:
            x[..., dy:, :] = 0
        if dx > 0:
            x[..., :, :dx] = 0
        elif dx < 0:
            x[..., :, dx:] = 0
        return x

    def _align_history(self, state, current):
        radius = self.alignment_radius
        step = max(1, self.channels // 32)
        state_key = state[:, ::step]
        current_key = current[:, ::step]
        with torch.no_grad():
            shifts = []
            scores = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    candidate = torch.roll(state_key, shifts=(dy, dx), dims=(2, 3))
                    candidate = self._zero_wrapped_edges(candidate, dy, dx)
                    shifts.append((dy, dx))
                    scores.append((candidate * current_key).mean())
            best_shift = shifts[torch.stack(scores).argmax().item()]
        aligned = torch.roll(state, shifts=best_shift, dims=(2, 3))
        return self._zero_wrapped_edges(aligned, *best_shift)

    def reset_state(self):
        self._stream_state = None

    def forward(self, x):
        if x.shape[0] != 1:
            raise ValueError("Streaming state-space modules require batch=1.")
        z = self.norm(x)
        proposal = torch.tanh(self.value(self.local(z)))
        input_gate = torch.sigmoid(self.input_gate(z))
        output_gate = torch.sigmoid(self.output_gate(z))

        state = self._stream_state
        if state is None or state.shape != x.shape or state.device != x.device or state.dtype != x.dtype:
            state = torch.zeros_like(x)
        else:
            if getattr(self, "history_only", False):
                state = self._align_history(state, z)
            state = torch.tanh(
                torch.nan_to_num(self.transport(state), nan=0.0, posinf=1.0, neginf=-1.0)
            )

        pooled = F.adaptive_avg_pool2d(z, 1).flatten(1)
        delta = F.softplus(self.dt(pooled)).view(1, self.channels, 1, 1)
        decay = torch.exp(-self.log_a.exp().view(1, self.channels, 1, 1) * delta)
        new_state = torch.nan_to_num(
            decay * state + (1.0 - decay) * input_gate * proposal,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp_(-1.0, 1.0)

        self._stream_state = new_state.detach()
        calibration = (
            getattr(self, "eval_calibration_scale", self.calibration_scale)
            if not self.training
            else self.calibration_scale
        )
        scale = (
            getattr(self, "max_gain", 1e-2)
            * calibration
            * torch.sigmoid(self.residual_logit)
        )
        if getattr(self, "history_only", False):
            read_gate = torch.sigmoid(self.history_gate(torch.cat((z, state), 1)))
            return x + scale * torch.tanh(self.history_out(read_gate * state))
        return x + scale * torch.tanh(self.out(output_gate * new_state))

    def __getstate__(self):
        state = super().__getstate__()
        buffers = state.get("_buffers", {}).copy()
        buffers["_stream_state"] = None
        state["_buffers"] = buffers
        return state


class StreamingBoxCoordinationRegressor(nn.Module):
    calibration_scale = 0.40

    def __init__(self, reg_max=16, hidden=32):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer("bins", torch.arange(reg_max, dtype=torch.float32), persistent=False)
        self.transport = nn.Conv2d(5, 5, 3, padding=1, groups=5, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(9, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 4 * reg_max, 1),
        )
        self.memory_logit = nn.Parameter(torch.tensor(3.8918203))  # sigmoid=0.98
        self.gain_logit = nn.Parameter(torch.tensor(-2.0))
        self.max_logit_shift = 1.0
        self.register_buffer("_box_state", None, persistent=False)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.dirac_(self.transport.weight, groups=5)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def reset_state(self):
        self._box_state = None

    def _current_boxes(self, raw):
        batch, _, height, width = raw.shape
        regression = raw[:, : 4 * self.reg_max].view(batch, 4, self.reg_max, height, width)
        probability = regression.softmax(2)
        bins = self.bins.to(dtype=raw.dtype).view(1, 1, self.reg_max, 1, 1)
        distance = (probability * bins).sum(2) / (self.reg_max - 1)
        confidence = raw[:, 4 * self.reg_max:].sigmoid().amax(1, keepdim=True)

        return torch.cat((distance, confidence), 1).to(dtype=raw.dtype)

    def forward(self, raw):
        if raw.shape[0] != 1:
            raise ValueError("Streaming box memory requires batch=1.")
        current = self._current_boxes(raw)
        state = self._box_state
        if state is None or state.shape != current.shape or state.device != raw.device or state.dtype != raw.dtype:
            self._box_state = current.detach()
            return raw
        history = torch.nan_to_num(
            self.transport(state), nan=0.0, posinf=1.0, neginf=0.0
        ).clamp_(0.0, 1.0)
        correction = self.refine(torch.cat((current[:, :4], history), 1))
        gain = (
            getattr(self, "max_logit_shift", 1.0)
            * self.calibration_scale
            * torch.sigmoid(self.gain_logit)
        ).to(dtype=raw.dtype)

        regression = raw[:, : 4 * self.reg_max] + gain * torch.tanh(correction)
        raw = torch.cat((regression, raw[:, 4 * self.reg_max:]), 1)
        base_update = (1.0 - torch.sigmoid(self.memory_logit)).to(dtype=current.dtype)
        update = base_update * current[:, 4:5].detach()
        new_state = torch.nan_to_num(
            (1.0 - update) * history + update * current.detach(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp_(0.0, 1.0)
        self._box_state = new_state.detach()
        return raw

    def __getstate__(self):
        state = super().__getstate__()
        buffers = state.get("_buffers", {}).copy()
        buffers["_box_state"] = None
        state["_buffers"] = buffers
        return state


BoxMemoryRefiner = StreamingBoxCoordinationRegressor
