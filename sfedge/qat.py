"""Training-side SuperFloat layers that export to the integer graph exactly.

Training runs in two phases:

1. float: ordinary Conv + BatchNorm + clipped ReLU, so the network learns to
   live inside the [0, 1) activation range from the start;
2. quantised (after `quantize_model`): BatchNorm statistics are frozen and
   folded into the conv, the folded weights are snapped to the SFx grid with
   the bounded straight-through estimator from aloshdenny/superfloat, biases
   to the accumulator grid, and activations to SF8.

In phase 2 the float32 forward pass computes *exactly* what the integer
runtime computes. Every weight is a multiple of 2^e/128, every activation a
multiple of 1/128, so every product and partial sum in the convolution is an
integer number of accumulator units well below 2^24 and float32 represents it
without rounding, in any summation order. tests/test_qat_export.py checks the
equality bit for bit, which is what makes the trained accuracy the deployed
accuracy.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .formats import SF8, get_format

ACT_SCALE = float(SF8.scale)  # activations are always SF8
ACTS = ("relu1", "linear", "raw")


def sf_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Snap to the SFx grid; bounded STE (zero gradient where saturated)."""
    fmt = get_format(bits)
    xc = torch.clamp(x, -fmt.vmax, fmt.vmax)
    return xc + (torch.round(xc * fmt.scale) / fmt.scale - xc).detach()


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def act_float(x: torch.Tensor, act: str) -> torch.Tensor:
    if act == "relu1":
        return torch.clamp(x, 0.0, SF8.vmax)
    if act == "linear":
        return torch.clamp(x, -SF8.vmax, SF8.vmax)
    return x


def act_quantize(x: torch.Tensor, act: str) -> torch.Tensor:
    """Clamp, then round half up onto the SF8 grid, as the runtime does.

    Half up rather than half to even: the runtime's (v + 2^(s-1)) >> s rounds
    ties up, and ties are common (one in 2^s accumulator values), so the
    training-side rounding has to agree on them.
    """
    if act == "raw":
        return x
    xc = act_float(x, act)
    return xc + (torch.floor(xc * ACT_SCALE + 0.5) / ACT_SCALE - xc).detach()


def choose_gain_exp(w: torch.Tensor, bits: int, max_exp: int = 7) -> int:
    """Smallest e >= 0 such that w / 2^e fits the SFx range."""
    m = float(w.detach().abs().max())
    vmax = get_format(bits).vmax
    if m <= vmax:
        return 0
    return min(max_exp, math.ceil(math.log2(m / vmax)))


class _SFLayer(nn.Module):
    """Shared state: weight width, activation, gain exponent, phase flag."""

    def __init__(self, act: str, wbits: int):
        super().__init__()
        if act not in ACTS:
            raise ValueError(f"act must be one of {ACTS}")
        get_format(wbits)
        self.act = act
        self.wbits = wbits
        self.register_buffer("gain_exp", torch.zeros((), dtype=torch.int32))
        self.register_buffer("quantized", torch.zeros((), dtype=torch.bool))

    def folded(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @property
    def acc_step(self) -> float:
        """Real value of one int32 accumulator unit: 2^e * 2^-7 * 2^-7."""
        return 2.0 ** int(self.gain_exp) / (128.0 * ACT_SCALE)

    def grid_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Folded weight and bias on the integer grid, still as real values."""
        w, b = self.folded()
        g = 2.0 ** int(self.gain_exp)
        wq = sf_quantize(w / g, self.wbits) * g
        bq = ste_round(b / self.acc_step) * self.acc_step
        return wq, bq

    @torch.no_grad()
    def integer_params(self):
        """(weight codes in `wbits`, int32 bias, gain_exp) for export."""
        w, b = self.folded()
        g = 2.0 ** int(self.gain_exp)
        fmt = get_format(self.wbits)
        wc = torch.clamp(torch.round(w / g * fmt.scale), -fmt.qmax, fmt.qmax).to(torch.int8)
        bc = torch.round(b / self.acc_step).to(torch.int64)
        if bc.abs().max() >= 2**31:
            raise OverflowError("bias does not fit int32")
        return wc, bc.to(torch.int32), int(self.gain_exp)

    @torch.no_grad()
    def start_quantized(self) -> None:
        w, _ = self.folded()
        self.gain_exp.fill_(choose_gain_exp(w, self.wbits))
        self.quantized.fill_(True)


class SFConv2d(_SFLayer):
    """Conv2d (+ BatchNorm) + activation, foldable into one integer conv op."""

    def __init__(self, cin, cout, k=3, stride=1, groups=1, act="relu1", wbits=8, bn=True):
        super().__init__(act, wbits)
        self.conv = nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=not bn)
        self.bn = nn.BatchNorm2d(cout) if bn else None

    def folded(self):
        w = self.conv.weight
        b = self.conv.bias if self.conv.bias is not None else torch.zeros_like(w[:, 0, 0, 0])
        if self.bn is None:
            return w, b
        scale = self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)
        return w * scale[:, None, None, None], self.bn.bias + (b - self.bn.running_mean) * scale

    def forward(self, x):
        if not bool(self.quantized):
            y = self.conv(x)
            if self.bn is not None:
                y = self.bn(y)
            return act_float(y, self.act)
        wq, bq = self.grid_params()
        c = self.conv
        y = F.conv2d(x, wq, bq, c.stride, c.padding, c.dilation, c.groups)
        return act_quantize(y, self.act)


class SFLinear(_SFLayer):
    def __init__(self, cin, cout, act="raw", wbits=8):
        super().__init__(act, wbits)
        self.fc = nn.Linear(cin, cout)

    def folded(self):
        return self.fc.weight, self.fc.bias

    def forward(self, x):
        if not bool(self.quantized):
            return act_float(self.fc(x), self.act)
        wq, bq = self.grid_params()
        return act_quantize(F.linear(x, wq, bq), self.act)


class SFGlobalAvgPool(nn.Module):
    """Mean over H and W; in the quantised phase, rounded half up to SF8."""

    def forward(self, x):
        mean = x.mean(dim=(2, 3))
        if not getattr(self, "quantized", False):
            return mean
        hw = x.shape[2] * x.shape[3]
        total = torch.round(x.sum(dim=(2, 3)) * ACT_SCALE)  # exact integers
        q = torch.floor((2 * total + hw) / (2 * hw)) / ACT_SCALE
        return mean + (q - mean).detach()


def quantize_model(model: nn.Module) -> nn.Module:
    """Switch every SuperFloat layer to the quantised phase, in place.

    Call after float training (or loading float weights). BatchNorm running
    statistics are frozen from here on: the folded forward pass never calls
    the BN module, so they are simply no longer updated.
    """
    for m in model.modules():
        if isinstance(m, _SFLayer):
            m.start_quantized()
        elif isinstance(m, SFGlobalAvgPool):
            m.quantized = True
    return model


@torch.no_grad()
def clamp_to_grid_(model: nn.Module) -> None:
    """Keep shadow weights inside the representable range after a step.

    The bounded STE zeroes the gradient of a saturated weight, so one that
    drifts past the range edge could never come back. For a BN-folded conv
    the bound applies to the folded weight, so it becomes a per-channel bound
    on the raw conv weight.
    """
    for m in model.modules():
        if not isinstance(m, _SFLayer) or not bool(m.quantized):
            continue
        limit = get_format(m.wbits).vmax * 2.0 ** int(m.gain_exp)
        if isinstance(m, SFConv2d) and m.bn is not None:
            scale = (m.bn.weight / torch.sqrt(m.bn.running_var + m.bn.eps)).abs().clamp_min(1e-8)
            bound = (limit / scale)[:, None, None, None]
            m.conv.weight.copy_(torch.maximum(torch.minimum(m.conv.weight, bound), -bound))
        else:
            weight = m.conv.weight if isinstance(m, SFConv2d) else m.fc.weight
            weight.clamp_(-limit, limit)


def image_to_input(images_u8: torch.Tensor) -> torch.Tensor:
    """uint8 NHWC or NCHW images -> real SF8 inputs (p - 128) / 128, NCHW."""
    x = images_u8
    if x.dim() == 4 and x.shape[-1] in (1, 2, 3, 4) and x.shape[1] not in (1, 2, 3, 4):
        x = x.permute(0, 3, 1, 2)
    return (x.float() - 128.0) / 128.0
