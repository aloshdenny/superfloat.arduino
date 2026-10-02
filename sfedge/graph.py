"""Integer graph IR shared by the exporter, the reference runtime and the C runtime.

A model is a straight chain of ops over NHWC int8 activations. Every
activation tensor is SF8 (int8 codes, scale 2^7). Weights are SF8 or SF4
codes; SF4 codes are widened to SF8 when executed (see formats.widen), so the
arithmetic below is written once, for SF8.

Accumulator arithmetic for a conv or dense layer with gain exponent e:

    real(acc)  = acc * 2^e * 2^-7 (weights) * 2^-7 (activations)
    out code   = real * 2^7 = acc * 2^(e - 7)

so requantisation is a rounding right shift by (7 - e). The gain exponent is
one power of two per layer, chosen at export so the BatchNorm-folded weights
fit SF8's [-1, 1) range. It is a shift amount, not a multiplier: there is no
per-channel scale anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .formats import SF8, get_format

ACT_LINEAR = 0  # saturate to the full SF8 range [-127, 127]
ACT_RELU1 = 1  # ReLU clipped at the top of the grid: [0, 127]
ACT_RAW = 2  # keep the int32 accumulator; only valid on the last op

ACT_NAMES = {ACT_LINEAR: "linear", ACT_RELU1: "relu1", ACT_RAW: "raw"}

ACT_FRAC_BITS = SF8.frac_bits


def act_bounds(act: int) -> tuple[int, int]:
    if act == ACT_RELU1:
        return 0, SF8.qmax
    if act == ACT_LINEAR:
        return -SF8.qmax, SF8.qmax
    raise ValueError(f"act {act} has no int8 bounds")


@dataclass
class Conv2d:
    weight: np.ndarray  # (cout, kh, kw, cin // groups) codes in `wbits`
    bias: np.ndarray  # (cout,) int32, accumulator units
    stride: int = 1
    pad: int = 0
    groups: int = 1
    gain_exp: int = 0
    act: int = ACT_RELU1
    wbits: int = 8

    @property
    def cout(self) -> int:
        return self.weight.shape[0]

    @property
    def kernel(self) -> tuple[int, int]:
        return self.weight.shape[1], self.weight.shape[2]

    @property
    def shift(self) -> int:
        return ACT_FRAC_BITS - self.gain_exp

    def out_shape(self, h: int, w: int) -> tuple[int, int, int]:
        kh, kw = self.kernel
        oh = (h + 2 * self.pad - kh) // self.stride + 1
        ow = (w + 2 * self.pad - kw) // self.stride + 1
        return oh, ow, self.cout


@dataclass
class GlobalAvgPool:
    """Mean over H and W, rounded half up back onto the SF8 grid."""


@dataclass
class Dense:
    weight: np.ndarray  # (cout, cin) codes in `wbits`
    bias: np.ndarray  # (cout,) int32
    gain_exp: int = 0
    act: int = ACT_RAW
    wbits: int = 8

    @property
    def cout(self) -> int:
        return self.weight.shape[0]

    @property
    def shift(self) -> int:
        return ACT_FRAC_BITS - self.gain_exp


@dataclass
class Graph:
    input_shape: tuple[int, int, int]  # (H, W, C)
    ops: list = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    name: str = "model"

    def shapes(self) -> list[tuple[int, ...]]:
        """Activation shape after each op, starting with the input."""
        h, w, c = self.input_shape
        out = [(h, w, c)]
        for op in self.ops:
            if isinstance(op, Conv2d):
                if op.weight.shape[3] * op.groups != c:
                    raise ValueError(f"conv expects {op.weight.shape[3] * op.groups} channels, got {c}")
                h, w, c = op.out_shape(h, w)
                out.append((h, w, c))
            elif isinstance(op, GlobalAvgPool):
                h, w = 1, 1
                out.append((c,))
            elif isinstance(op, Dense):
                if op.weight.shape[1] != h * w * c:
                    raise ValueError(f"dense expects {op.weight.shape[1]} inputs, got {h * w * c}")
                h, w, c = 1, 1, op.cout
                out.append((c,))
            else:
                raise TypeError(f"unknown op {type(op).__name__}")
        return out

    def validate(self) -> None:
        self.shapes()
        for i, op in enumerate(self.ops):
            wbits = getattr(op, "wbits", None)
            if wbits is not None:
                fmt = get_format(wbits)
                if fmt.bits not in (4, 8):
                    raise ValueError(f"op {i}: runtime supports SF4/SF8 weights, got {fmt}")
                if np.abs(op.weight).max(initial=0) > fmt.qmax:
                    raise ValueError(f"op {i}: weight codes exceed {fmt}")
            if getattr(op, "act", None) == ACT_RAW and i != len(self.ops) - 1:
                raise ValueError(f"op {i}: raw output is only allowed on the last op")
        if self.labels and self.ops and len(self.labels) != self.shapes()[-1][-1]:
            raise ValueError("label count does not match output width")

    def param_count(self) -> int:
        return sum(op.weight.size + op.bias.size for op in self.ops if hasattr(op, "weight"))

    def weight_bytes(self) -> int:
        """Storage for weights at their native width (SF4 packed two per byte)."""
        total = 0
        for op in self.ops:
            if hasattr(op, "weight"):
                total += (op.weight.size + 1) // 2 if op.wbits == 4 else op.weight.size
        return total

    def macs(self) -> int:
        total = 0
        shapes = self.shapes()
        for op, oshape in zip(self.ops, shapes[1:]):
            if isinstance(op, Conv2d):
                kh, kw, cin_g = op.weight.shape[1:]
                oh, ow, oc = oshape
                total += oh * ow * oc * kh * kw * cin_g
            elif isinstance(op, Dense):
                total += op.weight.size
        return total
