"""Bit-exact numpy executor for the integer graph.

This is the golden model. The C runtime is tested against it op by op, and
the training-side quantiser is tested against it end to end, so any
disagreement between the three is a bug in exactly one place. It favours
being obviously correct over being fast.

Rounding convention, used identically everywhere:

    rshift_round(v, s) = floor((v + 2^(s-1)) / 2^s)      (round half up)

which in C is `(v + (1 << (s - 1))) >> s` on an arithmetic shift.
"""

from __future__ import annotations

import numpy as np

from .formats import SF4, SF8, widen
from .graph import ACT_RAW, Conv2d, Dense, GlobalAvgPool, Graph, act_bounds


def rshift_round(v, s: int) -> np.ndarray:
    """Rounding arithmetic right shift; a negative s shifts left."""
    v = np.asarray(v, dtype=np.int64)
    if s <= 0:
        return v << -s
    return (v + (1 << (s - 1))) >> s


def div_round(v, d: int) -> np.ndarray:
    """Integer division rounding half up, valid for negative numerators."""
    v = np.asarray(v, dtype=np.int64)
    return np.floor_divide(2 * v + d, 2 * d)


def requantize(acc, shift: int, act: int) -> np.ndarray:
    if act == ACT_RAW:
        return np.asarray(acc, dtype=np.int32)
    lo, hi = act_bounds(act)
    return np.clip(rshift_round(acc, shift), lo, hi).astype(np.int8)


def sf8_weight(op) -> np.ndarray:
    """Weight codes as SF8 int8, widening SF4 if needed."""
    if op.wbits == 4:
        return widen(op.weight, SF4, SF8)
    return op.weight.astype(np.int8)


def rgb8_to_sf8(pixels) -> np.ndarray:
    """uint8 pixels -> SF8 codes: (p - 128) / 128, i.e. flip the top bit."""
    return (np.asarray(pixels, dtype=np.uint8) ^ 0x80).view(np.int8)


def conv2d(x: np.ndarray, op: Conv2d) -> np.ndarray:
    """x: (H, W, C) int8 -> (OH, OW, COUT) int8 (or int32 for ACT_RAW)."""
    h, w, c = x.shape
    kh, kw = op.kernel
    oh, ow, cout = op.out_shape(h, w)
    g = op.groups
    cin_g, cout_g = c // g, cout // g
    wt = sf8_weight(op).astype(np.int64)
    xp = np.zeros((h + 2 * op.pad, w + 2 * op.pad, c), dtype=np.int64)
    xp[op.pad:op.pad + h, op.pad:op.pad + w] = x
    s = op.stride
    acc = np.zeros((oh, ow, cout), dtype=np.int64)
    for ky in range(kh):
        for kx in range(kw):
            patch = xp[ky:ky + s * (oh - 1) + 1:s, kx:kx + s * (ow - 1) + 1:s]
            for gi in range(g):
                xs = patch[:, :, gi * cin_g:(gi + 1) * cin_g]
                ws = wt[gi * cout_g:(gi + 1) * cout_g, ky, kx, :]
                acc[:, :, gi * cout_g:(gi + 1) * cout_g] += xs @ ws.T
    acc += op.bias.astype(np.int64)
    return requantize(acc, op.shift, op.act)


def global_avg_pool(x: np.ndarray) -> np.ndarray:
    h, w, c = x.shape
    total = x.reshape(h * w, c).astype(np.int64).sum(axis=0)
    return div_round(total, h * w).astype(np.int8)


def dense(x: np.ndarray, op: Dense) -> np.ndarray:
    acc = sf8_weight(op).astype(np.int64) @ x.reshape(-1).astype(np.int64)
    acc += op.bias.astype(np.int64)
    return requantize(acc, op.shift, op.act)


def run(graph: Graph, x: np.ndarray, *, trace: bool = False):
    """Execute the graph on one SF8 input of shape graph.input_shape.

    Returns the final output, or the list of every intermediate when
    trace=True (used by the C runtime tests to localise mismatches).
    """
    x = np.asarray(x)
    if x.dtype != np.int8 or x.shape != tuple(graph.input_shape):
        raise ValueError(f"expected int8 {graph.input_shape}, got {x.dtype} {x.shape}")
    outs = []
    for op in graph.ops:
        if isinstance(op, Conv2d):
            x = conv2d(x, op)
        elif isinstance(op, GlobalAvgPool):
            x = global_avg_pool(x)
        elif isinstance(op, Dense):
            x = dense(x, op)
        else:
            raise TypeError(f"unknown op {type(op).__name__}")
        outs.append(x)
    return outs if trace else x


def run_rgb8(graph: Graph, image: np.ndarray):
    """Convenience: uint8 HWC image straight from a camera."""
    return run(graph, rgb8_to_sf8(image))
