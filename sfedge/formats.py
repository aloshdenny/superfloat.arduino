"""SuperFloat (SFx) number formats, seen from an integer runtime.

SFx is signed fractional fixed point: 1 sign bit, x-1 fraction bits, no
exponent and no integer bit, i.e. Q1.(x-1). The representable set is

    { k / 2^(x-1) : k integer, |k| <= 2^(x-1) - 1 }

which is the grid defined in aloshdenny/superfloat (benchmarks/superfloat.py).
That repository trains on the grid in floating point. This module adds what an
integer runtime needs on top of it: the code <-> value maps, the storage type
for each width, SF4 nibble packing, and the rounding right shift that every
requantisation in the runtime reduces to.

The property the whole runtime leans on: every SFx scale is a power of two, so
moving between formats, or from a wide accumulator back to a narrow
activation, is a shift. There are no per-tensor scales, no zero points and no
fixed-point multipliers anywhere in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SFFormat:
    """One member of the SFx schema."""

    bits: int

    def __post_init__(self):
        if not 2 <= self.bits <= 16:
            raise ValueError(f"SF{self.bits} is outside the schema's 2..16 range")

    @property
    def name(self) -> str:
        return f"SF{self.bits}"

    @property
    def frac_bits(self) -> int:
        return self.bits - 1

    @property
    def scale(self) -> int:
        return 1 << self.frac_bits

    @property
    def qmax(self) -> int:
        """Largest code. The grid is symmetric, so -qmax is the smallest."""
        return self.scale - 1

    @property
    def vmax(self) -> float:
        return self.qmax / self.scale

    @property
    def storage(self) -> type:
        return np.int8 if self.bits <= 8 else np.int16

    def encode(self, x) -> np.ndarray:
        """Real values -> integer codes (round half to even, then saturate).

        Half-to-even matches torch.round, which is what the training-side
        quantiser uses for weights.
        """
        q = np.rint(np.asarray(x, dtype=np.float64) * self.scale)
        return np.clip(q, -self.qmax, self.qmax).astype(self.storage)

    def decode(self, q) -> np.ndarray:
        return np.asarray(q, dtype=np.float32) / np.float32(self.scale)

    def quantize(self, x) -> np.ndarray:
        """Snap real values onto the grid, staying in floating point."""
        return self.decode(self.encode(x))

    def __str__(self) -> str:
        return self.name


SF4 = SFFormat(4)
SF8 = SFFormat(8)
SF16 = SFFormat(16)


def get_format(spec) -> SFFormat:
    """Accept an SFFormat, a bit width (8) or a name ("sf8", "SF8")."""
    if isinstance(spec, SFFormat):
        return spec
    if isinstance(spec, str):
        s = spec.strip().lower()
        if not s.startswith("sf"):
            raise ValueError(f"not an SFx format name: {spec!r}")
        spec = int(s[2:])
    return SFFormat(int(spec))


def widen(codes, src: SFFormat, dst: SFFormat) -> np.ndarray:
    """Re-express codes of a narrow format in a wider one, exactly.

    SFx is backward compatible down the schema: every SF4 value is an SF8
    value, every SF8 value an SF16 value. The conversion is a left shift.
    This is what lets the runtime keep a single int8 kernel and still serve
    SF4 models: SF4 codes are widened to SF8 codes when the model loads.
    """
    if dst.bits < src.bits:
        raise ValueError(f"cannot widen {src} to narrower {dst}")
    q = np.asarray(codes).astype(np.int32) << (dst.frac_bits - src.frac_bits)
    return q.astype(dst.storage)


def pack_sf4(codes) -> np.ndarray:
    """Pack SF4 codes (int, |k| <= 7) two per byte, low nibble first.

    Nibbles hold 4-bit two's complement. An odd count is padded with a zero
    nibble. The C runtime unpacks with the same convention (sf_unpack_sf4).
    """
    q = np.asarray(codes, dtype=np.int8).ravel()
    if q.size and np.abs(q).max() > SF4.qmax:
        raise ValueError("SF4 codes must lie in [-7, 7]")
    if q.size % 2:
        q = np.concatenate([q, np.zeros(1, np.int8)])
    nib = (q.astype(np.uint8) & 0x0F)
    return (nib[0::2] | (nib[1::2] << 4)).astype(np.uint8)


def unpack_sf4(packed, count: int) -> np.ndarray:
    """Inverse of pack_sf4: returns `count` int8 SF4 codes."""
    b = np.asarray(packed, dtype=np.uint8)
    nib = np.empty(b.size * 2, dtype=np.uint8)
    nib[0::2] = b & 0x0F
    nib[1::2] = b >> 4
    q = nib.astype(np.int8)
    q[q > 7] -= 16
    return q[:count]
