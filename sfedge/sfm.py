"""The .sfm container: one flat little-endian file per model.

The same bytes are read from disk on the Linux side and compiled into flash
on the MCU side, so the layout is designed to be parsed in place without an
allocator: fixed-size records, 4-byte aligned payloads, a CRC at the end.

    header (32 B)
      magic "SFM1" | u16 version | u16 n_ops | u16 in_h | u16 in_w | u16 in_c
      | u16 n_labels | u32 flags | 12 B reserved
    strings
      u8 len + name, then u8 len + label for each label, zero-padded to 4 B
    n_ops records
      u8 op | u8 act | u8 wbits | i8 gain_exp | 12 B op fields
      weights (SF8: 1 B/code, SF4: packed nibbles), zero-padded to 4 B
      bias (i32 x cout)
    u32 crc32 of everything above

Conv fields:  u16 cout | u8 kh | u8 kw | u16 cin_g | u8 stride | u8 pad | u16 groups | u16 0
Dense fields: u16 cout | u16 cin | 8 B 0
Pool fields:  12 B 0
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from .formats import pack_sf4, unpack_sf4
from .graph import Conv2d, Dense, GlobalAvgPool, Graph

MAGIC = b"SFM1"
VERSION = 1

OP_CONV = 1
OP_GAP = 2
OP_DENSE = 3

_HEADER = struct.Struct("<4sHHHHHHI12x")
_OPHEAD = struct.Struct("<BBBb")
_CONV = struct.Struct("<HBBHBBHH")
_DENSE = struct.Struct("<HH8x")


def _pad4(buf: bytearray) -> None:
    buf.extend(b"\0" * (-len(buf) % 4))


def _put_str(buf: bytearray, s: str) -> None:
    b = s.encode("utf-8")
    if len(b) > 255:
        raise ValueError(f"string too long for .sfm: {s!r}")
    buf.append(len(b))
    buf.extend(b)


def _put_weights(buf: bytearray, op) -> None:
    w = np.ascontiguousarray(op.weight, dtype=np.int8).ravel()
    buf.extend(pack_sf4(w).tobytes() if op.wbits == 4 else w.tobytes())
    _pad4(buf)
    buf.extend(np.ascontiguousarray(op.bias, dtype="<i4").tobytes())


def dumps(graph: Graph) -> bytes:
    graph.validate()
    h, w, c = graph.input_shape
    buf = bytearray(_HEADER.pack(MAGIC, VERSION, len(graph.ops), h, w, c, len(graph.labels), 0))
    _put_str(buf, graph.name)
    for label in graph.labels:
        _put_str(buf, label)
    _pad4(buf)
    for op in graph.ops:
        if isinstance(op, Conv2d):
            cout, kh, kw, cin_g = op.weight.shape
            buf.extend(_OPHEAD.pack(OP_CONV, op.act, op.wbits, op.gain_exp))
            buf.extend(_CONV.pack(cout, kh, kw, cin_g, op.stride, op.pad, op.groups, 0))
            _put_weights(buf, op)
        elif isinstance(op, Dense):
            cout, cin = op.weight.shape
            buf.extend(_OPHEAD.pack(OP_DENSE, op.act, op.wbits, op.gain_exp))
            buf.extend(_DENSE.pack(cout, cin))
            _put_weights(buf, op)
        elif isinstance(op, GlobalAvgPool):
            buf.extend(_OPHEAD.pack(OP_GAP, 0, 0, 0))
            buf.extend(b"\0" * 12)
        else:
            raise TypeError(f"unknown op {type(op).__name__}")
    buf.extend(struct.pack("<I", zlib.crc32(buf)))
    return bytes(buf)


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ValueError("truncated .sfm")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def unpack(self, st: struct.Struct):
        return st.unpack(self.take(st.size))

    def string(self) -> str:
        n = self.take(1)[0]
        return self.take(n).decode("utf-8")

    def align4(self) -> None:
        self.pos += -self.pos % 4


def _get_weights(r: _Reader, shape, wbits: int, cout: int):
    n = int(np.prod(shape))
    if wbits == 4:
        w = unpack_sf4(np.frombuffer(r.take((n + 1) // 2), np.uint8), n)
    elif wbits == 8:
        w = np.frombuffer(r.take(n), np.int8).copy()
    else:
        raise ValueError(f"unsupported weight width SF{wbits}")
    r.align4()
    b = np.frombuffer(r.take(4 * cout), "<i4").astype(np.int32)
    return w.reshape(shape), b


def loads(data: bytes) -> Graph:
    if len(data) < _HEADER.size + 4:
        raise ValueError("truncated .sfm")
    body, (crc,) = data[:-4], struct.unpack("<I", data[-4:])
    if zlib.crc32(body) != crc:
        raise ValueError("bad .sfm checksum")
    r = _Reader(body)
    magic, version, n_ops, h, w, c, n_labels, _flags = r.unpack(_HEADER)
    if magic != MAGIC:
        raise ValueError("not an .sfm file")
    if version != VERSION:
        raise ValueError(f"unsupported .sfm version {version}")
    name = r.string()
    labels = [r.string() for _ in range(n_labels)]
    r.align4()
    ops = []
    for _ in range(n_ops):
        kind, act, wbits, gain_exp = r.unpack(_OPHEAD)
        if kind == OP_CONV:
            cout, kh, kw, cin_g, stride, pad, groups, _ = r.unpack(_CONV)
            wt, b = _get_weights(r, (cout, kh, kw, cin_g), wbits, cout)
            ops.append(Conv2d(wt, b, stride=stride, pad=pad, groups=groups,
                              gain_exp=gain_exp, act=act, wbits=wbits))
        elif kind == OP_DENSE:
            cout, cin = r.unpack(_DENSE)
            wt, b = _get_weights(r, (cout, cin), wbits, cout)
            ops.append(Dense(wt, b, gain_exp=gain_exp, act=act, wbits=wbits))
        elif kind == OP_GAP:
            r.take(12)
            ops.append(GlobalAvgPool())
        else:
            raise ValueError(f"unknown op type {kind}")
    if r.pos != len(body):
        raise ValueError("trailing bytes in .sfm")
    graph = Graph((h, w, c), ops, labels=labels, name=name)
    graph.validate()
    return graph


def save(graph: Graph, path) -> int:
    data = dumps(graph)
    Path(path).write_bytes(data)
    return len(data)


def load(path) -> Graph:
    return loads(Path(path).read_bytes())
