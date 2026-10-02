"""ctypes binding to the C runtime (runtime/build/libsfrt.*).

ctypes drops the GIL for the duration of a foreign call, so one Model can be
run from several threads at once: the weights are shared and read-only, and
each thread brings its own scratch buffer. On the UNO Q that is how the four
A53 cores split the tiles of a frame.
"""

from __future__ import annotations

import ctypes as C
import os
import sys
import threading
from pathlib import Path

import numpy as np

from . import sfm

_LIB_NAMES = {"darwin": "libsfrt.dylib", "win32": "sfrt.dll"}
_lib = None


def _default_lib_path() -> Path:
    name = _LIB_NAMES.get(sys.platform, "libsfrt.so")
    return Path(__file__).resolve().parent.parent / "runtime" / "build" / name


def load_library(path=None):
    """Load libsfrt once. SFEDGE_LIB overrides the default build location."""
    global _lib
    if _lib is not None:
        return _lib
    path = Path(path or os.environ.get("SFEDGE_LIB") or _default_lib_path())
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build it with `make -C runtime`")
    lib = C.CDLL(str(path))
    p8, vp, sz = C.POINTER(C.c_uint8), C.c_void_p, C.c_size_t
    lib.sf_model_sizeof.restype = sz
    lib.sf_model_arena_bytes.argtypes = [p8, sz, C.POINTER(sz)]
    lib.sf_model_load.argtypes = [vp, p8, sz, vp, sz]
    lib.sf_model_scratch_bytes.argtypes = [vp]
    lib.sf_model_scratch_bytes.restype = sz
    lib.sf_model_run.argtypes = [vp, vp, vp, vp, sz]
    lib.sf_model_n_out.argtypes = [vp]
    lib.sf_model_out_raw.argtypes = [vp]
    lib.sf_status_str.argtypes = [C.c_int]
    lib.sf_status_str.restype = C.c_char_p
    _lib = lib
    return lib


class SFRuntimeError(RuntimeError):
    pass


def _check(lib, rc: int, what: str) -> None:
    if rc != 0:
        raise SFRuntimeError(f"{what}: {lib.sf_status_str(rc).decode()}")


class Model:
    """A loaded .sfm model running on the C runtime."""

    def __init__(self, source, lib_path=None):
        lib = self._lib = load_library(lib_path)
        data = source if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
        self.graph = sfm.loads(bytes(data))  # parsed again in Python for metadata
        # The runtime references weights in place, so these buffers must
        # outlive the model. ctypes buffers are malloc-aligned (>= 8 bytes).
        self._buf = C.create_string_buffer(bytes(data), len(data))
        buf_p = C.cast(self._buf, C.POINTER(C.c_uint8))
        arena_n = C.c_size_t()
        _check(lib, lib.sf_model_arena_bytes(buf_p, len(data), C.byref(arena_n)), "arena size")
        self._arena = C.create_string_buffer(max(arena_n.value, 1))
        self._model = C.create_string_buffer(lib.sf_model_sizeof())
        _check(lib, lib.sf_model_load(self._model, buf_p, len(data), self._arena, arena_n.value), "load")
        self.scratch_bytes = lib.sf_model_scratch_bytes(self._model)
        self.n_out = lib.sf_model_n_out(self._model)
        self.raw_output = bool(lib.sf_model_out_raw(self._model))
        self._tls = threading.local()

    @property
    def input_shape(self):
        return tuple(self.graph.input_shape)

    @property
    def labels(self):
        return list(self.graph.labels)

    def _scratch(self):
        s = getattr(self._tls, "scratch", None)
        if s is None:
            s = self._tls.scratch = C.create_string_buffer(self.scratch_bytes)
        return s

    def run(self, x: np.ndarray) -> np.ndarray:
        """One SF8 input (int8, HWC) -> int32 outputs."""
        x = np.ascontiguousarray(x)
        if x.dtype != np.int8 or x.shape != self.input_shape:
            raise ValueError(f"expected int8 {self.input_shape}, got {x.dtype} {x.shape}")
        out = np.empty(self.n_out, dtype=np.int32)
        rc = self._lib.sf_model_run(self._model, x.ctypes.data, out.ctypes.data,
                                    self._scratch(), self.scratch_bytes)
        _check(self._lib, rc, "run")
        return out

    def run_rgb8(self, image: np.ndarray) -> np.ndarray:
        return self.run((np.asarray(image, dtype=np.uint8) ^ 0x80).view(np.int8))
