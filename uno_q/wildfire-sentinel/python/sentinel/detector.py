"""Frame -> per-tile clear/smoke/flame probabilities on the C runtime."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from sfedge.export import logits_scale
from sfedge.runtime import Model
from sfedge.tiling import TileGrid


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class TileDetector:
    def __init__(self, model_path, grid: TileGrid, threads: int = 4, lib=None, mask=()):
        self.model = Model(model_path, lib_path=lib)
        if self.model.input_shape != (grid.tile, grid.tile, 3):
            raise ValueError(f"model expects {self.model.input_shape}, grid makes "
                             f"{grid.tile}x{grid.tile}x3 tiles")
        if not self.model.raw_output:
            raise ValueError("expected a model with raw logits as output")
        self.grid = grid
        bad = [t for t in mask if not 0 <= t < grid.count]
        if bad:
            raise ValueError(f"mask_tiles {bad} outside the {grid.count}-tile grid")
        self.active = [t for t in range(grid.count) if t not in set(mask)]
        self.scale = logits_scale(self.model.graph)
        self.labels = self.model.labels
        # One worker per A53 core. The runtime releases the GIL, so these run
        # truly in parallel.
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="tile")
        self.last_ms = 0.0

    def classify(self, frame_rgb: np.ndarray) -> np.ndarray:
        """(n_tiles, 3) probabilities; masked tiles are reported as certain clear."""
        t0 = time.perf_counter()
        tiles = self.grid.split(self.grid.fit(frame_rgb))
        probs = np.tile([1.0, 0.0, 0.0], (self.grid.count, 1))
        if self.active:
            raw = np.stack(list(self.pool.map(self.model.run_rgb8, tiles[self.active])))
            probs[self.active] = softmax(raw.astype(np.float64) * self.scale)
        self.last_ms = (time.perf_counter() - t0) * 1e3
        return probs

    def close(self):
        self.pool.shutdown(wait=False)
