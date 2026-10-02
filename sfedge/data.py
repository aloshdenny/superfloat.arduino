"""Datasets for training. Images stay uint8 until the model's first layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _torch():
    import torch

    return torch


class TileFolder:
    """<root>/<label>/*.png tiles, as written by tools/prepare_tiles.py.

    Returns (uint8 HWC tensor, label index). Augmentation is deliberately
    mild and orientation-preserving: smoke rises, so vertical flips and
    rotations would teach the model plumes that do not exist.
    """

    def __init__(self, root, labels, augment=False, seed=0):
        self.root = Path(root)
        self.labels = list(labels)
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.items = []
        for idx, name in enumerate(self.labels):
            d = self.root / name
            if d.is_dir():
                self.items += [(p, idx) for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        if not self.items:
            raise FileNotFoundError(f"no tiles under {self.root}/{{{','.join(self.labels)}}}")

    def __len__(self):
        return len(self.items)

    def counts(self) -> np.ndarray:
        return np.bincount([y for _, y in self.items], minlength=len(self.labels))

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() < 0.5:
            img = img[:, ::-1]
        # haze, dusk and exposure changes: contrast about the mean, then brightness
        f = img.astype(np.float32)
        contrast = self.rng.uniform(0.7, 1.3)
        bright = self.rng.uniform(-25, 25)
        f = (f - f.mean()) * contrast + f.mean() + bright
        return np.clip(f, 0, 255).astype(np.uint8)

    def __getitem__(self, i):
        from PIL import Image

        path, label = self.items[i]
        img = np.asarray(Image.open(path).convert("RGB"))
        if self.augment:
            img = self._augment(img)
        return _torch().from_numpy(np.ascontiguousarray(img)), label


def class_weights(counts) -> np.ndarray:
    """Inverse-frequency weights normalised to mean 1 (empty classes get 0)."""
    counts = np.asarray(counts, dtype=np.float64)
    w = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
    nz = w[w > 0]
    return w / nz.mean() if nz.size else w
