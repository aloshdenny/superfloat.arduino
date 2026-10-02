"""Frame <-> tile geometry, shared by dataset preparation and the device.

Training tiles and deployed tiles must be cut identically or the model is
evaluated on a distribution it never saw, so both sides use this module.

A frame is resized to cols x rows tiles of `tile` pixels and cut on a regular
grid. Distant smoke is often a handful of pixels wide, so the grid is fixed
rather than adaptive: a plume always lands in one or two tiles, and the
temporal voter (sentinel/temporal.py) can track it per tile across frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLEAR, SMOKE, FLAME = 0, 1, 2


@dataclass(frozen=True)
class TileGrid:
    cols: int = 5
    rows: int = 3
    tile: int = 128

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) the frame is resized to before cutting."""
        return self.cols * self.tile, self.rows * self.tile

    @property
    def count(self) -> int:
        return self.cols * self.rows

    def fit(self, frame: np.ndarray) -> np.ndarray:
        """Resize an HWC uint8 frame to the grid (area filter when shrinking)."""
        w, h = self.size
        if frame.shape[1] == w and frame.shape[0] == h:
            return frame
        try:
            import cv2

            return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        except ImportError:
            from PIL import Image

            return np.asarray(Image.fromarray(frame).resize((w, h), Image.BOX))

    def split(self, frame: np.ndarray) -> np.ndarray:
        """(H, W, C) frame at grid size -> (count, tile, tile, C), row-major."""
        w, h = self.size
        if frame.shape[:2] != (h, w):
            raise ValueError(f"frame is {frame.shape[1]}x{frame.shape[0]}, grid needs {w}x{h}")
        t, c = self.tile, frame.shape[2]
        return (frame.reshape(self.rows, t, self.cols, t, c)
                .transpose(0, 2, 1, 3, 4)
                .reshape(self.count, t, t, c))

    def cells(self) -> list[tuple[float, float, float, float]]:
        """Each tile's (x0, y0, x1, y1) in normalised frame coordinates."""
        return [(c / self.cols, r / self.rows, (c + 1) / self.cols, (r + 1) / self.rows)
                for r in range(self.rows) for c in range(self.cols)]


def _overlap(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def label_tiles(grid: TileGrid, boxes, min_tile_frac=0.04, min_box_frac=0.5, ignore_below=0.005):
    """Tile labels from normalised (cls, x0, y0, x1, y1) boxes.

    A tile is positive for a class when a box of that class covers at least
    `min_tile_frac` of the tile, or at least `min_box_frac` of the box lies in
    the tile (small, distant plumes). Flame outranks smoke. Tiles that only
    graze a box (more than `ignore_below` of the tile, but under both
    thresholds) are returned as None and should be left out of training:
    they are neither clean negatives nor convincing positives.

    `cls` uses the dataset's convention: SMOKE or FLAME.
    """
    labels = []
    for cell in grid.cells():
        cell_area = (cell[2] - cell[0]) * (cell[3] - cell[1])
        label, grazed = CLEAR, False
        for cls, *box in boxes:
            inter = _overlap(cell, box)
            if inter <= 0:
                continue
            box_area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-12)
            if inter >= min_tile_frac * cell_area or inter >= min_box_frac * box_area:
                label = max(label, int(cls))
            elif inter > ignore_below * cell_area:
                grazed = True
        labels.append(None if label == CLEAR and grazed else label)
    return labels


def yolo_to_corners(cx, cy, w, h):
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
