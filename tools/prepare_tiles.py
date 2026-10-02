#!/usr/bin/env python3
"""Cut a YOLO-format smoke/fire dataset into labelled training tiles.

Works with D-Fire (https://github.com/gaiasd/DFireDataset) as distributed,
and with any dataset in the same layout:

    <root>/images/*.jpg
    <root>/labels/*.txt     one "cls cx cy w h" line per box, normalised

Output is one folder per class, which tools/train.py reads directly:

    <out>/clear/*.png  <out>/smoke/*.png  <out>/flame/*.png

Example:
    python tools/prepare_tiles.py --root data/dfire/train --out data/tiles/train \
        --class-map 0:smoke,1:flame
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfedge.models import SMOKE_LABELS  # noqa: E402
from sfedge.tiling import TileGrid, label_tiles, yolo_to_corners  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_class_map(spec: str) -> dict[int, int]:
    out = {}
    for item in spec.split(","):
        src, name = item.split(":")
        out[int(src)] = SMOKE_LABELS.index(name.strip())
    return out


def read_boxes(path: Path, class_map: dict[int, int]):
    boxes = []
    if not path.exists():
        return boxes  # D-Fire ships negatives without a label file
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        if cls in class_map:
            boxes.append((class_map[cls], *yolo_to_corners(*map(float, parts[1:]))))
    return boxes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--class-map", default="0:smoke,1:flame",
                   help="dataset class id -> label, e.g. 0:smoke,1:flame")
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--tile", type=int, default=128)
    p.add_argument("--clear-keep", type=float, default=0.25,
                   help="fraction of clear tiles to keep; they outnumber positives ~20:1")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    grid = TileGrid(args.cols, args.rows, args.tile)
    class_map = parse_class_map(args.class_map)
    rng = random.Random(args.seed)
    for name in SMOKE_LABELS:
        (args.out / name).mkdir(parents=True, exist_ok=True)

    images = sorted(f for f in (args.root / "images").iterdir() if f.suffix.lower() in IMAGE_EXTS)
    counts = Counter()
    for i, img_path in enumerate(images):
        boxes = read_boxes(args.root / "labels" / (img_path.stem + ".txt"), class_map)
        frame = grid.fit(np.asarray(Image.open(img_path).convert("RGB")))
        for t, (tile, label) in enumerate(zip(grid.split(frame), label_tiles(grid, boxes))):
            if label is None:
                counts["ignored"] += 1
                continue
            if label == 0 and rng.random() > args.clear_keep:
                continue
            name = SMOKE_LABELS[label]
            Image.fromarray(tile).save(args.out / name / f"{img_path.stem}_t{t:02d}.png")
            counts[name] += 1
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(images)} images  {dict(counts)}", flush=True)
    print(f"done: {len(images)} images -> {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
