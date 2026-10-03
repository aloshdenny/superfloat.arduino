#!/usr/bin/env python3
"""Evaluate an exported .sfm on a tile folder through the C runtime.

This is the deployed number: the same bytes, the same kernels, the same
rounding as the board. The training loop's validation figures should match
it exactly (see tests/test_qat_export.py); if they ever differ, trust this.

    python tools/eval.py runs/smoke-sf8/smokenet-w1-sf8.sfm data/tiles/test
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfedge.metrics import confusion, format_report, report  # noqa: E402
from sfedge.runtime import Model  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", type=Path)
    p.add_argument("tiles", type=Path)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--json", type=Path, help="also write the report here")
    args = p.parse_args(argv)

    model = Model(args.model)
    labels = model.labels
    items = [(f, i) for i, name in enumerate(labels) if (args.tiles / name).is_dir()
             for f in sorted((args.tiles / name).iterdir()) if f.suffix.lower() in IMAGE_EXTS]
    if not items:
        print(f"no tiles under {args.tiles}/{{{','.join(labels)}}}", file=sys.stderr)
        return 1
    h, w, _ = model.input_shape

    def predict(path: Path) -> int:
        img = np.asarray(Image.open(path).convert("RGB"))
        if img.shape[:2] != (h, w):
            raise ValueError(f"{path} is {img.shape[1]}x{img.shape[0]}, model wants {w}x{h}")
        return int(np.argmax(model.run_rgb8(img)))

    with ThreadPoolExecutor(args.threads) as pool:
        preds = list(pool.map(predict, [f for f, _ in items]))
    rep = report(confusion([y for _, y in items], preds, len(labels)), labels)
    print(f"{model.graph.name} on {len(items)} tiles")
    print(format_report(rep))
    if args.json:
        args.json.write_text(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
