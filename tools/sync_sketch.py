#!/usr/bin/env python3
"""Copy the C runtime into the App Lab sketch, and optionally a model header.

Arduino builds compile everything under sketch/src/, but cannot reach files
outside the sketch folder, so the runtime is vendored there. CI runs this
with --check to make sure the copy never drifts from runtime/.

    python tools/sync_sketch.py                      # refresh the runtime copy
    python tools/sync_sketch.py --header runs/ember/embernet-sf8.h
    python tools/sync_sketch.py --check              # exit 1 if out of date
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKETCH = ROOT / "uno_q" / "wildfire-sentinel" / "sketch"
FILES = {
    ROOT / "runtime" / "include" / "sf_runtime.h": SKETCH / "src" / "sfrt" / "sf_runtime.h",
    ROOT / "runtime" / "src" / "sf_kernels.c": SKETCH / "src" / "sfrt" / "sf_kernels.c",
    ROOT / "runtime" / "src" / "sf_model.c": SKETCH / "src" / "sfrt" / "sf_model.c",
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    p.add_argument("--header", type=Path, help="generated EmberNet header to install as ember_model.h")
    args = p.parse_args(argv)

    stale = [dst for src, dst in FILES.items()
             if not dst.exists() or dst.read_bytes() != src.read_bytes()]
    if args.check:
        for dst in stale:
            print(f"out of date: {dst.relative_to(ROOT)}")
        return 1 if stale else 0
    for src, dst in FILES.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    if args.header:
        text = args.header.read_text()
        # the sketch refers to the model by fixed names
        import re

        m = re.search(r"static const size_t (\w+)_len", text)
        if not m:
            print("not a header written by sfedge.export", file=sys.stderr)
            return 1
        text = text.replace(m.group(1), "ember_sfm")
        (SKETCH / "ember_model.h").write_text(text)
    print(f"synced {len(FILES)} runtime files" + (" and model header" if args.header else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
