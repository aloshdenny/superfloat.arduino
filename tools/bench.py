#!/usr/bin/env python3
"""Latency of an .sfm model on the C runtime, single tile and full frame.

    python tools/bench.py runs/smoke-sf8/smokenet-w1-sf8.sfm --tiles 15 --threads 4
    python tools/bench.py --arch smokenet          # untrained weights: timing only

Latency does not depend on the weight values, so --arch is enough to size the
hardware before a model is trained.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfedge import sfm  # noqa: E402
from sfedge.runtime import Model  # noqa: E402


def arch_bytes(name: str, wbits: int) -> bytes:
    from sfedge.export import to_graph
    from sfedge.models import MODELS
    from sfedge.qat import quantize_model

    kw = {"wbits": wbits}
    return sfm.dumps(to_graph(quantize_model(MODELS[name](**kw).eval())))


def timeit(fn, warmup: int, reps: int) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?", type=Path)
    p.add_argument("--arch", choices=["smokenet", "embernet"])
    p.add_argument("--wbits", type=int, default=8, choices=[4, 8])
    p.add_argument("--tiles", type=int, default=15, help="tiles per frame (5x3 grid by default)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--reps", type=int, default=20)
    args = p.parse_args(argv)
    if bool(args.model) == bool(args.arch):
        p.error("give either an .sfm path or --arch")

    data = args.model.read_bytes() if args.model else arch_bytes(args.arch, args.wbits)
    model = Model(data)
    g = model.graph
    rng = np.random.default_rng(0)
    xs = [rng.integers(-128, 128, model.input_shape).astype(np.int8) for _ in range(args.tiles)]

    one = timeit(lambda: model.run(xs[0]), 3, args.reps)
    with ThreadPoolExecutor(args.threads) as pool:
        frame = timeit(lambda: list(pool.map(model.run, xs)), 2, max(3, args.reps // 4))

    med = statistics.median(one)
    print(f"host      {platform.machine()} / {platform.processor() or platform.system()}")
    print(f"model     {g.name}: {g.macs() / 1e6:.2f} M MACs, {g.weight_bytes() / 1024:.1f} KiB weights, "
          f"{len(data) / 1024:.1f} KiB file")
    print(f"tile      median {med:.2f} ms  ({g.macs() / med / 1e6:.2f} GMAC/s, 1 thread)")
    print(f"frame     {args.tiles} tiles on {args.threads} threads: median {statistics.median(frame):.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
