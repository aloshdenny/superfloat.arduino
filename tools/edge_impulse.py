#!/usr/bin/env python3
"""Edge Impulse integration: data in, baseline models out.

Edge Impulse is where the field dataset lives and gets labelled, and its
int8 pipeline is the baseline the SuperFloat runtime is measured against on
the UNO Q. Needs `pip install edgeimpulse requests` and an API key in
EI_API_KEY (project dashboard -> Keys).

    # push prepared tiles (tools/prepare_tiles.py output) into the project
    python tools/edge_impulse.py upload data/tiles/train --category training
    python tools/edge_impulse.py upload data/tiles/val --category testing

    # write the trained model's float twin as ONNX (Bring Your Own Model)
    python tools/edge_impulse.py onnx runs/smoke-sf8/model.pt --out runs/smoke-sf8/twin.onnx

    # on-device latency/memory estimates, and an .eim to run on the board
    python tools/edge_impulse.py devices
    python tools/edge_impulse.py profile runs/smoke-sf8/twin.onnx --device <id from `devices`>
    python tools/edge_impulse.py deploy runs/smoke-sf8/twin.onnx --out runs/smoke-sf8/ei

`deploy` defaults to the runner-linux-aarch64 target, the .eim build App Lab
uses for the UNO Q; pass --quantize to let Edge Impulse produce its int8
version from representative tiles.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfedge.labels import SMOKE_LABELS  # noqa: E402

INGESTION = "https://ingestion.edgeimpulse.com/api/{category}/files"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def api_key() -> str:
    key = os.environ.get("EI_API_KEY")
    if not key:
        sys.exit("set EI_API_KEY to the project's API key")
    return key


def cmd_upload(args) -> int:
    """Labelled upload through the ingestion API, one request per batch."""
    import requests

    key = api_key()
    total = 0
    for label in SMOKE_LABELS:
        files = sorted(p for p in (args.tiles / label).glob("*") if p.suffix.lower() in IMAGE_EXTS)
        for i in range(0, len(files), args.batch):
            chunk = files[i:i + args.batch]
            payload = [("data", (p.name, p.read_bytes(), "image/png")) for p in chunk]
            r = requests.post(INGESTION.format(category=args.category), files=payload, timeout=120,
                              headers={"x-api-key": key, "x-label": label, "x-disallow-duplicates": "1"})
            if r.status_code != 200:
                print(f"{label}: upload failed ({r.status_code}): {r.text[:200]}", file=sys.stderr)
                return 1
            total += len(chunk)
        print(f"{label}: {len(files)} tiles")
    print(f"uploaded {total} tiles to {args.category}")
    return 0


def cmd_onnx(args) -> int:
    import torch

    from sfedge.models import smokenet
    from sfedge.qat import quantize_model
    from sfedge.twin import export_onnx

    model = quantize_model(smokenet(width=args.width, wbits=args.wbits, tile=args.tile).eval())
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    export_onnx(model.eval(), args.out)
    print(f"wrote {args.out}")
    return 0


def _ei():
    import edgeimpulse as ei

    ei.API_KEY = api_key()
    return ei


def cmd_devices(args) -> int:
    ei = _ei()
    print("profile devices:", ", ".join(ei.model.list_profile_devices()))
    print("deployment targets:", ", ".join(ei.model.list_deployment_targets()))
    return 0


def cmd_profile(args) -> int:
    ei = _ei()
    result = ei.model.profile(model=str(args.model), device=args.device)
    print(result.summary())
    return 0


def _representative(tiles: Path, n: int) -> np.ndarray:
    from PIL import Image

    paths = sorted(p for label in SMOKE_LABELS for p in (tiles / label).glob("*.png"))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    imgs = [np.asarray(Image.open(paths[i]).convert("RGB"), dtype=np.float32) for i in pick]
    return np.stack(imgs).transpose(0, 3, 1, 2)  # NCHW, 0..255, matching the twin


def cmd_deploy(args) -> int:
    ei = _ei()
    rep = None
    if args.quantize:
        if not args.tiles:
            sys.exit("--quantize needs --tiles for representative data")
        rep = args.out / "representative.npy"
        args.out.mkdir(parents=True, exist_ok=True)
        np.save(rep, _representative(args.tiles, 200))
    args.out.mkdir(parents=True, exist_ok=True)
    ei.model.deploy(
        model=str(args.model),
        model_output_type=ei.model.output_type.Classification(labels=SMOKE_LABELS),
        model_input_type=ei.model.input_type.ImageInput(scaling_range="0..255"),
        representative_data_for_quantization=str(rep) if rep else None,
        deploy_model_type="int8" if args.quantize else "float32",
        deploy_target=args.target,
        output_directory=str(args.out),
    )
    print(f"deployment written to {args.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("upload", help="upload labelled tiles")
    s.add_argument("tiles", type=Path)
    s.add_argument("--category", choices=["training", "testing"], default="training")
    s.add_argument("--batch", type=int, default=50)
    s.set_defaults(fn=cmd_upload)

    s = sub.add_parser("onnx", help="write a trained SmokeNet's float twin as ONNX")
    s.add_argument("checkpoint", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--width", type=float, default=1.0)
    s.add_argument("--wbits", type=int, default=8)
    s.add_argument("--tile", type=int, default=128)
    s.set_defaults(fn=cmd_onnx)

    s = sub.add_parser("devices", help="list profiling devices and deployment targets")
    s.set_defaults(fn=cmd_devices)

    s = sub.add_parser("profile", help="estimate latency and memory on a target")
    s.add_argument("model", type=Path)
    s.add_argument("--device")
    s.set_defaults(fn=cmd_profile)

    s = sub.add_parser("deploy", help="build a deployable (.eim for the UNO Q by default)")
    s.add_argument("model", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--target", default="runner-linux-aarch64")
    s.add_argument("--quantize", action="store_true", help="int8 via Edge Impulse")
    s.add_argument("--tiles", type=Path, help="tile folder for representative data")
    s.set_defaults(fn=cmd_deploy)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
