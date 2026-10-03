"""Sentinel settings: defaults here, overrides from a JSON file.

The file is looked up at $SENTINEL_CONFIG, then /app/config.json (the app
folder as mounted in the App Lab container). Unknown keys are rejected so a
typo in the field cannot silently fall back to a default.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from .temporal import VoterConfig

DEFAULT_PATHS = ("/app/config.json",)


@dataclass
class Config:
    node_id: str = "sentinel-01"
    lat: float | None = None
    lon: float | None = None

    model: str = "/app/models/smokenet.sfm"
    lib: str | None = None  # libsfrt.so; None -> SFEDGE_LIB or the vendored copy
    threads: int = 4

    camera: str | None = None  # None: App Lab auto-detect (USB first, then CSI)
    resolution: tuple[int, int] = (1280, 720)
    camera_bgr: bool = True  # OpenCV-backed cameras deliver BGR
    period_s: float = 2.0  # one frame every period_s; smoke moves slowly
    # Smoke is invisible in the dark, so the camera tier slows down at night
    # and the thermal tier (MCU) carries detection. Darkness is judged from
    # the frame itself, so no clock, timezone or sun calculation is needed.
    night_period_s: float = 10.0
    night_luma: float = 25.0  # mean luma (0-255) below which a frame is "night"

    grid_cols: int = 5
    grid_rows: int = 3
    tile: int = 128
    # Tiles never scored and always reported clear: homes, roads, trails, or
    # anything else in view that is not the landscape being watched.
    mask_tiles: list[int] = field(default_factory=list)

    voter: VoterConfig = field(default_factory=VoterConfig)

    queue_path: str = "/app/data/alerts.jsonl"
    webhook_url: str | None = None
    webhook_token: str | None = None
    status_port: int = 7000


def load_config(path=None) -> Config:
    candidates = [path, os.environ.get("SENTINEL_CONFIG"), *DEFAULT_PATHS]
    for p in candidates:
        if p and Path(p).exists():
            return from_dict(json.loads(Path(p).read_text()))
    return Config()


def from_dict(d: dict) -> Config:
    known = {f.name for f in fields(Config)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    d = dict(d)
    if "voter" in d:
        vknown = {f.name for f in fields(VoterConfig)}
        bad = set(d["voter"]) - vknown
        if bad:
            raise ValueError(f"unknown voter keys: {sorted(bad)}")
        d["voter"] = VoterConfig(**d["voter"])
    if "resolution" in d:
        d["resolution"] = tuple(d["resolution"])
    return Config(**d)
