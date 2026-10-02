"""Per-tile evidence accumulation: turns noisy frame scores into alerts.

A single frame is weak evidence. Fog banks, dust, low cloud and lens glare
all produce smoke-like tiles for a frame or two; a plume persists and grows.
So an alert needs k hits in the last n frames for the same tile, plus a
smoothed score above threshold. Flame is held to a shorter window than
smoke because its false-positive modes (sunset glints, red vehicles) are
brief and its cost of delay is higher.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

CLEAR, SMOKE, FLAME = 0, 1, 2
KIND = {SMOKE: "smoke", FLAME: "flame", CLEAR: "clear"}


@dataclass
class VoterConfig:
    hit_threshold: float = 0.5  # per-frame hazard probability counted as a hit
    smoke_k: int = 4
    smoke_n: int = 6
    flame_k: int = 2
    flame_n: int = 3
    ema_alpha: float = 0.3
    ema_threshold: float = 0.4
    clear_frames: int = 15  # consecutive quiet frames before an all-clear
    cooldown_s: float = 600.0  # no repeat alert for the same tile and level


@dataclass
class Event:
    kind: str  # "smoke" | "flame" | "clear"
    tile: int
    score: float
    t: float


@dataclass
class _Tile:
    smoke_hits: deque
    flame_hits: deque
    ema: np.ndarray = field(default_factory=lambda: np.zeros(3))
    level: int = CLEAR
    quiet: int = 0
    last_alert: dict = field(default_factory=dict)


class TemporalVoter:
    def __init__(self, n_tiles: int, cfg: VoterConfig | None = None):
        self.cfg = cfg or VoterConfig()
        self.tiles = [
            _Tile(deque(maxlen=self.cfg.smoke_n), deque(maxlen=self.cfg.flame_n))
            for _ in range(n_tiles)
        ]

    @property
    def levels(self) -> list[int]:
        return [t.level for t in self.tiles]

    def update(self, probs: np.ndarray, t: float) -> list[Event]:
        """probs: (n_tiles, 3) clear/smoke/flame probabilities for one frame."""
        cfg = self.cfg
        probs = np.asarray(probs, dtype=np.float64)
        if probs.shape != (len(self.tiles), 3):
            raise ValueError(f"expected ({len(self.tiles)}, 3) probabilities, got {probs.shape}")
        events = []
        for i, (tile, p) in enumerate(zip(self.tiles, probs)):
            tile.ema = (1 - cfg.ema_alpha) * tile.ema + cfg.ema_alpha * p
            hazard = p[SMOKE] + p[FLAME]
            tile.smoke_hits.append(hazard >= cfg.hit_threshold)
            tile.flame_hits.append(p[FLAME] >= cfg.hit_threshold)

            level = CLEAR
            if sum(tile.flame_hits) >= cfg.flame_k and tile.ema[FLAME] >= cfg.ema_threshold:
                level = FLAME
            elif (sum(tile.smoke_hits) >= cfg.smoke_k
                  and tile.ema[SMOKE] + tile.ema[FLAME] >= cfg.ema_threshold):
                level = SMOKE

            if level > tile.level:
                last = tile.last_alert.get(level)
                tile.level = level
                tile.quiet = 0
                if last is None or t - last >= cfg.cooldown_s:
                    tile.last_alert[level] = t
                    score = float(tile.ema[SMOKE] + tile.ema[FLAME]) if level == SMOKE else float(tile.ema[FLAME])
                    events.append(Event(KIND[level], i, score, t))
            elif tile.level != CLEAR:
                tile.quiet = 0 if hazard >= cfg.hit_threshold else tile.quiet + 1
                if tile.quiet >= cfg.clear_frames:
                    tile.level = CLEAR
                    events.append(Event("clear", i, float(hazard), t))
        return events
