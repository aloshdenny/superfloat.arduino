"""Thermal input encoding and a synthetic bootstrap dataset for EmberNet.

The encoding is what the MCU computes from MLX90640 frames (degrees C), so it
is linear with integer-friendly constants; sketch.ino implements the same two
lines in C:

    ch0 = clamp(round((T - 25) * 127 / 100))          absolute temperature
    ch1 = clamp(round((T - T_prev) * 127 / 25))       frame-to-frame change

Everything above 125 C saturates ch0, which is fine: the network does not need
to tell a 300 C flame from a 600 C one.

The synthetic generator exists to exercise the pipeline and to give the MCU
a first model before field data exists. It is not a substitute for real
MLX90640 captures (see docs/roadmap.md): its negatives are only as hard as
what is modelled here (sun-warmed ground, warm static objects, sensor noise).
"""

from __future__ import annotations

import numpy as np

H, W = 24, 32


def encode(t_now: np.ndarray, t_prev: np.ndarray) -> np.ndarray:
    """Two (24, 32) temperature frames -> (24, 32, 2) int8 SF8 codes."""
    a = np.rint((np.asarray(t_now, np.float64) - 25.0) * 127.0 / 100.0)
    d = np.rint((np.asarray(t_now, np.float64) - np.asarray(t_prev, np.float64)) * 127.0 / 25.0)
    return np.clip(np.stack([a, d], axis=-1), -127, 127).astype(np.int8)


def _blob(rng, cy, cx, sigma, amp):
    yy, xx = np.mgrid[0:H, 0:W]
    return amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))


def _background(rng):
    ambient = rng.uniform(-5, 42)
    gy = rng.uniform(-8, 8) * np.linspace(-1, 1, H)[:, None]  # sky/ground gradient
    t = ambient + gy + np.zeros((H, W))
    for _ in range(rng.integers(0, 4)):  # sun-warmed rocks, roofs, vehicles
        t += _blob(rng, rng.uniform(0, H), rng.uniform(0, W), rng.uniform(1, 5), rng.uniform(5, 45))
    return t


def sample(rng, hotspot: bool):
    """One (t_prev, t_now) pair."""
    base = _background(rng)
    prev = base + rng.normal(0, 0.4, (H, W))
    now = base + rng.normal(0, 0.4, (H, W)) + rng.normal(0, 0.3)  # noise + global drift
    if hotspot:
        cy, cx = rng.uniform(0, H), rng.uniform(0, W)
        sigma = rng.uniform(0.4, 2.0)  # distant ignitions are sub-pixel to a few pixels
        peak = rng.uniform(60, 500) * min(1.0, sigma)  # partial pixel fill dilutes small fires
        flicker = rng.uniform(0.5, 1.0)
        prev = prev + _blob(rng, cy, cx, sigma, peak * flicker)
        now = now + _blob(rng, cy + rng.normal(0, 0.3), cx + rng.normal(0, 0.3), sigma, peak)
    return prev, now


def synthetic_dataset(n: int, positive_frac=0.5, seed=0):
    """(n, 24, 32, 2) int8 codes and (n,) labels (0 ambient, 1 hotspot)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < positive_frac).astype(np.int64)
    x = np.empty((n, H, W, 2), np.int8)
    for i in range(n):
        prev, now = sample(rng, bool(y[i]))
        x[i] = encode(now, prev)
    return x, y
