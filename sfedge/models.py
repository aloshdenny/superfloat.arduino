"""The two networks the sentinel runs.

SmokeNet: RGB tile classifier on the UNO Q's Linux side (Cortex-A53). A
frame is cut into tiles and each tile is classified as clear / smoke / flame,
the tile approach SmokeyNet (Dewangan et al., 2022) uses on HPWREN imagery,
cut down to plain 3x3 convs with stride-2 downsampling. Plain convs rather than
depthwise-separable blocks on purpose: the int8 dot-product kernel is most
efficient on long contiguous reductions (a full 3 x 3 x C kernel row), and
BN-folded depthwise filters are the layers most likely to need a large
gain exponent.

EmberNet: thermal hotspot classifier on the STM32U585 (Cortex-M33), fed by a
32 x 24 MLX90640 array. Two input channels: the current frame and its
difference from the previous one, so flicker counts as evidence and a
sun-warmed rock does not.

Both are plain chains of SFConv2d -> SFGlobalAvgPool -> SFLinear, which is
the shape the exporter and the runtimes accept.
"""

from __future__ import annotations

import torch.nn as nn

from .qat import SFConv2d, SFGlobalAvgPool, SFLinear

SMOKE_LABELS = ["clear", "smoke", "flame"]
EMBER_LABELS = ["ambient", "hotspot"]


class SFChain(nn.Sequential):
    """nn.Sequential with the metadata the exporter needs."""

    def __init__(self, *layers, input_shape, labels, name):
        super().__init__(*layers)
        self.input_shape = tuple(input_shape)  # (H, W, C), as the runtime sees it
        self.labels = list(labels)
        self.model_name = name


def _scaled(c: int, width: float) -> int:
    return max(8, int(round(c * width / 8)) * 8)


def smokenet(width: float = 1.0, wbits: int = 8, tile: int = 128) -> SFChain:
    """~28 M MACs and ~275 k weights at width 1.0 on a 128 x 128 tile.

    Five stride-2 stages take 128 -> 4, so the global pool averages 16 values
    and its division is a shift.
    """
    c = [_scaled(x, width) for x in (16, 24, 48, 96, 128)]
    layers = [
        SFConv2d(3, c[0], stride=2, wbits=wbits),
        SFConv2d(c[0], c[1], stride=2, wbits=wbits),
        SFConv2d(c[1], c[1], wbits=wbits),
        SFConv2d(c[1], c[2], stride=2, wbits=wbits),
        SFConv2d(c[2], c[2], wbits=wbits),
        SFConv2d(c[2], c[3], stride=2, wbits=wbits),
        SFConv2d(c[3], c[3], wbits=wbits),
        SFConv2d(c[3], c[4], stride=2, wbits=wbits),
        SFGlobalAvgPool(),
        SFLinear(c[4], len(SMOKE_LABELS), act="raw", wbits=wbits),
    ]
    return SFChain(*layers, input_shape=(tile, tile, 3), labels=SMOKE_LABELS,
                   name=f"smokenet-w{width:g}-sf{wbits}")


def embernet(wbits: int = 8) -> SFChain:
    """~0.66 M MACs, ~15 k weights: comfortable on a 160 MHz Cortex-M33."""
    layers = [
        SFConv2d(2, 8, wbits=wbits),
        SFConv2d(8, 16, stride=2, wbits=wbits),
        SFConv2d(16, 32, stride=2, wbits=wbits),
        SFConv2d(32, 32, stride=2, wbits=wbits),
        SFGlobalAvgPool(),
        SFLinear(32, len(EMBER_LABELS), act="raw", wbits=wbits),
    ]
    return SFChain(*layers, input_shape=(24, 32, 2), labels=EMBER_LABELS,
                   name=f"embernet-sf{wbits}")


MODELS = {"smokenet": smokenet, "embernet": embernet}
