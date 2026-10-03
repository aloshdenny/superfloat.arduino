# Integer-only SuperFloat inference

This page explains how a SuperFloat-trained CNN runs on the UNO Q without
floating point, and why the accuracy measured in training is exactly the
accuracy that ships.

## The format

SuperFloat (SFx, [aloshdenny/superfloat](https://github.com/aloshdenny/superfloat))
spends 1 bit on the sign and every other bit on the significand. It has no
exponent and no integer bit. That makes SFx the signed fixed-point format
Q1.(x−1):

| Format | Scale | Range | Used here for |
| --- | --- | --- | --- |
| SF16 | 2^15 | ±0.99997 | (not used on device) |
| SF8 | 2^7 | ±0.99219 | all activations; weights by default |
| SF4 | 2^3 | ±0.875 | weights, when flash is tight |

SF4 is a subset of SF8: k/8 = 16k/128. So the runtime keeps one int8 kernel and
widens SF4 weights with a 4-bit left shift at load time.

## One layer, in integers

Take a conv layer with BatchNorm folded in. Its weight codes are `w` (SF8,
int8), its input activation codes are `a` (SF8, int8), its bias is `b` (int32)
and its gain exponent is `e` (see below).

```
acc  = b + Σ w·a                                  int32
out  = clamp((acc + 2^(6−e)) >> (7−e), lo, 127)   int8, SF8
```

- `lo` is 0 for the clipped ReLU used between layers and −127 for a linear
  output.
- The last layer keeps `acc` as raw int32 logits.

The arithmetic is int8 multiplies, int32 adds, one rounding shift and one
clamp. Compare this with standard int8 affine quantisation (TFLite and similar),
which needs, for each tensor:

- a zero point,
- a scale,
- a requantisation multiplier, which is per output channel when weights are
  per-channel quantised (a Q31 multiply plus shift).

SuperFloat needs none of these. Every scale is a power of two that is fixed by
the format, so they all collapse into a shift.

## The gain exponent

BatchNorm folding multiplies each output channel's weights by γ/σ, which can
push folded weights past ±1. Rather than clipping them, the exporter picks the
smallest `e ≥ 0` such that `max|w_folded| / 2^e` fits SF8, and stores `w/2^e`.

- It is one small integer per layer, folded into the output shift above.
- It is not a per-element exponent and not a multiplier.
- In practice `e` is 0 to 2.

## Input encoding

Camera pixels are uint8 `p`, and the network sees `(p − 128) / 128`. In SF8
codes that is `p − 128`, which is the same as flipping the top bit:
`(int8)(p ^ 0x80)`. Normalisation costs one XOR per byte.

## Why training accuracy is deployed accuracy

In the QAT phase every weight the forward pass uses is a multiple of
2^e/128, and every activation is a multiple of 1/128. So:

- every product in a convolution is an integer number of accumulator units
  (2^(e−14)), and so is every partial sum;
- for the layer sizes used here (fan-in up to 864, codes up to 127),
  |partial sums| stay below 2^24 units;
- float32 represents every integer below 2^24 exactly.

So the float32 convolution in PyTorch performs no rounding at all, in any
summation order, on CPU or GPU (with TF32 disabled, which `tools/train.py`
does). Activation rounding is implemented as `floor(x·128 + 0.5)`. That matches
the runtime's round-half-up shift, including on ties, which occur in about one
in 2^(7−e) values and so are not rare.

`tests/test_qat_export.py` checks the consequence bit for bit. The torch logits,
the numpy reference executor and the C runtime agree exactly, on SmokeNet with
SF8 and SF4 weights and on EmberNet, on random weights with non-trivial
BatchNorm statistics.

## What is and is not claimed

- **Exactness** is a property of the arithmetic, checked by tests on macOS
  (arm64) and Linux aarch64 (GCC 14); CI adds Linux x86_64 for the scalar path.
- **Speed on the UNO Q** has not been measured yet; the boards ship to
  semi-finalists on Oct 12.
  - Host numbers from `tools/bench.py` are in the README, labelled as host
    numbers.
  - The A53 projection is an estimate: 28 M MACs per 128×128 tile at a
    conservative 1–2 GMAC/s per core (ARMv8.0 has no SDOT, so the int8 path
    is `vmull_s8` + `vpadalq_s16`). That gives roughly 15–30 ms per tile, or
    60–120 ms for a 15-tile frame across four cores.
- **Accuracy** on real smoke imagery comes after training on D-Fire and HPWREN
  data in stage 2 (docs/roadmap.md).
  - The SuperFloat repository reports that SF8 weights match FP32 on EuroSAT
    classification (96.4 vs 96.3).
  - It also reports that activation clamping is nearly free in BatchNorm CNNs,
    which is the regime used here.
  - Whether that holds for this small network on this task is what stage 2
    measures.
