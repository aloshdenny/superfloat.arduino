# superfloat.arduino

Integer-only [SuperFloat](https://github.com/aloshdenny/superfloat) inference
for the Arduino UNO Q, and the application it was built for: **Wildfire
Sentinel**, an off-grid node that detects wildfire smoke and ignitions on the
board itself and never sends an image anywhere.

SuperFloat (SFx) drops the exponent: a value is a sign plus fraction bits, i.e.
Q1.(x−1) fixed point. Every SFx scale is a power of two, so a CNN trained on the
SF grid runs on integers alone:

- int8 multiplies;
- int32 accumulation;
- one rounding shift and a clamp per layer.

There are no zero points, no per-channel float scales and no requantisation
multipliers. The quantisation-aware training is built so that the PyTorch forward
pass equals the C runtime **bit for bit**, so the accuracy you measure in
training is the accuracy that ships.

```
 camera ─► 5x3 tiles ─► SmokeNet (SF8, A53 x4, NEON) ─► temporal voter ─► alert queue ─► webhook
                                                                │
 MLX90640 ─► EmberNet (SF8, Cortex-M33) ─► buzzer, LED matrix ◄─┘   watchdog keeps the MCU tier
                                                                    alive if Linux goes down
```

## Status

| | |
| --- | --- |
| C runtime: allocator-free, NEON on aarch64, scalar elsewhere | done; bit-exact with the reference on macOS arm64 and Linux aarch64 (CI adds x86_64) |
| QAT on the SF grid with BatchNorm folding, export to `.sfm` and C headers | done; torch == runtime, bit for bit (`tests/test_qat_export.py`) |
| App Lab app: camera tiling, parallel inference, temporal voting, store-and-forward alerts, status page | done; runs end to end on a laptop in `--replay` mode |
| MCU sketch: tile map, buzzer, mute, Linux watchdog, EmberNet thermal tier | done; compiled and exercised on host against stub Arduino APIs |
| Edge Impulse: dataset upload, ONNX twin, profiling, `.eim` baseline | done; needs an API key |
| Hardware: BOM, power budget, wiring, printable mount | done |
| On-board measurements (latency, power) | **not yet**: UNO Q kits ship Oct 12 |
| SmokeNet trained on real smoke imagery | **not yet**: stage 2 (D-Fire, HPWREN) |
| EmberNet trained on real thermal frames | **not yet**: the bundled model is a synthetic-data bootstrap |

See [docs/roadmap.md](docs/roadmap.md) for the stage 2 plan.

## Models

| Model | Runs on | Input | Compute | Weights (SF8 / SF4) |
| --- | --- | --- | --- | --- |
| SmokeNet | QRB2210, 4x Cortex-A53 | 128x128 RGB tile | 28.3 M MACs | 269 KiB / 135 KiB |
| EmberNet | STM32U585, Cortex-M33 | 32x24 thermal + frame delta | 0.66 M MACs | 15 KiB / 7.4 KiB |

Host timing from `tools/bench.py`. **These are Apple M4 numbers, not UNO Q
numbers:**

| | SmokeNet SF8 | SmokeNet SF4 |
| --- | --- | --- |
| one tile, 1 thread | 1.20 ms | 1.23 ms |
| one frame, 15 tiles, 4 threads | 6.9 ms | 6.9 ms |

On the A53 (ARMv8.0, no SDOT) the estimate is 15–30 ms per tile, or about 0.1 s
per frame on four cores, against a frame period of 2 s. These numbers will be
replaced by measurements; see [docs/numerics.md](docs/numerics.md).

## Layout

```
sfedge/            Python: SFx formats, QAT layers, models, exporter, reference executor,
                   .sfm container, ctypes binding, tiling, thermal encoding, metrics
runtime/           C99 runtime (include/, src/), self-test, sfrt_run CLI
tools/             prepare_tiles, train, eval, bench, edge_impulse, sync_sketch
uno_q/             App Lab app (wildfire-sentinel/) and on-board deploy script
hardware/          BOM, power budget, wiring, OpenSCAD mount
docs/              numerics, architecture, privacy, datasets, roadmap, proposal
tests/             pytest suite, including C runtime and sketch-on-host tests
```

## Quick start

```bash
pip install -e ".[train,dev]"
make -C runtime && make -C runtime test
pytest -q
```

Train and export SmokeNet:

```bash
python tools/prepare_tiles.py --root data/dfire/train --out data/tiles/train
python tools/prepare_tiles.py --root data/dfire/test --out data/tiles/val --clear-keep 1.0
python tools/train.py smokenet --data data/tiles --out runs/smoke-sf8
python tools/eval.py runs/smoke-sf8/smokenet-w1-sf8.sfm data/tiles/val      # deployed accuracy
python tools/bench.py runs/smoke-sf8/smokenet-w1-sf8.sfm                     # latency
```

Deploy to an UNO Q (run on the board, from a clone of this repo):

```bash
./uno_q/deploy.sh runs/smoke-sf8/smokenet-w1-sf8.sfm
```

Then open **Wildfire Sentinel** in Arduino App Lab and press Run. To try the
alerting logic without a board:

```bash
cd uno_q/wildfire-sentinel/python
python main.py --replay some_clip.mp4 --model ../../../runs/smoke-sf8/smokenet-w1-sf8.sfm
```

Edge Impulse baseline (needs `EI_API_KEY`):

```bash
python tools/edge_impulse.py onnx runs/smoke-sf8/model.pt --out runs/smoke-sf8/twin.onnx
python tools/edge_impulse.py deploy runs/smoke-sf8/twin.onnx --out runs/smoke-sf8/ei
```

## Docs

- [Integer-only SuperFloat inference](docs/numerics.md): the arithmetic, and why
  training accuracy equals deployed accuracy
- [Architecture](docs/architecture.md): the two tiers, the Bridge protocol and
  failure handling
- [Privacy and scope](docs/privacy.md): no person class, no stored frames, siting
  guidance
- [Datasets](docs/datasets.md) and [roadmap](docs/roadmap.md)
- [Hardware](hardware/BOM.md): BOM, power budget, [wiring](hardware/wiring.md),
  [mount](hardware/enclosure/sentinel_mount.scad)
- [Contest proposal](docs/proposal.md): Resilient America Preparedness Challenge

## Related

| Repository | |
| --- | --- |
| [superfloat](https://github.com/aloshdenny/superfloat) | The format, training benchmarks and results |
| [superfloat.gpu](https://github.com/aloshdenny/superfloat.gpu) | Atreides: a Q1.15 (SF16) accelerator in Sky130 |
| [superfloat.llvm](https://github.com/aloshdenny/superfloat.llvm) | Clang/LLVM with `sf16` as a builtin type |

## License

MIT
