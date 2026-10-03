# Roadmap

## Done (proposal stage, Oct 2026)

- SuperFloat integer runtime in C99:
  - allocator-free;
  - NEON path for aarch64;
  - bit-exact with a numpy reference and with the QAT forward pass;
  - tested on macOS arm64 and Linux aarch64.
- QAT training on the SF grid with BatchNorm folding and power-of-two layer
  gain, exporting to `.sfm` files and C headers.
- SmokeNet and EmberNet definitions, sized for the A53 cluster and the M33.
- App Lab app:
  - camera tiling and parallel inference;
  - temporal voting;
  - store-and-forward alerts;
  - offline status page;
  - MCU sketch with watchdog, matrix, buzzer and thermal tier.
- Edge Impulse integration: dataset upload, ONNX float twin, profiling and
  `.eim` baseline deployment.
- Hardware: BOM, power budget, wiring, printable mount.

Not yet done: nothing has run on an UNO Q. The sketch is compiled and exercised
on a host against stub Arduino APIs only.

## Stage 2 (Oct 12 – Dec 2)

| Week of | Milestone |
| --- | --- |
| Oct 12 | Kit arrives. On-target build of sketch and runtime. Measure A53 and M33 latency and board power; replace estimates in the README and BOM. |
| Oct 19 | Train SmokeNet on D-Fire tiles (SF8 and SF4). Report deployed accuracy via `tools/eval.py`. Edge Impulse int8 baseline (`.eim`) on the same tiles, same board. |
| Oct 26 | Tune the temporal voter on HPWREN FIgLib sequences: time-to-detect vs false alarms per camera-day. |
| Nov 2 | Thermal tier on real MLX90640 data: field captures, Edge Impulse labelling, EmberNet retrain. |
| Nov 9 | Outdoor node: printed mount, enclosure, solar. Multi-day run with logging. |
| Nov 16 | Tile masking for privacy; LoRa or Meshtastic alert transport for no-coverage sites. |
| Nov 23 | Write-up, adaptation guide, demo video. |
| Dec 2 | Submission. |

## Known gaps and risks

- **A53 throughput is projected, not measured.** If a 15-tile frame takes over
  2 s, fall back to width-0.75 SmokeNet or fewer tiles. The period between
  frames is configurable, and smoke tolerates a slower cadence.
- **Night smoke is invisible to an RGB camera.** At night the system relies on
  the thermal tier and on flame or glow detection.
- **The MLX90640 library is untested on the UNO Q.** The Adafruit library has
  not been tested on the UNO Q's Zephyr core. If it does not build, port the
  Melexis reference driver, which is plain C over I2C.
- **SF8 activations assume BatchNorm-normalised CNNs.** The SuperFloat results
  show this holds for CNNs and fails for transformer residual streams. Both
  models here are plain BatchNorm CNNs.
