# Proposal: Wildfire Sentinel

**Resilient America Preparedness Challenge**: Track A (Developers)

**Challenge areas:** 1 Anticipate and Mitigate Risk, 3 Enable Real-Time
Response, 5 Operate in Mission-Critical Environments

**One line:** a solar-powered camera and thermal node that detects wildfire
smoke and ignitions entirely on an Arduino UNO Q, using SuperFloat, a numeric
format that makes CNN inference integer-only. It alerts in minutes over
whatever link survives, and never sends an image anywhere.

---

## 1. What problem are you solving, and which community is affected?

Wildfires are cheapest to stop in their first minutes and most destructive once
they run. The Camp Fire (2018), Lahaina (2023) and the Los Angeles fires of
January 2025 are recent reminders: in each, time to detection and time to
warning shaped the losses.

The places where ignitions start are where detection is weakest:

- wildland–urban interface towns, ranches, tribal lands, utility corridors and
  remote infrastructure;
- often beyond reliable cellular coverage;
- often reliant on someone noticing smoke and calling it in.

Camera networks exist, but the mature ones stream video to cloud AI. That needs:

- continuous high-bandwidth backhaul;
- per-camera subscriptions that small counties, fire districts and tribes
  struggle to fund;
- links that stay up. A fire damages power and communications first, so the
  systems are weakest when they matter most.

The affected community is rural and wildland–urban interface residents in
fire-prone regions of the US, and the small, often volunteer, fire services that
protect them.

## 2. Who benefits?

- **Volunteer and county fire departments, and tribal fire programs.** They get
  minutes-scale alerts with a bearing (the tile index maps to a direction) for
  the price of a node, not a subscription.
- **Residents** of WUI communities, ranches and remote homes. A local buzzer
  and LED alarm work even when the network does not, and webhook alerts can feed
  existing community alerting.
- **Utilities and operators of remote infrastructure** (substations, pumping
  stations, telecom sites). They can watch their own assets and corridors
  without streaming video off site.
- **Land managers and researchers.** The models, firmware and data pipeline are
  open source and adaptable to other hazards, such as smoke from industrial
  sites or post-fire hotspot monitoring.

## 3. Technical approach: hardware and edge AI model

**Hardware.**

- Arduino UNO Q: Qualcomm Dragonwing QRB2210 with 4x Cortex-A53 on Debian, plus
  an STM32U585 Cortex-M33.
- A USB camera.
- An MLX90640 32x24 thermal array on Qwiic.
- A buzzer, a mute button and the on-board LED matrix.
- A 60 W solar panel with a 320 Wh LiFePO4 battery, about 3 days of autonomy.
- A 3D-printed mount in an IP65 box.

All hardware design files are in the repository.

**Two tiers on one runtime.**

- **Camera tier (QRB2210).** Each frame is resized and cut into 15 tiles of
  128 px, following the tiled approach of SmokeyNet (Dewangan et al., 2022).
  - Each tile is classified clear, smoke or flame by **SmokeNet**: 28 M MACs and
    269 KiB of weights.
  - Four threads run on the four A53 cores.
  - A temporal voter requires 4 of 6 frames for smoke and 2 of 3 for flame, which
    suppresses glare, cloud and dust.
- **Thermal tier (STM32U585).** **EmberNet** (0.66 M MACs, 15 KiB) classifies
  MLX90640 frames together with their frame-to-frame difference, so flicker
  counts as evidence.
  - It runs on the MCU, independently of Linux.
  - It keeps watching day and night, even if Linux hangs; a heartbeat watchdog
    shows the fault on the matrix.

**The edge AI method: SuperFloat.**

SuperFloat (my prior work,
[aloshdenny/superfloat](https://github.com/aloshdenny/superfloat)) drops the
exponent entirely: a value is a sign plus fraction bits, i.e. Q1.(x−1) fixed
point. Trained network weights almost never leave [−1, 1], and my benchmarks
show SF8 weights matching FP32 on EuroSAT remote-sensing classification.

Because every SuperFloat scale is a power of two, a whole CNN runs on integers:

- int8 multiplies and int32 accumulation;
- one rounding shift and a clamp per layer;
- no zero points, no per-channel float scales, no requantisation multipliers.

SF4 weights (2 per byte) are a subset of SF8 and use the same kernel.

The quantisation-aware training is constructed so that the PyTorch forward pass
is bit-for-bit identical to the C runtime on the board. This is checked by the
test suite, so the accuracy measured in training is the accuracy deployed.

**Edge Impulse.**

- **Dataset management:** tiles from D-Fire and HPWREN, then our own field
  captures, are uploaded and labelled in Edge Impulse.
- **Baseline comparison:** each trained model is exported as an ONNX float twin
  for Edge Impulse BYOM. Edge Impulse profiles it and builds an int8 `.eim` for
  the UNO Q (`runner-linux-aarch64`).
  - SuperFloat and Edge Impulse int8 are then compared on the same board and
    the same tiles, for accuracy, latency and power.
- **Integration:** the app runs in Arduino App Lab, using the Bridge, Camera and
  WebUI bricks.

## 4. Why must it run at the edge?

- **Connectivity fails first.**
  - Many ignition-prone areas have little or no coverage.
  - Fires damage the power and communications infrastructure cloud systems
    depend on.
  - An edge node decides locally. It sounds a local alarm with no network, and
    queues alerts to disk until any link returns.
- **Bandwidth and cost.**
  - Streaming 720p around the clock is gigabytes per camera per day, which is
    prohibitive over cellular or satellite and on solar power.
  - An alert is about 200 bytes. Moving the decision to the board cuts backhaul
    by roughly six orders of magnitude.
- **Latency.** The system catches smoke within seconds of a frame, without
  queueing behind a cloud pipeline.
- **Privacy.**
  - Frames never leave the board and are never stored.
  - The model has no person class.
  - Alerts carry a tile index and a score, not an image.
  - This keeps the project clear of surveillance concerns and makes community
    consent much easier.
- **Power.** Integer-only inference on the A53 and the M33 fits a solar budget.
  The MCU tier keeps watching even when the Linux side is throttled.

## 5. How will you share the journey?

- **Open source from day one:** the code, the trained models, the hardware files
  and the test suite are public under MIT, with commit history showing how it was
  built.
- **devEco project page:** weekly updates through stage 2, with on-board
  measurements (latency, power, accuracy) as soon as the kit arrives, including
  the ones that disappoint.
- **Edge Impulse public project:** the labelled field dataset and the int8
  baseline, so others can reproduce the comparison.
- **Write-ups:**
  - an adaptation guide (other hazards, other cameras, other boards);
  - a technical article on integer-only SuperFloat inference;
  - a demo video of the node detecting smoke on recorded HPWREN sequences and
    a controlled burn-barrel test.
- **Negative results get published too.** The SuperFloat benchmark repository
  already documents where the format fails, such as transformer activations. This
  project will do the same.

## 6. Honest feasibility assessment

**Already working (before the kit arrives):**

- The SuperFloat integer runtime: C99, no allocator, NEON on aarch64.
  - It is bit-exact against a reference implementation and against the training
    forward pass.
  - It is tested on macOS arm64 and Linux aarch64, with about 120 automated
    tests.
- The QAT pipeline and exporter, the SmokeNet and EmberNet definitions, the full
  App Lab application, and the MCU sketch.
  - The sketch is compiled and exercised on a host against stub Arduino APIs.
  - Those host tests cover the watchdog, alarms, mute and a thermal hotspot
    detected by EmberNet.
- Edge Impulse tooling, the BOM, wiring, the power budget and the printable
  mount.

**Not yet proven, and the plan for each:**

- **A53 speed.**
  - Projected at 15–30 ms per tile and about 0.1 s per 15-tile frame on four
    cores. A frame every 2 s leaves a 10–20x margin.
  - Fallbacks: a narrower SmokeNet or fewer tiles.
  - First measurement in week one with the kit.
- **Accuracy on real smoke.**
  - Neither model has been trained on real imagery yet. The thermal model is a
    bootstrap trained on synthetic frames.
  - SmokeNet training on D-Fire is a stage 2 week-one task.
  - Success means meeting or beating the Edge Impulse int8 baseline on the same
    tiles.
- **False alarms.** These are the reason fire agencies distrust automated
  detection. The temporal voter will be tuned on real HPWREN sequences, and false
  alarms per camera-day will be reported alongside time to detect.
- **Library and toolchain unknowns on the UNO Q Zephyr core**, such as the
  MLX90640 driver. Fallback: port the Melexis reference C driver.
- **Night.** RGB smoke detection does not work in darkness. The thermal tier
  covers ignitions at night, and that limitation is stated plainly.

**Scope and support.** I am a solo developer, and the scope is sized for one
person over seven weeks: everything except on-board measurement and field data is
built. The main risks are performance and data, not missing software.

The deliverable for December 2 is:

- one working outdoor node;
- measured accuracy, latency and power against the Edge Impulse baseline;
- a documented path to more nodes.
