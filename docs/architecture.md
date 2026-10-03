# Architecture

```
                         Arduino UNO Q
 ┌────────────────────────────────────────────────────────────────────┐
 │  Qualcomm QRB2210 (Debian, 4x Cortex-A53)                          │
 │                                                                    │
 │  USB camera ─► resize to 5x3 tiles ─► SmokeNet x 15 (4 threads)    │
 │                     (sfedge.tiling)    libsfrt, NEON int8          │
 │                                              │                     │
 │                          per-tile clear/smoke/flame probabilities  │
 │                                              ▼                     │
 │                                    temporal voter (k of n frames)  │
 │                                              │ events              │
 │            status page ◄── Sentinel ─────────┤                     │
 │            (web_ui, :7000)                   ▼                     │
 │                                    alert queue (fsync'd JSONL)     │
 │                                      └─► webhook / log (retry)     │
 │                      │ Bridge RPC: heartbeat, set_tiles  ▲         │
 ├──────────────────────┼───────────────────────────────────┼─────────┤
 │  STM32U585 (Zephyr, Cortex-M33)                         │         │
 │                      ▼                                   │ notify  │
 │  LED matrix tile map, buzzer, mute      MLX90640 ─► EmberNet ──────┘
 │  watchdog: no heartbeat for 30 s ─► fault display, thermal tier    │
 │                                       keeps running on its own     │
 └────────────────────────────────────────────────────────────────────┘
```

## Two tiers, one runtime

| | Camera tier | Thermal tier |
| --- | --- | --- |
| Runs on | Linux side, 4x Cortex-A53 | MCU side, Cortex-M33 |
| Sensor | USB camera, 1280x720 | MLX90640, 32x24 thermal array |
| Model | SmokeNet, 28 M MACs/tile, 269 KiB (SF8) | EmberNet, 0.66 M MACs, 15 KiB (SF8) |
| Detects | smoke plumes at distance, by day | ignition heat signatures, day and night |
| Survives | – | Linux crash, Linux hang, camera failure |

Both tiers execute `.sfm` files on the same C runtime (`runtime/`):

- **MCU:** the model is compiled into flash as a const array and parsed in
  place. No allocator, no arena, and 12 KiB of scratch for EmberNet.
- **Linux:** the Python app calls the same code through ctypes. The GIL is
  released during each call, so four threads score four tiles at once.

## Failure modes and responses

| Failure | Detected by | Response |
| --- | --- | --- |
| Camera unplugged or stalled | `capture()` returns None | Status reports a camera fault; the MCU shows it on the matrix; the thermal tier continues |
| Linux side hangs or reboots | No heartbeat for 30 s | MCU blinks a fault bar and keeps the thermal tier and buzzer running |
| Network down | Webhook delivery fails | Alert stays queued on disk; retried with exponential backoff up to 15 min |
| Power loss mid-write | Torn last JSONL line | Ignored on reload; acknowledged alerts are not re-sent |
| Bridge RPC errors | Exception in `Bridge.call` | Counted and logged; the camera loop never stops for them |
| Corrupted model file | CRC32 in the `.sfm` | Refuses to load (both runtimes) instead of running garbage |
| Single-frame false positive (glare, cloud, dust) | Temporal voter | Needs 4 of 6 frames (smoke) or 2 of 3 (flame) with a smoothed score |

## Why tiles

Smoke at 10–20 km is a few pixels wide. Classifying fixed 128 px tiles of a
resized frame keeps that signal at a resolution the network can use.

- Every tile gets the same compute.
- Each tile maps to a bearing, which lets two or more nodes triangulate a
  plume.
- The tile index is what an alert reports, so no image ever has to leave the
  board.

The approach follows SmokeyNet (Dewangan et al., 2022, *Remote Sensing* 14(4)),
reduced to a model an A53 can run in tens of milliseconds.
