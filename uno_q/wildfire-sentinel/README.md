# Wildfire Sentinel (App Lab app)

On-device wildfire smoke and ignition detection for the Arduino UNO Q.

| Side | What runs |
| --- | --- |
| **Linux (`python/`)** | Camera capture, 5x3 tiling, SmokeNet on the SuperFloat C runtime across four threads, temporal voting, store-and-forward alerts, and a status page on port 7000 |
| **MCU (`sketch/`)** | LED-matrix tile map, buzzer and mute button, a heartbeat watchdog on the Linux side, and EmberNet on an MLX90640 thermal array |

## Install

On the board, from a clone of the repository:

```bash
./uno_q/deploy.sh path/to/smokenet.sfm
```

This copies the app to `~/ArduinoApps/wildfire-sentinel` with the runtime and
model vendored in. Edit `config.json` there, then open the app in App Lab and
press Run.

## Configure (`config.json`)

| Key | Default | Meaning |
| --- | --- | --- |
| `node_id`, `lat`, `lon` | `sentinel-01`, none | Included in every alert |
| `period_s` / `night_period_s` | 2 / 10 | Seconds between frames by day and by night |
| `grid_cols`, `grid_rows`, `tile` | 5, 3, 128 | Must match the model's input |
| `mask_tiles` | `[]` | Tiles never scored (homes, roads, trails) |
| `voter.smoke_k` of `voter.smoke_n` | 4 of 6 | Frames needed for a smoke alert |
| `voter.flame_k` of `voter.flame_n` | 2 of 3 | Frames needed for a flame alert |
| `webhook_url`, `webhook_token` | none | JSON POST per alert; retried until acknowledged |

## LED matrix

```
 cols 0-9, rows 0-5   camera tiles, 2x2 px each (dim: clear, bright: smoke, blinking: flame)
 row 7, cols 0-9      thermal hotspot probability bar
 col 12               heartbeat blink; a blinking bar means Linux has stopped responding
```

## Thermal tier

The thermal tier is off by default.

1. Wire the MLX90640 to Qwiic.
2. Add the Adafruit MLX90640 library to `sketch/sketch.yaml`.
3. Set `SENTINEL_THERMAL` to 1 in `sketch.ino`.

The bundled `ember_model.h` is a bootstrap model trained on synthetic frames
only. Replace it with one trained on field captures:

```bash
python tools/train.py embernet --data <tiles> --out runs/ember
python tools/sync_sketch.py --header runs/ember/embernet-sf8.h
```

## Privacy

The app has no person class, and frames are never stored or sent. Alerts carry
the tile index and a score. See [docs/privacy.md](../../docs/privacy.md).
