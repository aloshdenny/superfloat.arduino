# Datasets

Nothing here is redistributed. Download each dataset from its source, respect
its licence, and keep it under `data/` (git-ignored).

## Camera tier (SmokeNet)

| Dataset | What it is | Use |
| --- | --- | --- |
| [D-Fire](https://github.com/gaiasd/DFireDataset) | 21k images with YOLO boxes for smoke and fire, many from fixed cameras | Main training set; `tools/prepare_tiles.py` reads its layout directly |
| [HPWREN FIgLib](https://www.hpwren.ucsd.edu/FIgLib/) | Image sequences from fixed mountaintop cameras in southern California, before and after ignition | Hard negatives (haze, cloud, glare) and temporal-voter tuning on real sequences |
| [AI For Mankind wildfire smoke](https://github.com/aiformankind/wildfire-smoke-dataset) | Bounding-box annotated smoke from HPWREN cameras | Extra early-smoke positives |
| Own captures (stage 2) | Frames from the deployed node, uploaded to Edge Impulse for labelling | Site-specific fine-tuning |

Pipeline:

```bash
python tools/prepare_tiles.py --root data/dfire/train --out data/tiles/train
python tools/prepare_tiles.py --root data/dfire/test  --out data/tiles/val --clear-keep 1.0
python tools/train.py smokenet --data data/tiles --out runs/smoke-sf8
python tools/eval.py runs/smoke-sf8/smokenet-w1-sf8.sfm data/tiles/val
```

Tiles that only graze a box are left out rather than labelled clear. That
removes the worst label noise at tile boundaries.

- **Clear tiles:** subsampled 1 in 4 by default, because they outnumber
  positives roughly 20 to 1.
- **Class weighting:** the loss weights classes by inverse frequency to handle
  the remaining imbalance.

## Thermal tier (EmberNet)

There is no public dataset of MLX90640 frames of vegetation fires, so the tier
starts in two steps.

1. **Bootstrap.**
   - `sfedge/thermal.py` generates synthetic frames: ambient gradients,
     sun-warmed objects and sensor noise, plus sub-pixel to few-pixel
     flickering hotspots.
   - The model in `uno_q/wildfire-sentinel/sketch/ember_model.h` was trained
     on these frames.
   - Its role is to make the tier run end to end; it is not evidence of
     field accuracy.
2. **Field data (stage 2).**
   - The node records thermal frames continuously, plus controlled-burn and
     burn-barrel sessions where permitted.
   - These are labelled in Edge Impulse, and EmberNet is retrained on them.
