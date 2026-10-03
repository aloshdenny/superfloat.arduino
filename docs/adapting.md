# Adapting the sentinel

The runtime, training pipeline and app are general. Wildfire is the first
application. This page lists what to change for others.

## Another hazard, same camera setup

The pipeline assumes a fixed camera, fixed tiles and a slowly evolving hazard.
That fits several other problems:

| Hazard | Classes | Notes |
| --- | --- | --- |
| Flood stage at a bridge or culvert | `normal`, `high`, `overtopping` | Mask everything except the water tiles; the persistence rule fits rising water well |
| Debris flow or landslide on a slope | `stable`, `movement` | Feed frame differences as extra channels, as EmberNet does |
| Post-fire hotspot watch | `ambient`, `hotspot` | Thermal tier only; MCU and solar, no Linux needed |
| Industrial smoke or flaring | `clear`, `smoke`, `flame` | Same model; retrain on site captures |

Steps:

1. **Labels.** Edit `sfedge/labels.py` and give the model its new label list
   in `sfedge/models.py`. The runtime and app read labels from the `.sfm`
   file.
2. **Data.** Cut your dataset into tiles with `tools/prepare_tiles.py` and
   `--class-map`. Or write tiles straight into `<out>/<label>/` folders.
3. **Train and evaluate.** Run `tools/train.py` and `tools/eval.py`.
4. **Alert logic.** If the hazard is not "higher class is worse", adjust
   `sentinel/temporal.py`. It assumes class 0 is benign and higher classes
   escalate.

## Another model

The exporter accepts any `nn.Sequential` chain of `SFConv2d` →
`SFGlobalAvgPool` → `SFLinear`. Each `SFConv2d` can set its stride, groups
(including depthwise) and kernel size.

- **Not yet in the runtime:** residual connections, concatenation and
  upsampling. Adding an op means:
  1. the IR (`sfedge/graph.py`);
  2. the reference executor;
  3. the `.sfm` record;
  4. a C kernel;
  5. a test that holds all four to bit-exactness.
- **Keep activations BatchNorm-normalised.** SF8 activations live in [−1, 1).
  The SuperFloat results show CNNs with BatchNorm stay there and transformer
  residual streams do not.
- **Budget for the M33.** Below about 1 M MACs and 100 KiB of weights at SF8,
  or half that at SF4, a model runs comfortably on the M33. Weights stay in
  flash and are parsed in place.

## Another board

`runtime/` is plain C99 with an optional NEON path. Any Cortex-A board with
Linux runs the app's Python side unchanged; point `SFEDGE_LIB` at the built
library.

Any Cortex-M with about 32 KiB of free RAM runs EmberNet. Copy
`runtime/src` and `runtime/include` into the firmware, embed the model with
`sfedge.export.to_c_header`, and call `sf_model_load` and `sf_model_run`.

## Another alert path

Implement `Transport.send(alert) -> bool` in `sentinel/alerts.py` and add it
to the transport list in `main.py`. Return True only when the far end has
acknowledged the alert. The queue handles retries, backoff and persistence.
Candidates: MQTT, LoRaWAN or Meshtastic through a serial radio, or SMS through a
cellular modem.
