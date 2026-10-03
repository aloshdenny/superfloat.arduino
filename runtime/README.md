# sf_runtime

Integer-only SuperFloat inference in portable C99. One header, two source files,
no allocator, no floating point.

```
include/sf_runtime.h   API
src/sf_model.c         .sfm parsing (in place) and graph execution
src/sf_kernels.c       conv2d (incl. grouped/depthwise), global average pool, dense
tests/selftest.c       kernel checks that need no model file
tools/sfrt_run.c       command-line runner: check outputs or time a model on a board
```

## Build

```bash
make            # build/libsfrt.{so,dylib}, build/selftest, build/sfrt_run
make test
make CFLAGS="-O3 -DSF_NO_NEON"          # force the scalar kernels
make CC=aarch64-linux-gnu-gcc           # cross-compile for the UNO Q's Linux side
```

On the MCU the sources are compiled by the Arduino build from the copy in
`uno_q/wildfire-sentinel/sketch/src/sfrt/`. Keep it in sync with
`tools/sync_sketch.py`; CI fails if it drifts.

## Use

```c
#include "sf_runtime.h"

static sf_model m;                      /* ~2 KiB: keep it static on an MCU */
size_t arena_n;
sf_model_arena_bytes(buf, len, &arena_n);       /* 0 unless the model has SF4 weights */
sf_model_load(&m, buf, len, arena, arena_n);    /* buf must be 4-byte aligned and outlive m */

size_t scratch_n = sf_model_scratch_bytes(&m);  /* per call; one per thread */
int32_t logits[3];
sf_model_run(&m, input_sf8, logits, scratch, scratch_n);
```

- **Input:** SF8 codes, HWC. For camera pixels use `sf_rgb8_to_sf8`, which
  flips the top bit.
- **Output:** raw int32 logits. One unit is 2^e / 16384, where `e` is the last
  layer's `gain_exp`; the argmax needs no scaling.
- **Corruption:** every load checks the file's CRC32, so a corrupted model is
  refused rather than run.

## Arithmetic

Per layer: `acc = bias + Σ w·a` (int8 × int8 into int32), then
`out = clamp((acc + 2^(s−1)) >> s)` with `s = 7 − gain_exp`. See
[docs/numerics.md](../docs/numerics.md) for the derivation.

The NEON path uses `vmull_s8` and `vpadalq_s16`, which are ARMv8.0
instructions. The Cortex-A53 has no `SDOT`, so the dot-product extension is not
assumed.
