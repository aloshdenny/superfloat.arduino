/*
 * sf_runtime: integer-only SuperFloat (SFx) inference.
 *
 * Portable C99, no allocator, no floating point. The same sources build for
 * the UNO Q's Linux side (Cortex-A53, NEON path) and its STM32U585 MCU
 * (Cortex-M33, scalar path).
 *
 * Every activation is SF8 (int8 codes, value = code / 128). Weights are SF8,
 * or SF4 widened to SF8 at load. Requantisation is a rounding right shift,
 * because every scale in the SFx schema is a power of two. See
 * sfedge/graph.py for the arithmetic and sfedge/sfm.py for the file layout;
 * sfedge/reference.py is the golden model these kernels are tested against.
 */
#ifndef SF_RUNTIME_H
#define SF_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SF_MAX_OPS 32
#define SF_MAX_LABELS 16

enum sf_status {
    SF_OK = 0,
    SF_ERR_TRUNCATED = -1,
    SF_ERR_MAGIC = -2,
    SF_ERR_VERSION = -3,
    SF_ERR_CHECKSUM = -4,
    SF_ERR_UNSUPPORTED = -5,
    SF_ERR_SHAPE = -6,
    SF_ERR_ARENA = -7,
    SF_ERR_ARG = -8,
};

enum sf_op_kind { SF_OP_CONV = 1, SF_OP_GAP = 2, SF_OP_DENSE = 3 };
enum sf_act { SF_ACT_LINEAR = 0, SF_ACT_RELU1 = 1, SF_ACT_RAW = 2 };

typedef struct {
    uint8_t kind, act, wbits;
    int8_t gain_exp;
    uint16_t cout, kh, kw, cin_g, stride, pad, groups;
    uint16_t in_h, in_w, in_c; /* resolved at load */
    uint16_t out_h, out_w;
    const int8_t *weight;      /* SF8 codes: in the file, or widened into the arena */
    const int32_t *bias;       /* points into the file buffer (4-byte aligned) */
} sf_op;

typedef struct {
    uint16_t n_ops, in_h, in_w, in_c, n_labels, n_out;
    uint8_t out_raw;           /* 1: output is int32 logits, 0: int8 codes */
    sf_op ops[SF_MAX_OPS];
    char name[64];
    char labels[SF_MAX_LABELS][32];
    size_t max_act;            /* largest int8 activation, in bytes */
} sf_model;

/* Bytes of arena the model needs for widened SF4 weights (0 for pure SF8). */
int sf_model_arena_bytes(const uint8_t *buf, size_t len, size_t *out);

/*
 * Parse an .sfm image in place. `buf` must stay alive and be 4-byte aligned;
 * SF8 weights and all biases are referenced, not copied. `arena` may be NULL
 * when sf_model_arena_bytes reported 0.
 */
int sf_model_load(sf_model *m, const uint8_t *buf, size_t len, void *arena, size_t arena_len);

/* Scratch bytes one sf_model_run call needs. Scratch is per call, so several
 * threads can run the same model concurrently with their own scratch. */
size_t sf_model_scratch_bytes(const sf_model *m);

/*
 * Run on one SF8 input (in_h * in_w * in_c int8, HWC). Writes n_out values to
 * `out`: int32 when m->out_raw, otherwise int8 widened to int32.
 */
int sf_model_run(const sf_model *m, const int8_t *input, int32_t *out, void *scratch, size_t scratch_len);

/* For FFI callers that cannot see the struct definition. */
size_t sf_model_sizeof(void);
int sf_model_n_out(const sf_model *m);
int sf_model_out_raw(const sf_model *m);

/* uint8 camera pixels -> SF8 codes: (p - 128) / 128 is a flip of the top bit. */
void sf_rgb8_to_sf8(const uint8_t *px, int8_t *out, size_t n);

/* Kernels, exposed for testing and for callers that build graphs by hand. */
int32_t sf_dot_s8(const int8_t *a, const int8_t *b, int n);
void sf_conv2d(const sf_op *op, const int8_t *in, void *out);
void sf_global_avg_pool(const int8_t *in, int hw, int c, int8_t *out);
void sf_dense(const sf_op *op, const int8_t *in, void *out);
void sf_unpack_sf4(const uint8_t *packed, size_t count, int8_t *out_sf8);

uint32_t sf_crc32(const uint8_t *buf, size_t len);
const char *sf_status_str(int status);

#ifdef __cplusplus
}
#endif

#endif /* SF_RUNTIME_H */
