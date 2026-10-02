/*
 * Integer kernels. Activations are NHWC int8 SF8 codes; weights are
 * [cout][kh][kw][cin_g] int8 SF8 codes; accumulation is int32.
 *
 * Requantisation is (acc + 2^(s-1)) >> s followed by a clamp, which assumes
 * an arithmetic right shift on signed integers. GCC, Clang and armcc all
 * provide one on every target we build for.
 */
#include "sf_runtime.h"

#if defined(__ARM_NEON) && defined(__aarch64__) && !defined(SF_NO_NEON)
#include <arm_neon.h>
#define SF_HAVE_NEON 1
#endif

int32_t sf_dot_s8(const int8_t *a, const int8_t *b, int n)
{
    int32_t sum = 0;
    int i = 0;
#ifdef SF_HAVE_NEON
    /* Cortex-A53 is ARMv8.0: no SDOT. Widening multiply into int16, then
     * pairwise accumulate into int32. |a*b| <= 128*127 fits int16 and the
     * pairwise add happens in int32, so nothing can overflow. */
    int32x4_t acc = vdupq_n_s32(0);
    for (; i + 16 <= n; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va), vget_low_s8(vb)));
        acc = vpadalq_s16(acc, vmull_high_s8(va, vb));
    }
    for (; i + 8 <= n; i += 8)
        acc = vpadalq_s16(acc, vmull_s8(vld1_s8(a + i), vld1_s8(b + i)));
    sum = vaddvq_s32(acc);
#endif
    for (; i < n; i++)
        sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

static inline int32_t rshift_round(int32_t v, int s)
{
    if (s <= 0)
        return v * (1 << -s);
    return (v + (1 << (s - 1))) >> s;
}

static inline void act_bounds(int act, int32_t *lo, int32_t *hi)
{
    *lo = act == SF_ACT_RELU1 ? 0 : -127;
    *hi = 127;
}

static inline void store(void *out, int idx, int32_t acc, int shift, int act, int32_t lo, int32_t hi)
{
    if (act == SF_ACT_RAW) {
        ((int32_t *)out)[idx] = acc;
        return;
    }
    int32_t v = rshift_round(acc, shift);
    ((int8_t *)out)[idx] = (int8_t)(v < lo ? lo : v > hi ? hi : v);
}

void sf_conv2d(const sf_op *op, const int8_t *in, void *out)
{
    const int H = op->in_h, W = op->in_w, C = op->in_c;
    const int OH = op->out_h, OW = op->out_w, COUT = op->cout;
    const int KH = op->kh, KW = op->kw, S = op->stride, P = op->pad;
    const int CG = op->cin_g, COUT_G = COUT / op->groups;
    const int shift = 7 - op->gain_exp;
    const int dense_rows = op->groups == 1; /* a full kernel row is contiguous */
    int32_t lo, hi;
    act_bounds(op->act, &lo, &hi);

    for (int oy = 0; oy < OH; oy++) {
        const int iy0 = oy * S - P;
        for (int ox = 0; ox < OW; ox++) {
            const int ix0 = ox * S - P;
            const int row_inside = dense_rows && ix0 >= 0 && ix0 + KW <= W;
            for (int oc = 0; oc < COUT; oc++) {
                const int g = oc / COUT_G;
                const int8_t *wk = op->weight + (size_t)oc * KH * KW * CG;
                int32_t acc = op->bias[oc];
                for (int ky = 0; ky < KH; ky++) {
                    const int iy = iy0 + ky;
                    if (iy < 0 || iy >= H)
                        continue;
                    const int8_t *irow = in + (size_t)iy * W * C;
                    if (row_inside) {
                        /* KW pixels x C channels are adjacent in NHWC, and so
                         * is the matching weight row: one long dot product. */
                        acc += sf_dot_s8(irow + (size_t)ix0 * C, wk + (size_t)ky * KW * CG, KW * CG);
                        continue;
                    }
                    for (int kx = 0; kx < KW; kx++) {
                        const int ix = ix0 + kx;
                        if (ix < 0 || ix >= W)
                            continue;
                        acc += sf_dot_s8(irow + (size_t)ix * C + g * CG,
                                         wk + ((size_t)ky * KW + kx) * CG, CG);
                    }
                }
                store(out, (oy * OW + ox) * COUT + oc, acc, shift, op->act, lo, hi);
            }
        }
    }
}

static inline int32_t div_round(int32_t v, int32_t d)
{
    /* floor((2v + d) / 2d) with C's truncating division corrected for n < 0 */
    const int32_t n = 2 * v + d, dd = 2 * d;
    int32_t q = n / dd;
    if (n % dd != 0 && n < 0)
        q -= 1;
    return q;
}

void sf_global_avg_pool(const int8_t *in, int hw, int c, int8_t *out)
{
    for (int ch = 0; ch < c; ch++) {
        int32_t sum = 0;
        for (int i = 0; i < hw; i++)
            sum += in[(size_t)i * c + ch];
        out[ch] = (int8_t)div_round(sum, hw);
    }
}

void sf_dense(const sf_op *op, const int8_t *in, void *out)
{
    const int cin = op->in_h * op->in_w * op->in_c;
    const int shift = 7 - op->gain_exp;
    int32_t lo, hi;
    act_bounds(op->act, &lo, &hi);
    for (int o = 0; o < op->cout; o++) {
        int32_t acc = op->bias[o] + sf_dot_s8(op->weight + (size_t)o * cin, in, cin);
        store(out, o, acc, shift, op->act, lo, hi);
    }
}

void sf_unpack_sf4(const uint8_t *packed, size_t count, int8_t *out_sf8)
{
    for (size_t i = 0; i < count; i++) {
        uint8_t nib = (packed[i >> 1] >> ((i & 1) * 4)) & 0x0F;
        int8_t code = (int8_t)(nib > 7 ? (int)nib - 16 : (int)nib);
        out_sf8[i] = (int8_t)(code * 16); /* SF4 k/8 == SF8 16k/128 */
    }
}

void sf_rgb8_to_sf8(const uint8_t *px, int8_t *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = (int8_t)(px[i] ^ 0x80);
}

uint32_t sf_crc32(const uint8_t *buf, size_t len)
{
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= buf[i];
        for (int k = 0; k < 8; k++)
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}
