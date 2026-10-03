/*
 * .sfm parsing and graph execution. The layout is documented in
 * sfedge/sfm.py. Multi-byte fields are little-endian; both UNO Q cores are
 * little-endian, so biases are referenced in place rather than copied.
 */
#include <string.h>

#include "sf_runtime.h"

#define HEADER_BYTES 32
#define OPHEAD_BYTES 16

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static size_t align4(size_t x) { return (x + 3u) & ~(size_t)3u; }

static int read_str(const uint8_t *buf, size_t end, size_t *pos, char *dst, size_t cap)
{
    if (*pos + 1 > end)
        return SF_ERR_TRUNCATED;
    size_t n = buf[*pos];
    if (*pos + 1 + n > end)
        return SF_ERR_TRUNCATED;
    if (dst) {
        size_t k = n < cap - 1 ? n : cap - 1;
        memcpy(dst, buf + *pos + 1, k);
        dst[k] = '\0';
    }
    *pos += 1 + n;
    return SF_OK;
}

/*
 * One pass over the file. With m == NULL it only validates and totals the
 * arena; otherwise it also fills m and widens SF4 weights into the arena.
 */
static int parse(sf_model *m, const uint8_t *buf, size_t len, int8_t *arena, size_t arena_len,
                 size_t *arena_used)
{
    if (!buf || len < HEADER_BYTES + 4)
        return SF_ERR_TRUNCATED;
    if (((uintptr_t)buf & 3u) != 0)
        return SF_ERR_ARG;
    const size_t end = len - 4;
    if (sf_crc32(buf, end) != rd32(buf + end))
        return SF_ERR_CHECKSUM;
    if (memcmp(buf, "SFM1", 4) != 0)
        return SF_ERR_MAGIC;
    if (rd16(buf + 4) != 1)
        return SF_ERR_VERSION;

    const uint16_t n_ops = rd16(buf + 6);
    uint16_t h = rd16(buf + 8), w = rd16(buf + 10), c = rd16(buf + 12);
    const uint16_t n_labels = rd16(buf + 14);
    if (n_ops == 0 || n_ops > SF_MAX_OPS || n_labels > SF_MAX_LABELS)
        return SF_ERR_UNSUPPORTED;
    if (m) {
        memset(m, 0, sizeof *m);
        m->n_ops = n_ops;
        m->in_h = h;
        m->in_w = w;
        m->in_c = c;
        m->n_labels = n_labels;
        m->max_act = (size_t)h * w * c;
    }

    size_t pos = HEADER_BYTES;
    int rc = read_str(buf, end, &pos, m ? m->name : NULL, sizeof m->name);
    for (int i = 0; rc == SF_OK && i < n_labels; i++)
        rc = read_str(buf, end, &pos, m ? m->labels[i] : NULL, sizeof m->labels[0]);
    if (rc != SF_OK)
        return rc;
    pos = align4(pos);

    size_t used = 0;
    for (int i = 0; i < n_ops; i++) {
        if (pos + OPHEAD_BYTES > end)
            return SF_ERR_TRUNCATED;
        const uint8_t *p = buf + pos;
        sf_op op;
        memset(&op, 0, sizeof op);
        op.kind = p[0];
        op.act = p[1];
        op.wbits = p[2];
        op.gain_exp = (int8_t)p[3];
        op.in_h = h;
        op.in_w = w;
        op.in_c = c;
        pos += OPHEAD_BYTES;

        size_t n_w = 0;
        if (op.kind == SF_OP_CONV) {
            op.cout = rd16(p + 4);
            op.kh = p[6];
            op.kw = p[7];
            op.cin_g = rd16(p + 8);
            op.stride = p[10];
            op.pad = p[11];
            op.groups = rd16(p + 12);
            if (!op.stride || !op.groups || op.cin_g * op.groups != c || op.cout % op.groups)
                return SF_ERR_SHAPE;
            if (h + 2 * op.pad < op.kh || w + 2 * op.pad < op.kw)
                return SF_ERR_SHAPE;
            op.out_h = (uint16_t)((h + 2 * op.pad - op.kh) / op.stride + 1);
            op.out_w = (uint16_t)((w + 2 * op.pad - op.kw) / op.stride + 1);
            n_w = (size_t)op.cout * op.kh * op.kw * op.cin_g;
            h = op.out_h;
            w = op.out_w;
            c = op.cout;
        } else if (op.kind == SF_OP_DENSE) {
            op.cout = rd16(p + 4);
            if (rd16(p + 6) != (size_t)h * w * c)
                return SF_ERR_SHAPE;
            op.out_h = op.out_w = 1;
            n_w = (size_t)op.cout * h * w * c;
            h = w = 1;
            c = op.cout;
        } else if (op.kind == SF_OP_GAP) {
            op.cout = c;
            op.out_h = op.out_w = 1;
            h = w = 1;
        } else {
            return SF_ERR_UNSUPPORTED;
        }

        if (op.act > SF_ACT_RAW || (op.act == SF_ACT_RAW && i != n_ops - 1))
            return SF_ERR_UNSUPPORTED;

        if (n_w) {
            if (op.wbits == 8) {
                if (pos + n_w > end)
                    return SF_ERR_TRUNCATED;
                op.weight = (const int8_t *)(buf + pos);
                pos += n_w;
            } else if (op.wbits == 4) {
                const size_t packed = (n_w + 1) / 2;
                if (pos + packed > end)
                    return SF_ERR_TRUNCATED;
                if (m) {
                    if (!arena || used + n_w > arena_len)
                        return SF_ERR_ARENA;
                    sf_unpack_sf4(buf + pos, n_w, arena + used);
                    op.weight = arena + used;
                }
                used += align4(n_w);
                pos += packed;
            } else {
                return SF_ERR_UNSUPPORTED;
            }
            pos = align4(pos);
            if (pos + 4u * op.cout > end)
                return SF_ERR_TRUNCATED;
            op.bias = (const int32_t *)(const void *)(buf + pos);
            pos += 4u * op.cout;
        }

        if (m) {
            m->ops[i] = op;
            const size_t act_bytes = (size_t)h * w * c;
            if (act_bytes > m->max_act)
                m->max_act = act_bytes;
            if (i == n_ops - 1) {
                m->n_out = (uint16_t)((size_t)h * w * c);
                m->out_raw = op.act == SF_ACT_RAW;
            }
        }
    }
    if (pos != end)
        return SF_ERR_TRUNCATED;
    if (arena_used)
        *arena_used = used;
    return SF_OK;
}

int sf_model_arena_bytes(const uint8_t *buf, size_t len, size_t *out)
{
    return parse(NULL, buf, len, NULL, 0, out);
}

int sf_model_load(sf_model *m, const uint8_t *buf, size_t len, void *arena, size_t arena_len)
{
    if (!m)
        return SF_ERR_ARG;
    return parse(m, buf, len, (int8_t *)arena, arena_len, NULL);
}

size_t sf_model_scratch_bytes(const sf_model *m)
{
    return 2 * align4(m->max_act);
}

int sf_model_run(const sf_model *m, const int8_t *input, int32_t *out, void *scratch, size_t scratch_len)
{
    if (!m || !input || !out || !scratch)
        return SF_ERR_ARG;
    if (scratch_len < sf_model_scratch_bytes(m))
        return SF_ERR_ARENA;
    int8_t *bufs[2] = {(int8_t *)scratch, (int8_t *)scratch + align4(m->max_act)};
    const int8_t *cur = input;
    int which = 0;
    for (int i = 0; i < m->n_ops; i++) {
        const sf_op *op = &m->ops[i];
        const int raw = op->act == SF_ACT_RAW && op->kind != SF_OP_GAP;
        void *dst = raw ? (void *)out : (void *)bufs[which];
        switch (op->kind) {
        case SF_OP_CONV:
            sf_conv2d(op, cur, dst);
            break;
        case SF_OP_GAP:
            sf_global_avg_pool(cur, op->in_h * op->in_w, op->in_c, (int8_t *)dst);
            break;
        case SF_OP_DENSE:
            sf_dense(op, cur, dst);
            break;
        default:
            return SF_ERR_UNSUPPORTED;
        }
        if (!raw) {
            cur = bufs[which];
            which ^= 1;
        }
    }
    if (!m->out_raw)
        for (int i = 0; i < m->n_out; i++)
            out[i] = cur[i];
    return SF_OK;
}

size_t sf_model_sizeof(void) { return sizeof(sf_model); }

int sf_model_n_out(const sf_model *m) { return m->n_out; }

int sf_model_out_raw(const sf_model *m) { return m->out_raw; }

const char *sf_status_str(int status)
{
    switch (status) {
    case SF_OK: return "ok";
    case SF_ERR_TRUNCATED: return "truncated model";
    case SF_ERR_MAGIC: return "not an .sfm file";
    case SF_ERR_VERSION: return "unsupported .sfm version";
    case SF_ERR_CHECKSUM: return "checksum mismatch";
    case SF_ERR_UNSUPPORTED: return "unsupported op or format";
    case SF_ERR_SHAPE: return "inconsistent shapes";
    case SF_ERR_ARENA: return "arena or scratch too small";
    case SF_ERR_ARG: return "bad argument";
    default: return "unknown error";
    }
}
