/*
 * sfrt_run: run an .sfm model from the command line, no Python needed.
 *
 *   sfrt_run model.sfm input.bin      # raw int8 SF8 input, HWC
 *   sfrt_run model.sfm --random 100   # timing on random inputs
 *
 * Prints the outputs (or the median latency), so a board can be checked
 * against outputs produced on a workstation by sfedge.reference.
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sf_runtime.h"

static void *read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc(n > 0 ? (size_t)n : 1); /* malloc is suitably aligned */
    if (buf && fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    *len = (size_t)n;
    return buf;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.sfm (input.bin | --random N)\n", argv[0]);
        return 2;
    }
    size_t len;
    uint8_t *buf = read_file(argv[1], &len);
    if (!buf) {
        fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    size_t arena_n = 0;
    int rc = sf_model_arena_bytes(buf, len, &arena_n);
    static sf_model m;
    void *arena = arena_n ? malloc(arena_n) : NULL;
    if (rc == SF_OK)
        rc = sf_model_load(&m, buf, len, arena, arena_n);
    if (rc != SF_OK) {
        fprintf(stderr, "load failed: %s\n", sf_status_str(rc));
        return 1;
    }
    const size_t in_n = (size_t)m.in_h * m.in_w * m.in_c;
    const size_t scratch_n = sf_model_scratch_bytes(&m);
    int8_t *input = malloc(in_n);
    void *scratch = malloc(scratch_n);
    int32_t *out = malloc(sizeof(int32_t) * m.n_out);

    if (strcmp(argv[2], "--random") == 0) {
        int reps = argc > 3 ? atoi(argv[3]) : 50;
        if (reps < 1)
            reps = 1;
        double *t = malloc(sizeof(double) * reps);
        srand(0);
        for (size_t i = 0; i < in_n; i++)
            input[i] = (int8_t)(rand() % 256 - 128);
        sf_model_run(&m, input, out, scratch, scratch_n); /* warm up */
        for (int r = 0; r < reps; r++) {
            double t0 = now_ms();
            sf_model_run(&m, input, out, scratch, scratch_n);
            t[r] = now_ms() - t0;
        }
        qsort(t, reps, sizeof(double), cmp_double);
        printf("%s: median %.3f ms over %d runs (arena %zu B, scratch %zu B)\n",
               m.name, t[reps / 2], reps, arena_n, scratch_n);
        free(t);
    } else {
        size_t got;
        int8_t *x = read_file(argv[2], &got);
        if (!x || got != in_n) {
            fprintf(stderr, "input must be %zu bytes of int8 HWC\n", in_n);
            return 1;
        }
        rc = sf_model_run(&m, x, out, scratch, scratch_n);
        if (rc != SF_OK) {
            fprintf(stderr, "run failed: %s\n", sf_status_str(rc));
            return 1;
        }
        for (int i = 0; i < m.n_out; i++)
            printf("%s%d", i ? " " : "", out[i]);
        printf("\n");
        free(x);
    }
    free(out);
    free(scratch);
    free(input);
    free(arena);
    free(buf);
    return 0;
}
