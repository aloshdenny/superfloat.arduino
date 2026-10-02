/* Kernel-level checks that need no model file. Model-level equivalence with
 * the reference executor is tested from Python (tests/test_runtime_c.py). */
#include <stdio.h>
#include <stdlib.h>

#include "sf_runtime.h"

static int failures;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                        \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static void test_crc32(void)
{
    const uint8_t v[] = "123456789";
    CHECK(sf_crc32(v, 9) == 0xCBF43926u, "crc32 check value");
}

static void test_dot(void)
{
    int8_t a[203], b[203];
    srand(1);
    for (int i = 0; i < 203; i++) {
        a[i] = (int8_t)(rand() % 256 - 128);
        b[i] = (int8_t)(rand() % 255 - 127);
    }
    for (int n = 0; n <= 203; n += 7) {
        int32_t want = 0;
        for (int i = 0; i < n; i++)
            want += a[i] * b[i];
        CHECK(sf_dot_s8(a, b, n) == want, "dot n=%d", n);
    }
    /* worst case magnitude: every product is -128 * 127 */
    for (int i = 0; i < 203; i++) {
        a[i] = -128;
        b[i] = 127;
    }
    CHECK(sf_dot_s8(a, b, 203) == 203 * -128 * 127, "dot saturating inputs");
}

static void test_unpack_sf4(void)
{
    /* nibbles, low first: 1, -1, 7, -8, 7 (pad) -> SF8 codes are 16x */
    const uint8_t packed[] = {0xF1, 0x87, 0x07};
    const int8_t want[] = {16, -16, 112, -128, 112};
    int8_t out[5];
    sf_unpack_sf4(packed, 5, out);
    for (int i = 0; i < 5; i++)
        CHECK(out[i] == want[i], "sf4[%d] = %d, want %d", i, out[i], want[i]);
}

static void test_gap_rounding(void)
{
    const int8_t a[] = {1, 2};
    const int8_t b[] = {-1, -2};
    const int8_t c[] = {-1, -1, -2};
    int8_t out;
    sf_global_avg_pool(a, 2, 1, &out);
    CHECK(out == 2, "1.5 rounds up, got %d", out);
    sf_global_avg_pool(b, 2, 1, &out);
    CHECK(out == -1, "-1.5 rounds up, got %d", out);
    sf_global_avg_pool(c, 3, 1, &out);
    CHECK(out == -1, "-4/3 rounds to -1, got %d", out);
}

static void test_rgb8(void)
{
    const uint8_t px[] = {0, 128, 255};
    int8_t out[3];
    sf_rgb8_to_sf8(px, out, 3);
    CHECK(out[0] == -128 && out[1] == 0 && out[2] == 127, "rgb8 -> sf8");
}

int main(void)
{
    test_crc32();
    test_dot();
    test_unpack_sf4();
    test_gap_rounding();
    test_rgb8();
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("selftest: all checks passed\n");
    return 0;
}
