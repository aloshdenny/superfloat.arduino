/*
 * Wildfire sentinel: STM32U585 (Cortex-M33) side of the UNO Q.
 *
 * The MCU owns everything that has to keep working when Linux does not:
 *
 *   - the 8x13 LED matrix: a live map of the camera tiles plus status;
 *   - the buzzer and a mute button for crews on site;
 *   - a heartbeat watchdog on the Linux side (Bridge "heartbeat");
 *   - the thermal tier: EmberNet on an MLX90640 over Qwiic, running on the
 *     same integer SuperFloat runtime as the camera model, sized for the M33.
 *
 * If heartbeats stop, the matrix shows a fault and the thermal tier keeps
 * watching on its own. The thermal tier needs the Adafruit MLX90640 library
 * (add it to sketch.yaml) and is off by default; the runtime self-test runs
 * either way, so a fresh board reports EmberNet latency on the Monitor.
 */
#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>

#include <vector>

#include "ember_model.h"
#include "src/sfrt/sf_runtime.h"

#ifndef SENTINEL_THERMAL
#define SENTINEL_THERMAL 0
#endif

#if SENTINEL_THERMAL
#include <Adafruit_MLX90640.h>
#include <Wire.h>
Adafruit_MLX90640 mlx;
static float t_now[32 * 24], t_prev[32 * 24];
#endif

static const int BUZZER_PIN = 9;  // active buzzer (HIGH = sound) via a transistor
static const int MUTE_PIN = 8;    // momentary button to GND
static const uint32_t HEARTBEAT_TIMEOUT_MS = 30000;
static const uint32_t MUTE_MS = 10UL * 60UL * 1000UL;
static const uint32_t THERMAL_PERIOD_MS = 250;  // MLX90640 at 4 Hz
static const int GRID_COLS = 5, GRID_ROWS = 3, N_TILES = GRID_COLS * GRID_ROWS;
static const int MAT_W = 13, MAT_H = 8;

enum { LEVEL_CLEAR = 0, LEVEL_SMOKE = 1, LEVEL_FLAME = 2 };

Arduino_LED_Matrix matrix;
static uint8_t frame_buf[MAT_W * MAT_H];

static sf_model ember;
static bool ember_ok = false;
static uint8_t scratch[16384] __attribute__((aligned(4)));
static int8_t ember_in[24 * 32 * 2];

static uint32_t last_heartbeat_ms = 0;
static bool linux_seen = false;
static uint8_t linux_state = 0;
static uint8_t tile_levels[N_TILES];
static uint8_t thermal_level = LEVEL_CLEAR;
static uint16_t thermal_milli = 0;
static uint32_t muted_until = 0;

/* ---- Bridge handlers (provide_safe: they run on the loop thread) ---- */

bool on_heartbeat(int seq, int state)
{
    (void)seq;
    last_heartbeat_ms = millis();
    linux_seen = true;
    linux_state = (uint8_t)state;
    return true;
}

bool on_tiles(std::vector<uint8_t> levels)
{
    for (int i = 0; i < N_TILES; i++)
        tile_levels[i] = i < (int)levels.size() ? levels[i] : 0;
    return true;
}

/* ---- EmberNet ---- */

static int32_t ember_hotspot_margin(void)
{
    int32_t logits[2];
    if (sf_model_run(&ember, ember_in, logits, scratch, sizeof scratch) != SF_OK)
        return INT32_MIN;
    return logits[1] - logits[0];
}

static float margin_to_prob(int32_t margin)
{
    const int e = ember.ops[ember.n_ops - 1].gain_exp;
    const float real = (float)margin * (float)(1 << e) / 16384.0f;
    return 1.0f / (1.0f + expf(-real));
}

static void runtime_selftest(void)
{
    size_t arena = 0;
    int rc = sf_model_arena_bytes(ember_sfm, ember_sfm_len, &arena);
    if (rc == SF_OK && arena == 0)
        rc = sf_model_load(&ember, ember_sfm, ember_sfm_len, NULL, 0);
    if (rc == SF_OK && sf_model_scratch_bytes(&ember) > sizeof scratch)
        rc = SF_ERR_ARENA;
    ember_ok = rc == SF_OK;
    if (!ember_ok) {
        Monitor.print("embernet: load failed: ");
        Monitor.println(sf_status_str(rc));
        return;
    }
    memset(ember_in, 0, sizeof ember_in);
    const uint32_t t0 = micros();
    const int32_t m = ember_hotspot_margin();
    const uint32_t dt = micros() - t0;
    Monitor.print("embernet: ");
    Monitor.print(ember.name);
    Monitor.print(" ok, ");
    Monitor.print(dt);
    Monitor.print(" us/inference, blank-frame hotspot p=");
    Monitor.println((int)(margin_to_prob(m) * 1000.0f));
}

#if SENTINEL_THERMAL
/* Same encoding as sfedge/thermal.py. */
static int8_t sat8(float v)
{
    long r = lroundf(v);
    return (int8_t)(r < -127 ? -127 : r > 127 ? 127 : r);
}

static void thermal_step(uint32_t now)
{
    static uint32_t next = 0;
    static uint8_t streak = 0;
    static uint32_t last_notify = 0;
    static bool primed = false;
    if (!ember_ok || (int32_t)(now - next) < 0)
        return;
    next = now + THERMAL_PERIOD_MS;
    if (mlx.getFrame(t_now) != 0)
        return;
    if (!primed) {
        memcpy(t_prev, t_now, sizeof t_now);
        primed = true;
        return;
    }
    int hot = 0;
    for (int i = 0; i < 32 * 24; i++) {
        ember_in[2 * i] = sat8((t_now[i] - 25.0f) * 127.0f / 100.0f);
        ember_in[2 * i + 1] = sat8((t_now[i] - t_prev[i]) * 127.0f / 25.0f);
        if (t_now[i] > t_now[hot])
            hot = i;
    }
    memcpy(t_prev, t_now, sizeof t_now);

    const float p = margin_to_prob(ember_hotspot_margin());
    thermal_milli = (uint16_t)(p * 1000.0f);
    streak = p > 0.5f ? (uint8_t)(streak < 255 ? streak + 1 : 255) : 0;
    thermal_level = streak >= 2 ? LEVEL_FLAME : LEVEL_CLEAR;
    if (thermal_level == LEVEL_FLAME && (last_notify == 0 || now - last_notify > 60000UL)) {
        last_notify = now;
        Bridge.notify("thermal_hotspot", (int)thermal_milli, hot % 32, hot / 32);
    }
}
#else
static void thermal_step(uint32_t) {}
#endif

/* ---- Matrix and buzzer ---- */

static void px(int x, int y, uint8_t v)
{
    if (x >= 0 && x < MAT_W && y >= 0 && y < MAT_H)
        frame_buf[y * MAT_W + x] = v;
}

static void draw(uint32_t now, bool linux_alive, uint8_t level)
{
    memset(frame_buf, 0, sizeof frame_buf);
    const bool blink = (now / 250) & 1;
    /* camera tiles: 2x2 pixels each in the top-left 10x6 block */
    for (int t = 0; t < N_TILES; t++) {
        const uint8_t l = linux_alive ? tile_levels[t] : 0;
        const uint8_t v = l == LEVEL_FLAME ? (blink ? 7 : 2) : l == LEVEL_SMOKE ? 5 : 1;
        const int x = (t % GRID_COLS) * 2, y = (t / GRID_COLS) * 2;
        px(x, y, v);
        px(x + 1, y, v);
        px(x, y + 1, v);
        px(x + 1, y + 1, v);
    }
    /* bottom row: thermal hotspot probability as a 10-pixel bar */
    for (int i = 0; i < (thermal_milli + 50) / 100 && i < 10; i++)
        px(i, 7, thermal_level == LEVEL_FLAME && blink ? 7 : 3);
    /* right edge: heartbeat pixel, or a blinking bar when Linux is gone */
    if (linux_alive) {
        px(12, 0, (now / 1000) & 1 ? 4 : 1);
        if (linux_state != 0)
            px(12, 2, 6);  // camera fault reported by Linux
    } else {
        for (int y = 0; y < MAT_H; y++)
            px(12, y, blink ? 6 : 0);
    }
    if (level == LEVEL_FLAME)
        px(11, 7, 7);
    matrix.draw(frame_buf);
}

static void buzz(uint32_t now, uint8_t level)
{
    bool on = false;
    if ((int32_t)(now - muted_until) >= 0) {
        if (level == LEVEL_FLAME)
            on = (now % 1000) < 500;
        else if (level == LEVEL_SMOKE)
            on = (now % 5000) < 150;
    }
    digitalWrite(BUZZER_PIN, on ? HIGH : LOW);
}

void setup()
{
    Bridge.begin();
    Monitor.begin(115200);
    matrix.begin();
    matrix.setGrayscaleBits(3);
    matrix.clear();
    pinMode(BUZZER_PIN, OUTPUT);
    pinMode(MUTE_PIN, INPUT_PULLUP);

    Bridge.provide_safe("heartbeat", on_heartbeat);
    Bridge.provide_safe("set_tiles", on_tiles);

    runtime_selftest();
#if SENTINEL_THERMAL
    if (!mlx.begin(MLX90640_I2CADDR_DEFAULT, &Wire1)) {
        Monitor.println("thermal: MLX90640 not found on Qwiic");
    } else {
        mlx.setMode(MLX90640_CHESS);
        mlx.setResolution(MLX90640_ADC_18BIT);
        mlx.setRefreshRate(MLX90640_8_HZ);  // two subpages per frame -> ~4 Hz
    }
#endif
}

void loop()
{
    const uint32_t now = millis();
    const bool linux_alive = linux_seen && now - last_heartbeat_ms < HEARTBEAT_TIMEOUT_MS;
    if (digitalRead(MUTE_PIN) == LOW)
        muted_until = now + MUTE_MS;

    thermal_step(now);

    uint8_t level = thermal_level;
    if (linux_alive)
        for (int i = 0; i < N_TILES; i++)
            if (tile_levels[i] > level)
                level = tile_levels[i];

    draw(now, linux_alive, level);
    buzz(now, level);
    delay(20);
}
