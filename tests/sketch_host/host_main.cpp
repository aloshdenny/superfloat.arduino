/* Drives the sketch through boot, a live Linux side, a flame tile, the mute
 * button and a Linux outage, checking the matrix and buzzer at each step. */
#include "Arduino.h"
#include "Arduino_RouterBridge.h"
#include "Arduino_LED_Matrix.h"

uint32_t host_millis = 0;
int host_pins[64];
HostMonitor Monitor;
HostBridge Bridge;
#if SENTINEL_THERMAL
#include "Wire.h"
TwoWire Wire1;
float host_thermal[32 * 24];
#endif

#include "sketch.ino"

static int failures;
#define CHECK(c, msg)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf("FAIL: %s\n", msg);                                         \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static void run_ms(uint32_t ms)
{
    const uint32_t end = host_millis + ms;
    while (host_millis < end)
        loop();
}

int main()
{
    host_pins[MUTE_PIN] = HIGH;  // pulled up, not pressed
    setup();
    CHECK(ember_ok, "EmberNet loads from flash without an arena");

    Bridge.heartbeat(1, 0);
    run_ms(100);
    CHECK(matrix.last[0] == 1, "clear tile drawn dim");
    CHECK(host_pins[BUZZER_PIN] == LOW, "silent when clear");

    std::vector<uint8_t> levels(N_TILES, 0);
    levels[7] = LEVEL_FLAME;
    Bridge.tiles(levels);
    bool sounded = false;
    for (int i = 0; i < 60; i++) {
        run_ms(20);
        sounded |= host_pins[BUZZER_PIN] == HIGH;
    }
    CHECK(sounded, "flame tile sounds the buzzer");
    CHECK(matrix.last[11 + 7 * 13] == 7, "flame marker lit");

    host_pins[MUTE_PIN] = LOW;
    run_ms(40);
    host_pins[MUTE_PIN] = HIGH;
    bool quiet = true;
    for (int i = 0; i < 100; i++) {
        run_ms(20);
        quiet &= host_pins[BUZZER_PIN] == LOW;
    }
    CHECK(quiet, "mute button silences the buzzer");

    run_ms(HEARTBEAT_TIMEOUT_MS + 1000);  // Linux goes quiet
    bool fault_shown = false;
    for (int i = 0; i < 40; i++) {
        run_ms(20);
        fault_shown |= matrix.last[12 + 3 * 13] == 6;
    }
    CHECK(fault_shown, "Linux outage shown on the right edge");
    CHECK(matrix.last[7 % 5 * 2 + 7 / 5 * 2 * 13] == 1, "stale camera tiles are not shown during an outage");

#if SENTINEL_THERMAL
    /* thermal tier, with Linux down: ambient scene, then a flickering hotspot */
    for (int i = 0; i < 32 * 24; i++)
        host_thermal[i] = 18.0f + (i / 32) * 0.3f;
    run_ms(3000);
    CHECK(Bridge.notified.empty(), "no thermal alert on an ambient scene");
    CHECK(thermal_level == LEVEL_CLEAR, "thermal tier clear on ambient scene");
    for (int step = 0; step < 16; step++) {
        const float peak = step % 2 ? 380.0f : 260.0f;
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
                host_thermal[(12 + dy) * 32 + 20 + dx] = dx || dy ? peak * 0.4f : peak;
        run_ms(260);
    }
    CHECK(thermal_level == LEVEL_FLAME, "hotspot confirmed by EmberNet on the MCU");
    CHECK(!Bridge.notified.empty() && Bridge.notified[0] == "thermal_hotspot", "Linux notified of hotspot");
    CHECK(thermal_milli > 500, "hotspot probability above one half");
#endif

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("sketch host test: all checks passed\n");
    return 0;
}
