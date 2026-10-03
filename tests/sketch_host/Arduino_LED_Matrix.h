#pragma once
#include "Arduino.h"

struct Arduino_LED_Matrix {
    uint8_t last[104] = {0};
    int bits = 0;
    void begin() {}
    void clear() { memset(last, 0, sizeof last); }
    void setGrayscaleBits(int b) { bits = b; }
    void draw(const uint8_t *f) { memcpy(last, f, sizeof last); }
};
