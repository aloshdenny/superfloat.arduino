#pragma once
#include "Arduino.h"
#include "Wire.h"

#define MLX90640_I2CADDR_DEFAULT 0x33
enum { MLX90640_CHESS = 1 };
enum { MLX90640_ADC_18BIT = 2 };
enum { MLX90640_8_HZ = 4 };

extern float host_thermal[32 * 24];

struct Adafruit_MLX90640 {
    bool begin(uint8_t, TwoWire *) { return true; }
    void setMode(int) {}
    void setResolution(int) {}
    void setRefreshRate(int) {}
    int getFrame(float *out) {
        memcpy(out, host_thermal, sizeof host_thermal);
        return 0;
    }
};
