/* Minimal host stand-ins for the Arduino APIs the sentinel sketch uses, so
 * it can be compiled and exercised off-target (tests/test_sketch_host.py).
 * Behaviour is only as deep as the tests need. */
#pragma once
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <functional>
#include <map>
#include <string>
#include <vector>

#define HIGH 1
#define LOW 0
#define OUTPUT 1
#define INPUT_PULLUP 2

extern uint32_t host_millis;
inline uint32_t millis() { return host_millis; }
inline uint32_t micros() { return host_millis * 1000u; }
inline void delay(uint32_t ms) { host_millis += ms; }

extern int host_pins[64];
inline void pinMode(int, int) {}
inline void digitalWrite(int pin, int v) { host_pins[pin] = v; }
inline int digitalRead(int pin) { return host_pins[pin]; }

struct HostMonitor {
    void begin(unsigned long) {}
    template <typename T> void print(T v) { std::string s = fmt(v); fputs(s.c_str(), stdout); }
    template <typename T> void println(T v) { print(v); fputs("\n", stdout); }
    static std::string fmt(const char *s) { return s; }
    static std::string fmt(long long v) { return std::to_string(v); }
    static std::string fmt(int v) { return std::to_string(v); }
    static std::string fmt(unsigned v) { return std::to_string(v); }
    static std::string fmt(long v) { return std::to_string(v); }
    static std::string fmt(unsigned long v) { return std::to_string(v); }
};
extern HostMonitor Monitor;
