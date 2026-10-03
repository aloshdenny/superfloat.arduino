#pragma once
#include "Arduino.h"

struct HostBridge {
    std::function<bool(int, int)> heartbeat;
    std::function<bool(std::vector<uint8_t>)> tiles;
    std::vector<std::string> notified;

    bool begin(unsigned long = 0) { return true; }
    void provide_safe(const char *name, bool (*fn)(int, int)) {
        if (std::string(name) == "heartbeat") heartbeat = fn;
    }
    void provide_safe(const char *name, bool (*fn)(std::vector<uint8_t>)) {
        if (std::string(name) == "set_tiles") tiles = fn;
    }
    template <typename... A> void notify(const char *name, A...) { notified.push_back(name); }
};
extern HostBridge Bridge;
