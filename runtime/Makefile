# Builds the shared library used by the Python binding and the C self-test.
# Cross-compiling for the UNO Q from another machine:
#   make CC=aarch64-linux-gnu-gcc
CC ?= cc
CFLAGS ?= -O3
# Kept separate so `make CFLAGS=...` adds options instead of dropping these.
REQUIRED := -std=c99 -Wall -Wextra -Wpedantic -Werror -Iinclude
BUILD := build
SRCS := src/sf_kernels.c src/sf_model.c
HDRS := include/sf_runtime.h

ifeq ($(shell uname -s),Darwin)
  LIB := $(BUILD)/libsfrt.dylib
  SHARED := -dynamiclib
else
  LIB := $(BUILD)/libsfrt.so
  SHARED := -shared
endif

.PHONY: all test clean

all: $(LIB) $(BUILD)/selftest

$(BUILD):
	mkdir -p $(BUILD)

$(LIB): $(SRCS) $(HDRS) | $(BUILD)
	$(CC) $(REQUIRED) $(CFLAGS) -fPIC $(SHARED) -o $@ $(SRCS)

$(BUILD)/selftest: tests/selftest.c $(SRCS) $(HDRS) | $(BUILD)
	$(CC) $(REQUIRED) $(CFLAGS) -o $@ tests/selftest.c $(SRCS)

test: $(BUILD)/selftest
	./$(BUILD)/selftest

clean:
	rm -rf $(BUILD)
