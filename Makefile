# Common tasks. The runtime has its own Makefile in runtime/.
PY ?= python3

.PHONY: all test runtime check bench clean

all: runtime

runtime:
	$(MAKE) -C runtime

test: runtime
	$(MAKE) -C runtime test
	$(PY) -m pytest -q

# what CI checks beyond the tests: the sketch's runtime copy is current
check:
	$(PY) tools/sync_sketch.py --check

bench: runtime
	$(PY) tools/bench.py --arch smokenet
	$(PY) tools/bench.py --arch embernet --tiles 1 --threads 1

clean:
	$(MAKE) -C runtime clean
