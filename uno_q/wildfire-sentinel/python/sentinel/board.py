"""The MCU half of the UNO Q, as seen from Linux.

The STM32 owns everything that must keep working when Linux does not: the
LED matrix, the buzzer, the thermal tier and the heartbeat watchdog. Linux
tells it what the cameras see; if the heartbeats stop, the MCU raises its
own fault indication and carries on with the thermal tier alone.

Bridge failures are logged and counted, never raised: a wedged RPC link must
not stop the camera loop.
"""

from __future__ import annotations

import logging

log = logging.getLogger("sentinel.board")


class NullBoard:
    """Stand-in for laptop replay runs and tests."""

    errors = 0

    def heartbeat(self, seq: int, state: int) -> None:
        pass

    def show(self, levels) -> None:
        pass

    def on_thermal(self, callback) -> None:
        pass


class ArduinoBoard:
    def __init__(self, timeout: float = 2.0):
        from arduino.app_utils import Bridge  # only exists inside App Lab

        self.bridge = Bridge
        self.timeout = timeout
        self.errors = 0

    def _call(self, name: str, *args) -> None:
        try:
            self.bridge.call(name, *args, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001 - any RPC failure is non-fatal
            self.errors += 1
            log.warning("bridge %s failed (%d so far): %s", name, self.errors, exc)

    def heartbeat(self, seq: int, state: int) -> None:
        self._call("heartbeat", seq, state)

    def show(self, levels) -> None:
        """Per-tile levels (0 clear, 1 smoke, 2 flame) for the LED matrix."""
        self._call("set_tiles", bytes(int(v) for v in levels))

    def on_thermal(self, callback) -> None:
        """callback(score_milli, x, y), called on the RPC thread: keep it short
        and never call back into the Bridge from it."""
        self.bridge.provide("thermal_hotspot", callback)
