"""Store-and-forward alerting for links that come and go.

Wildfire country is where cellular coverage is worst and where it fails
first, so an alert is written to local storage before any attempt to send
it, and stays queued until a transport acknowledges it. Delivery is retried
with exponential backoff; the queue survives reboots and power loss (it is
an append-only JSON-lines log plus a small ack file, both fsynced).

Local signalling (LED matrix, buzzer) does not go through this queue: it is
driven over the Bridge immediately and does not depend on any network.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Alert:
    kind: str
    tile: int
    score: float
    node_id: str
    created: float
    source: str = "camera"  # "camera" | "thermal"
    lat: float | None = None
    lon: float | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    attempts: int = 0
    next_try: float = 0.0


class Transport:
    name = "transport"

    def send(self, alert: Alert) -> bool:
        raise NotImplementedError


class LogTransport(Transport):
    """Always succeeds; useful as the only transport when bench testing."""

    name = "log"

    def __init__(self, printer=print):
        self.printer = printer

    def send(self, alert: Alert) -> bool:
        self.printer(f"ALERT {alert.kind} tile={alert.tile} score={alert.score:.2f} id={alert.id}")
        return True


class WebhookTransport(Transport):
    """POST the alert as JSON. Any 2xx response is an acknowledgement."""

    name = "webhook"

    def __init__(self, url: str, timeout: float = 5.0, token: str | None = None):
        self.url = url
        self.timeout = timeout
        self.token = token

    def send(self, alert: Alert) -> bool:
        body = {k: v for k, v in asdict(alert).items() if k not in ("attempts", "next_try")}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return 200 <= resp.status < 300
        except OSError:
            return False


class AlertQueue:
    def __init__(self, path, transports, base_backoff=10.0, max_backoff=900.0, clock=time.time):
        self.path = Path(path)
        self.ack_path = self.path.with_suffix(self.path.suffix + ".ack")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.transports = list(transports)
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.clock = clock
        self.pending: list[Alert] = self._load()

    def _load(self) -> list[Alert]:
        acked = set(self.ack_path.read_text().split()) if self.ack_path.exists() else set()
        out = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    a = Alert(**json.loads(line))
                except (ValueError, TypeError):
                    continue  # a torn final line after power loss
                if a.id not in acked:
                    out.append(a)
        return out

    @staticmethod
    def _append(path: Path, text: str) -> None:
        with open(path, "a") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())

    def push(self, alert: Alert) -> None:
        self._append(self.path, json.dumps(asdict(alert)) + "\n")
        self.pending.append(alert)

    def flush(self) -> int:
        """Try every due alert on each transport in order; returns # delivered."""
        now = self.clock()
        delivered = 0
        for alert in list(self.pending):
            if alert.next_try > now:
                continue
            if any(t.send(alert) for t in self.transports):
                self._append(self.ack_path, alert.id + "\n")
                self.pending.remove(alert)
                delivered += 1
            else:
                alert.attempts += 1
                delay = min(self.max_backoff, self.base_backoff * 2 ** (alert.attempts - 1))
                alert.next_try = now + delay
        return delivered

    def compact(self) -> None:
        """Rewrite the log with only undelivered alerts (call when idle)."""
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(asdict(a)) + "\n" for a in self.pending))
        os.replace(tmp, self.path)
        self.ack_path.unlink(missing_ok=True)
