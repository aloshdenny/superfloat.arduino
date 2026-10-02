import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "uno_q" / "wildfire-sentinel" / "python"))

from sentinel.alerts import Alert, AlertQueue, LogTransport, Transport  # noqa: E402


class Flaky(Transport):
    def __init__(self):
        self.up = False
        self.sent = []

    def send(self, alert):
        if self.up:
            self.sent.append(alert.id)
        return self.up


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make_alert(kind="smoke"):
    return Alert(kind=kind, tile=3, score=0.9, node_id="n1", created=0.0)


def test_alert_survives_restart_until_delivered(tmp_path):
    link, clock = Flaky(), Clock()
    q = AlertQueue(tmp_path / "alerts.jsonl", [link], clock=clock)
    a = make_alert()
    q.push(a)
    assert q.flush() == 0

    # power cycle while the link is down
    q2 = AlertQueue(tmp_path / "alerts.jsonl", [link], clock=clock)
    assert [p.id for p in q2.pending] == [a.id]

    link.up = True
    clock.t += 10_000
    assert q2.flush() == 1 and link.sent == [a.id]
    assert AlertQueue(tmp_path / "alerts.jsonl", [link], clock=clock).pending == []


def test_backoff_grows_and_caps(tmp_path):
    link, clock = Flaky(), Clock()
    q = AlertQueue(tmp_path / "a.jsonl", [link], base_backoff=10, max_backoff=60, clock=clock)
    q.push(make_alert())
    delays = []
    for _ in range(5):
        q.flush()
        delays.append(q.pending[0].next_try - clock.t)
        clock.t = q.pending[0].next_try
    assert delays == [10, 20, 40, 60, 60]


def test_falls_through_to_next_transport(tmp_path):
    down, log = Flaky(), []
    q = AlertQueue(tmp_path / "a.jsonl", [down, LogTransport(log.append)])
    q.push(make_alert("flame"))
    assert q.flush() == 1 and "ALERT flame" in log[0]


def test_torn_last_line_is_ignored(tmp_path):
    q = AlertQueue(tmp_path / "a.jsonl", [Flaky()])
    q.push(make_alert())
    with open(tmp_path / "a.jsonl", "a") as f:
        f.write('{"kind": "smo')  # power lost mid-write
    assert len(AlertQueue(tmp_path / "a.jsonl", [Flaky()]).pending) == 1


def test_compact_keeps_only_pending(tmp_path):
    link = Flaky()
    q = AlertQueue(tmp_path / "a.jsonl", [link])
    q.push(make_alert())
    link.up = True
    q.flush()
    link.up = False
    keep = make_alert()
    q.push(keep)
    q.compact()
    assert [a.id for a in AlertQueue(tmp_path / "a.jsonl", [link]).pending] == [keep.id]
