import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "uno_q" / "wildfire-sentinel" / "python"))

from sentinel.temporal import TemporalVoter, VoterConfig  # noqa: E402

CLEAR_P = [0.95, 0.04, 0.01]
SMOKE_P = [0.10, 0.85, 0.05]
FLAME_P = [0.05, 0.15, 0.80]


def frames(voter, seq, t0=0.0, dt=2.0):
    events = []
    for i, p in enumerate(seq):
        probs = np.tile(CLEAR_P, (len(voter.tiles), 1))
        probs[0] = p
        events += voter.update(probs, t0 + i * dt)
    return events


def test_single_frame_spike_does_not_alert():
    v = TemporalVoter(4)
    assert frames(v, [CLEAR_P, SMOKE_P, CLEAR_P, CLEAR_P, FLAME_P, CLEAR_P]) == []


def test_persistent_smoke_alerts_once():
    v = TemporalVoter(4)
    ev = frames(v, [SMOKE_P] * 10)
    assert [e.kind for e in ev] == ["smoke"]
    assert ev[0].tile == 0 and v.levels[0] == 1


def test_flame_escalates_quickly():
    v = TemporalVoter(4)
    ev = frames(v, [SMOKE_P] * 6 + [FLAME_P] * 3)
    assert [e.kind for e in ev] == ["smoke", "flame"]


def test_all_clear_after_quiet_period():
    cfg = VoterConfig(clear_frames=5)
    v = TemporalVoter(2, cfg)
    ev = frames(v, [SMOKE_P] * 6 + [CLEAR_P] * 5)
    assert [e.kind for e in ev] == ["smoke", "clear"]
    assert v.levels[0] == 0


def test_cooldown_suppresses_flapping():
    cfg = VoterConfig(clear_frames=2, cooldown_s=1000)
    v = TemporalVoter(1, cfg)
    seq = ([SMOKE_P] * 6 + [CLEAR_P] * 2) * 3
    kinds = [e.kind for e in frames(v, seq)]
    assert kinds.count("smoke") == 1  # re-detections inside the cooldown stay silent


def test_shape_is_checked():
    import pytest

    with pytest.raises(ValueError):
        TemporalVoter(3).update(np.zeros((2, 3)), 0.0)
