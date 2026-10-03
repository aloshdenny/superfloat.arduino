"""The sentinel loop, end to end on the host: config, tiling, C runtime,
voting, queueing and board calls, with a stand-in board."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "uno_q" / "wildfire-sentinel" / "python"))

from sentinel.config import Config, from_dict, load_config  # noqa: E402


class RecordingBoard:
    errors = 0

    def __init__(self):
        self.calls = []
        self.thermal_cb = None

    def heartbeat(self, seq, state):
        self.calls.append(("heartbeat", seq, state))

    def show(self, levels):
        self.calls.append(("show", list(levels)))

    def on_thermal(self, cb):
        self.thermal_cb = cb


def test_config_overrides_and_rejects_typos(tmp_path):
    cfg = from_dict({"node_id": "ridge-7", "resolution": [640, 480], "voter": {"smoke_k": 3}})
    assert cfg.node_id == "ridge-7" and cfg.resolution == (640, 480) and cfg.voter.smoke_k == 3
    with pytest.raises(ValueError, match="unknown config keys"):
        from_dict({"nodeid": "x"})
    with pytest.raises(ValueError, match="unknown voter keys"):
        from_dict({"voter": {"smoke_q": 1}})
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"period_s": 5}))
    assert load_config(p).period_s == 5


@pytest.fixture(scope="module")
def smoke_model(tmp_path_factory):
    torch = pytest.importorskip("torch")  # noqa: F841
    import subprocess

    subprocess.run(["make", "-s", "-C", str(ROOT / "runtime")], check=True)
    from sfedge import sfm
    from sfedge.export import to_graph
    from sfedge.models import smokenet
    from sfedge.qat import quantize_model

    path = tmp_path_factory.mktemp("m") / "smoke.sfm"
    path.write_bytes(sfm.dumps(to_graph(quantize_model(smokenet(width=0.5, tile=32).eval()))))
    return path


def test_loop_runs_with_stand_in_board(smoke_model, tmp_path):
    from main import Sentinel

    cfg = Config(model=str(smoke_model), tile=32, grid_cols=4, grid_rows=2, threads=2,
                 queue_path=str(tmp_path / "q.jsonl"))
    board = RecordingBoard()
    s = Sentinel(cfg, board)
    frame = np.random.default_rng(0).integers(0, 256, (90, 160, 3), dtype=np.uint8)
    s.step(frame, t=0.0)
    s.step(None, t=2.0)  # camera dropout must not crash the loop
    assert s.last_probs.shape == (8, 3)
    assert np.allclose(s.last_probs.sum(1), 1.0)
    assert ("heartbeat", 2, 1) in board.calls
    st = s.status()
    assert st["seq"] == 2 and len(st["levels"]) == 8


def test_thermal_reports_become_alerts(smoke_model, tmp_path):
    from main import Sentinel

    cfg = Config(model=str(smoke_model), tile=32, grid_cols=4, grid_rows=2, threads=1,
                 queue_path=str(tmp_path / "q.jsonl"))
    board = RecordingBoard()
    s = Sentinel(cfg, board, transports=[])  # no link: alerts stay queued
    assert board.thermal_cb(870, 12, 5) is True
    s.step(None, t=1.0)
    assert [(a.kind, a.source) for a in s.queue.pending] == [("hotspot", "thermal")]
    assert s.queue.pending[0].score == pytest.approx(0.87)


def test_detector_rejects_mismatched_grid(smoke_model):
    from sentinel.detector import TileDetector

    from sfedge.tiling import TileGrid

    with pytest.raises(ValueError, match="model expects"):
        TileDetector(smoke_model, TileGrid(4, 2, 64))


def test_masked_tiles_are_never_scored(smoke_model, tmp_path):
    from sentinel.detector import TileDetector

    from sfedge.tiling import TileGrid

    det = TileDetector(smoke_model, TileGrid(4, 2, 32), threads=1, mask=[0, 5])
    calls = []
    run = det.model.run_rgb8
    det.model.run_rgb8 = lambda t: calls.append(1) or run(t)
    probs = det.classify(np.random.default_rng(1).integers(0, 256, (64, 128, 3), dtype=np.uint8))
    assert len(calls) == 6
    assert probs[0].tolist() == [1.0, 0.0, 0.0] and probs[5].tolist() == [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="outside"):
        TileDetector(smoke_model, TileGrid(4, 2, 32), mask=[8])


def test_example_config_parses():
    cfg = load_config(ROOT / "uno_q" / "wildfire-sentinel" / "config.example.json")
    assert cfg.grid_cols * cfg.grid_rows == 15 and cfg.voter.smoke_k == 4


def test_night_slows_the_camera_tier(smoke_model, tmp_path):
    from main import Sentinel

    cfg = Config(model=str(smoke_model), tile=32, grid_cols=4, grid_rows=2, threads=1,
                 queue_path=str(tmp_path / "q.jsonl"), period_s=2.0, night_period_s=10.0)
    s = Sentinel(cfg, RecordingBoard())
    s.step(np.full((64, 128, 3), 140, np.uint8), t=0.0)
    assert not s.night and s.period() == 2.0
    s.step(np.full((64, 128, 3), 8, np.uint8), t=2.0)
    assert s.night and s.period() == 10.0 and s.status()["night"] is True
