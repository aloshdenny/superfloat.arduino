"""Wildfire sentinel: Arduino App Lab entry point.

On the UNO Q this runs inside App Lab's Python container with the app folder
mounted at /app. It also runs on a laptop against recorded footage, which is
how the alerting logic is tuned before a board is in the field:

    python main.py --replay hpwren_clip.mp4 --model ../models/smokenet.sfm

Nothing in the loop stores or transmits images. Frames are tiled, scored and
dropped; only alert metadata (tile index, score, time) leaves the process.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))  # sfedge + libsfrt, copied in by deploy.sh

from sentinel.alerts import Alert, AlertQueue, LogTransport, WebhookTransport  # noqa: E402
from sentinel.board import ArduinoBoard, NullBoard  # noqa: E402
from sentinel.config import Config, load_config  # noqa: E402
from sentinel.detector import TileDetector  # noqa: E402
from sentinel.temporal import TemporalVoter  # noqa: E402

from sfedge.tiling import TileGrid  # noqa: E402

log = logging.getLogger("sentinel")

STATE_OK, STATE_CAMERA_FAULT = 0, 1


class Sentinel:
    def __init__(self, cfg: Config, board, transports=None):
        self.cfg = cfg
        self.board = board
        self.grid = TileGrid(cfg.grid_cols, cfg.grid_rows, cfg.tile)
        lib = cfg.lib or _vendored_lib()
        self.detector = TileDetector(cfg.model, self.grid, cfg.threads, lib=lib, mask=cfg.mask_tiles)
        self.voter = TemporalVoter(self.grid.count, cfg.voter)
        if transports is None:
            transports = [LogTransport(log.info)]
            if cfg.webhook_url:
                transports.insert(0, WebhookTransport(cfg.webhook_url, token=cfg.webhook_token))
        self.queue = AlertQueue(cfg.queue_path, transports)
        self.seq = 0
        self.state = STATE_OK
        self.last_probs = None
        self._thermal = []
        self._lock = threading.Lock()
        board.on_thermal(self.on_thermal)

    # Called on the Bridge RPC thread: record and return, nothing else.
    def on_thermal(self, score_milli: int, x: int, y: int) -> bool:
        with self._lock:
            self._thermal.append((score_milli / 1000.0, int(x), int(y), time.time()))
        return True

    def _alert(self, kind: str, tile: int, score: float, t: float, source: str = "camera"):
        self.queue.push(Alert(kind=kind, tile=tile, score=score, node_id=self.cfg.node_id,
                              created=t, source=source, lat=self.cfg.lat, lon=self.cfg.lon))
        log.warning("%s %s on tile %d (score %.2f)", source, kind, tile, score)

    def step(self, frame, t: float | None = None) -> list:
        """Process one frame (or None when the camera failed)."""
        t = time.time() if t is None else t
        self.seq += 1
        events = []
        if frame is None:
            self.state = STATE_CAMERA_FAULT
        else:
            self.state = STATE_OK
            rgb = frame[..., ::-1] if self.cfg.camera_bgr else frame
            self.last_probs = self.detector.classify(rgb)
            events = self.voter.update(self.last_probs, t)
            for ev in events:
                if ev.kind != "clear":
                    self._alert(ev.kind, ev.tile, ev.score, ev.t)
        with self._lock:
            thermal, self._thermal = self._thermal, []
        for score, x, y, tt in thermal:
            self._alert("hotspot", -1, score, tt, source="thermal")
        self.board.heartbeat(self.seq, self.state)
        self.board.show(self.voter.levels)
        self.queue.flush()
        return events

    def status(self) -> dict:
        return {
            "node_id": self.cfg.node_id,
            "seq": self.seq,
            "state": self.state,
            "levels": self.voter.levels,
            "grid": [self.grid.cols, self.grid.rows],
            "frame_ms": round(self.detector.last_ms, 1),
            "pending_alerts": len(self.queue.pending),
            "bridge_errors": getattr(self.board, "errors", 0),
        }


def _vendored_lib():
    for name in ("libsfrt.so", "libsfrt.dylib"):
        p = HERE / "vendor" / name
        if p.exists():
            return str(p)
    return None


def camera_frames(cfg: Config):
    from arduino.app_peripherals.camera import Camera  # App Lab only

    cam = Camera(source=cfg.camera, resolution=tuple(cfg.resolution), fps=max(1, int(1 / cfg.period_s)))
    cam.start()
    try:
        while True:
            yield cam.capture()
    finally:
        cam.stop()


def file_frames(path: str, every_s: float):
    import cv2

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(fps * every_s))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        if i % stride == 0:
            yield frame
        i += 1


def start_status_page(sentinel: Sentinel, port: int) -> None:
    try:
        from arduino.app_bricks.web_ui import WebUI
    except ImportError:
        log.info("web_ui brick not available; status page disabled")
        return
    ui = WebUI(port=port)
    ui.expose_api("GET", "/api/status", sentinel.status)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--model", help="override the model path from the config")
    p.add_argument("--replay", help="run on a video file instead of the camera (no board)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    if args.model:
        cfg.model = args.model

    if args.replay:
        if cfg.queue_path.startswith("/app/"):
            cfg.queue_path = "alerts.jsonl"
        sentinel = Sentinel(cfg, NullBoard())
        for i, frame in enumerate(file_frames(args.replay, cfg.period_s)):
            sentinel.step(frame, t=i * cfg.period_s)
            print(sentinel.status(), flush=True)
        return 0

    from arduino.app_utils import App  # App Lab only

    sentinel = Sentinel(cfg, ArduinoBoard())
    start_status_page(sentinel, cfg.status_port)
    frames = camera_frames(cfg)

    def loop():
        t0 = time.monotonic()
        sentinel.step(next(frames))
        time.sleep(max(0.0, cfg.period_s - (time.monotonic() - t0)))

    App.run(user_loop=loop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
