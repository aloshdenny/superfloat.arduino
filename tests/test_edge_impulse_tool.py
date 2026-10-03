"""The parts of the Edge Impulse tool that need no account or network."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from sfedge.models import smokenet  # noqa: E402
from sfedge.qat import quantize_model  # noqa: E402
from tools import edge_impulse  # noqa: E402


def test_onnx_from_trained_checkpoint(tmp_path):
    model = quantize_model(smokenet(width=0.5, tile=32).eval())
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)
    out = tmp_path / "twin.onnx"
    rc = edge_impulse.main(["onnx", str(ckpt), "--out", str(out), "--width", "0.5", "--tile", "32"])
    assert rc == 0 and out.stat().st_size > 10_000


def test_missing_api_key_is_reported(monkeypatch):
    monkeypatch.delenv("EI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="EI_API_KEY"):
        edge_impulse.api_key()
