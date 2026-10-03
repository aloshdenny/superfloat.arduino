import subprocess
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from sfedge import sfm  # noqa: E402
from sfedge.export import to_graph  # noqa: E402
from sfedge.models import smokenet  # noqa: E402
from sfedge.qat import quantize_model  # noqa: E402
from tools import eval as eval_tool  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_eval_runs_on_c_runtime(tmp_path, capsys):
    subprocess.run(["make", "-s", "-C", str(ROOT / "runtime")], check=True)
    model = tmp_path / "m.sfm"
    model.write_bytes(sfm.dumps(to_graph(quantize_model(smokenet(width=0.5, tile=32).eval()))))
    rng = np.random.default_rng(0)
    for name in ("clear", "smoke"):
        (tmp_path / "tiles" / name).mkdir(parents=True)
        for i in range(3):
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
                tmp_path / "tiles" / name / f"{i}.png")
    out = tmp_path / "rep.json"
    assert eval_tool.main([str(model), str(tmp_path / "tiles"), "--json", str(out)]) == 0
    assert "on 6 tiles" in capsys.readouterr().out
    assert out.exists()
