import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from tools import prepare_tiles  # noqa: E402


def test_end_to_end_on_tiny_dataset(tmp_path):
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    Image.fromarray(np.zeros((60, 100, 3), np.uint8)).save(root / "images" / "a.jpg")
    # smoke in the top-left tile, fire in the bottom-right tile of a 2x2 grid
    (root / "labels" / "a.txt").write_text("0 0.25 0.25 0.3 0.3\n1 0.75 0.75 0.3 0.3\n")
    Image.fromarray(np.zeros((60, 100, 3), np.uint8)).save(root / "images" / "b.jpg")  # negative

    out = tmp_path / "tiles"
    rc = prepare_tiles.main(["--root", str(root), "--out", str(out), "--cols", "2", "--rows", "2",
                             "--tile", "16", "--clear-keep", "1.0"])
    assert rc == 0
    assert sorted(p.name for p in (out / "smoke").iterdir()) == ["a_t00.png"]
    assert sorted(p.name for p in (out / "flame").iterdir()) == ["a_t03.png"]
    assert len(list((out / "clear").iterdir())) == 6
    assert Image.open(out / "smoke" / "a_t00.png").size == (16, 16)
