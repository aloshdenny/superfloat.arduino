import numpy as np
import pytest

from sfedge import thermal
from sfedge.data import class_weights
from sfedge.metrics import confusion, report


def test_class_weights_mean_one_and_inverse():
    w = class_weights([100, 10, 0])
    assert w[2] == 0
    assert w[1] == pytest.approx(10 * w[0])
    assert w[:2].mean() == pytest.approx(1.0)


def test_tile_folder(tmp_path):
    pytest.importorskip("torch")
    from PIL import Image

    from sfedge.data import TileFolder

    for name, n in [("clear", 3), ("smoke", 2)]:
        (tmp_path / name).mkdir()
        for i in range(n):
            Image.fromarray(np.full((8, 8, 3), i, np.uint8)).save(tmp_path / name / f"{i}.png")
    ds = TileFolder(tmp_path, ["clear", "smoke", "flame"], augment=True)
    assert len(ds) == 5
    assert ds.counts().tolist() == [3, 2, 0]
    x, y = ds[4]
    assert x.shape == (8, 8, 3) and str(x.dtype) == "torch.uint8" and y == 1


def test_tile_folder_empty(tmp_path):
    from sfedge.data import TileFolder

    with pytest.raises(FileNotFoundError):
        TileFolder(tmp_path, ["clear"])


def test_thermal_encoding_constants():
    now = np.full((24, 32), 125.0)
    prev = np.full((24, 32), 100.0)
    codes = thermal.encode(now, prev)
    assert codes.shape == (24, 32, 2) and codes.dtype == np.int8
    assert (codes[..., 0] == 127).all() and (codes[..., 1] == 127).all()
    assert (thermal.encode(prev - 75, prev - 75)[..., 0] == 0).all()  # 25 C -> 0


def test_synthetic_thermal_is_separable_by_peak():
    x, y = thermal.synthetic_dataset(400, seed=1)
    assert 120 < y.sum() < 280
    peak = x[..., 0].reshape(len(x), -1).max(1)
    # not a trivial threshold problem, but hotspots should run hotter on average
    assert peak[y == 1].mean() > peak[y == 0].mean()


def test_confusion_and_report():
    cm = confusion([0, 0, 0, 1, 1, 2], [0, 1, 0, 1, 2, 2], 3)
    assert cm.tolist() == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]
    rep = report(cm, ["clear", "smoke", "flame"])
    assert rep["hazard_recall"] == 1.0  # the smoke tile called flame still counts
    assert rep["false_alarm_rate"] == pytest.approx(1 / 3)
    assert rep["per_class"]["smoke"]["recall"] == 0.5
