"""The C runtime must agree with the numpy reference bit for bit."""

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from sfedge import reference as ref
from sfedge import sfm
from tests.conftest import random_graph

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def runtime():
    if shutil.which("make") is None or shutil.which("cc") is None:
        pytest.skip("no C toolchain")
    subprocess.run(["make", "-s", "-C", str(ROOT / "runtime")], check=True)
    from sfedge import runtime as rt

    return rt


@pytest.mark.parametrize("wbits", [8, 4])
@pytest.mark.parametrize("seed", range(4))
def test_matches_reference(runtime, wbits, seed):
    g = random_graph(seed=seed, wbits=wbits)
    model = runtime.Model(sfm.dumps(g))
    assert model.raw_output and model.n_out == len(g.labels)
    rng = np.random.default_rng(100 + seed)
    for _ in range(5):
        x = rng.integers(-128, 128, g.input_shape).astype(np.int8)
        np.testing.assert_array_equal(model.run(x), ref.run(g, x))


def test_int8_output_is_widened(runtime):
    from sfedge.graph import ACT_RELU1

    g = random_graph(seed=9)
    g.ops[-1].act = ACT_RELU1
    model = runtime.Model(sfm.dumps(g))
    assert not model.raw_output
    x = np.random.default_rng(0).integers(-128, 128, g.input_shape).astype(np.int8)
    np.testing.assert_array_equal(model.run(x), ref.run(g, x).astype(np.int32))


def test_rgb8_entry_point(runtime):
    g = random_graph(seed=1)
    model = runtime.Model(sfm.dumps(g))
    img = np.random.default_rng(2).integers(0, 256, g.input_shape).astype(np.uint8)
    np.testing.assert_array_equal(model.run_rgb8(img), ref.run_rgb8(g, img))


def test_threads_share_one_model(runtime):
    g = random_graph(seed=4)
    model = runtime.Model(sfm.dumps(g))
    rng = np.random.default_rng(3)
    xs = [rng.integers(-128, 128, g.input_shape).astype(np.int8) for _ in range(32)]
    with ThreadPoolExecutor(4) as pool:
        got = list(pool.map(model.run, xs))
    for x, y in zip(xs, got):
        np.testing.assert_array_equal(y, ref.run(g, x))


def test_corrupt_model_is_rejected(runtime):
    data = bytearray(sfm.dumps(random_graph()))
    data[-1] ^= 0xFF
    with pytest.raises(ValueError):  # caught by the Python parser first
        runtime.Model(bytes(data))


def test_wrong_input_shape(runtime):
    model = runtime.Model(sfm.dumps(random_graph()))
    with pytest.raises(ValueError):
        model.run(np.zeros((2, 2, 3), np.int8))


def test_cli_runner_matches_reference(runtime, tmp_path):
    g = random_graph(seed=6, wbits=4)
    (tmp_path / "m.sfm").write_bytes(sfm.dumps(g))
    x = np.random.default_rng(5).integers(-128, 128, g.input_shape).astype(np.int8)
    x.tofile(tmp_path / "x.bin")
    out = subprocess.run([str(ROOT / "runtime" / "build" / "sfrt_run"), str(tmp_path / "m.sfm"),
                          str(tmp_path / "x.bin")], check=True, capture_output=True, text=True)
    got = np.array(out.stdout.split(), dtype=np.int64)
    np.testing.assert_array_equal(got, ref.run(g, x))
