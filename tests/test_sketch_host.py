"""Compile the App Lab sketch for the host against stub Arduino headers and
run it through boot, alerts, mute, a Linux outage and (with the thermal tier
enabled) a synthetic hotspot. Catches sketch breakage before a board is
involved; it does not replace an on-target build."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKETCH = ROOT / "uno_q" / "wildfire-sentinel" / "sketch"
STUBS = ROOT / "tests" / "sketch_host"


@pytest.fixture(scope="module")
def objects(tmp_path_factory):
    if not (shutil.which("cc") and shutil.which("c++")):
        pytest.skip("no C/C++ toolchain")
    out = tmp_path_factory.mktemp("sketch")
    objs = []
    for name in ("sf_kernels.c", "sf_model.c"):
        o = out / (name + ".o")
        subprocess.run(["cc", "-std=c99", "-O1", "-Wall", "-Wextra", "-Werror", "-c",
                        str(SKETCH / "src" / "sfrt" / name), "-o", str(o)], check=True)
        objs.append(str(o))
    return out, objs


@pytest.mark.parametrize("thermal", [0, 1])
def test_sketch_on_host(objects, thermal):
    out, objs = objects
    exe = out / f"sketch_host_{thermal}"
    subprocess.run(["c++", "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                    f"-DSENTINEL_THERMAL={thermal}", f"-I{STUBS}", f"-I{SKETCH}",
                    "-x", "c++", str(STUBS / "host_main.cpp"), "-x", "none", *objs,
                    "-o", str(exe)], check=True)
    res = subprocess.run([str(exe)], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "embernet-sf8 ok" in res.stdout


def test_vendored_runtime_is_in_sync():
    res = subprocess.run(["python3", str(ROOT / "tools" / "sync_sketch.py"), "--check"],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout
