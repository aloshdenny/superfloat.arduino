import numpy as np
import pytest

from sfedge.graph import ACT_LINEAR, ACT_RAW, ACT_RELU1, Conv2d, Dense, GlobalAvgPool, Graph


def random_conv(rng, cin, cout, k=3, stride=1, groups=1, act=ACT_RELU1, wbits=8, gain_exp=None):
    qmax = 7 if wbits == 4 else 127
    w = rng.integers(-qmax, qmax + 1, (cout, k, k, cin // groups)).astype(np.int8)
    b = rng.integers(-4000, 4000, cout).astype(np.int32)
    e = int(rng.integers(0, 3)) if gain_exp is None else gain_exp
    return Conv2d(w, b, stride=stride, pad=k // 2, groups=groups, gain_exp=e, act=act, wbits=wbits)


def random_graph(seed=0, wbits=8, size=(24, 20, 3), classes=3):
    """A small chain that exercises every op and option the runtime supports."""
    rng = np.random.default_rng(seed)
    ops = [
        random_conv(rng, size[2], 8, stride=2, wbits=wbits),
        random_conv(rng, 8, 8, groups=8, wbits=wbits),  # depthwise
        random_conv(rng, 8, 16, k=1, act=ACT_LINEAR, wbits=wbits),
        random_conv(rng, 16, 16, stride=2, groups=2, wbits=wbits, gain_exp=0),
        GlobalAvgPool(),
        Dense(rng.integers(-127, 128, (classes, 16)).astype(np.int8),
              rng.integers(-500, 500, classes).astype(np.int32), gain_exp=1, act=ACT_RAW),
    ]
    if wbits == 4:
        ops[-1].weight = np.clip(ops[-1].weight // 16, -7, 7).astype(np.int8)
        ops[-1].wbits = 4
    g = Graph(size, ops, labels=[f"c{i}" for i in range(classes)], name=f"random{seed}")
    g.validate()
    return g


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
