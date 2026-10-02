import numpy as np
import pytest

from sfedge import reference as ref
from sfedge.graph import ACT_LINEAR, ACT_RAW, ACT_RELU1, Dense
from tests.conftest import random_conv, random_graph


def naive_conv(x, op):
    """Direct seven-loop convolution, written independently of reference.conv2d."""
    h, w, c = x.shape
    kh, kw = op.kernel
    oh, ow, cout = op.out_shape(h, w)
    cin_g = c // op.groups
    cout_g = cout // op.groups
    wt = ref.sf8_weight(op).astype(np.int64)
    out = np.zeros((oh, ow, cout), dtype=np.int64)
    for oy in range(oh):
        for ox in range(ow):
            for oc in range(cout):
                g = oc // cout_g
                acc = int(op.bias[oc])
                for ky in range(kh):
                    for kx in range(kw):
                        iy = oy * op.stride + ky - op.pad
                        ix = ox * op.stride + kx - op.pad
                        if 0 <= iy < h and 0 <= ix < w:
                            for ic in range(cin_g):
                                acc += int(x[iy, ix, g * cin_g + ic]) * int(wt[oc, ky, kx, ic])
                out[oy, ox, oc] = acc
    return ref.requantize(out, op.shift, op.act)


def test_rshift_round_is_half_up():
    v = np.array([-3, -2, -1, 0, 1, 2, 3]) * 64  # multiples of half a step at s=7
    np.testing.assert_array_equal(ref.rshift_round(v, 7), [-1, -1, 0, 0, 1, 1, 2])


def test_rshift_round_negative_shift_is_left_shift():
    np.testing.assert_array_equal(ref.rshift_round([3, -3], -2), [12, -12])


@pytest.mark.parametrize("d", [1, 3, 16, 36, 49])
def test_div_round_matches_float_half_up(d):
    v = np.arange(-500, 500)
    want = np.floor(v / d + 0.5).astype(np.int64)
    np.testing.assert_array_equal(ref.div_round(v, d), want)


def test_rgb8_to_sf8():
    np.testing.assert_array_equal(ref.rgb8_to_sf8([0, 128, 255]), [-128, 0, 127])


@pytest.mark.parametrize(
    "kw",
    [
        dict(cin=3, cout=4, k=3, stride=1),
        dict(cin=3, cout=4, k=3, stride=2),
        dict(cin=6, cout=6, k=3, groups=6),
        dict(cin=4, cout=8, k=1, act=ACT_LINEAR),
        dict(cin=4, cout=4, k=5, stride=2, groups=2),
        dict(cin=3, cout=4, k=3, wbits=4),
    ],
)
def test_conv_matches_naive_loops(rng, kw):
    op = random_conv(rng, **kw)
    x = rng.integers(-128, 128, (7, 9, kw["cin"])).astype(np.int8)
    np.testing.assert_array_equal(ref.conv2d(x, op), naive_conv(x, op))


def test_relu1_bounds(rng):
    op = random_conv(rng, 3, 8)
    y = ref.conv2d(rng.integers(-128, 128, (8, 8, 3)).astype(np.int8), op)
    assert y.min() >= 0 and y.max() <= 127


def test_global_avg_pool_rounds_half_up():
    x = np.array([[[1], [2]]], dtype=np.int8)  # mean 1.5 -> 2
    np.testing.assert_array_equal(ref.global_avg_pool(x), [2])
    x = np.array([[[-1], [-2]]], dtype=np.int8)  # mean -1.5 -> -1
    np.testing.assert_array_equal(ref.global_avg_pool(x), [-1])


def test_dense_raw_returns_int32_accumulator(rng):
    op = Dense(np.array([[1, 2], [3, -4]], np.int8), np.array([10, -10], np.int32), act=ACT_RAW)
    out = ref.dense(np.array([5, 6], np.int8), op)
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, [27, -19])


@pytest.mark.parametrize("wbits", [8, 4])
def test_random_graph_runs_and_traces(wbits):
    g = random_graph(seed=3, wbits=wbits)
    x = np.random.default_rng(0).integers(-128, 128, g.input_shape).astype(np.int8)
    trace = ref.run(g, x, trace=True)
    assert len(trace) == len(g.ops)
    assert [t.shape for t in trace] == [tuple(s) for s in g.shapes()[1:]]
    np.testing.assert_array_equal(trace[-1], ref.run(g, x))


def test_run_rejects_wrong_input():
    g = random_graph()
    with pytest.raises(ValueError):
        ref.run(g, np.zeros(g.input_shape, np.uint8))
