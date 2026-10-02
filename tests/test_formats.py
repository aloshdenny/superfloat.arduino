import numpy as np
import pytest

from sfedge.formats import SF4, SF8, SF16, SFFormat, get_format


@pytest.mark.parametrize(
    "fmt, vmax",
    [(SF16, 0.999969482421875), (SF8, 0.9921875), (SF4, 0.875)],
)
def test_grid_constants_match_paper(fmt, vmax):
    assert fmt.vmax == vmax


@pytest.mark.parametrize("fmt", [SF4, SF8, SF16])
def test_encode_decode_roundtrip_on_grid(fmt):
    codes = np.arange(-fmt.qmax, fmt.qmax + 1)
    np.testing.assert_array_equal(fmt.encode(fmt.decode(codes)), codes)


@pytest.mark.parametrize("fmt", [SF4, SF8, SF16])
def test_encode_saturates_symmetrically(fmt):
    q = fmt.encode([-5.0, -1.0, 1.0, 5.0])
    np.testing.assert_array_equal(q, [-fmt.qmax, -fmt.qmax, fmt.qmax, fmt.qmax])


def test_encode_rounds_half_to_even():
    # 0.5 and 1.5 code units: half-to-even gives 0 and 2, matching torch.round
    np.testing.assert_array_equal(SF8.encode([0.5 / 128, 1.5 / 128]), [0, 2])


def test_storage_types():
    assert SF4.storage is np.int8
    assert SF8.storage is np.int8
    assert SF16.storage is np.int16


def test_quantize_error_is_bounded_by_half_step():
    x = np.random.default_rng(0).uniform(-0.99, 0.99, 10_000)
    err = np.abs(SF8.quantize(x) - x)
    assert err.max() <= 0.5 / SF8.scale + 1e-7


@pytest.mark.parametrize("spec", [8, "sf8", "SF8", SF8])
def test_get_format(spec):
    assert get_format(spec) == SF8


def test_rejects_out_of_schema_widths():
    with pytest.raises(ValueError):
        SFFormat(1)
    with pytest.raises(ValueError):
        SFFormat(17)
    with pytest.raises(ValueError):
        get_format("fp8")
