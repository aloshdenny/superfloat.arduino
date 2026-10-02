import numpy as np
import pytest

from sfedge import reference as ref
from sfedge import sfm
from tests.conftest import random_graph


@pytest.mark.parametrize("wbits", [8, 4])
def test_roundtrip_preserves_outputs(wbits, tmp_path):
    g = random_graph(seed=5, wbits=wbits)
    path = tmp_path / "m.sfm"
    sfm.save(g, path)
    g2 = sfm.load(path)
    assert g2.name == g.name and g2.labels == g.labels
    assert g2.input_shape == g.input_shape
    x = np.random.default_rng(1).integers(-128, 128, g.input_shape).astype(np.int8)
    np.testing.assert_array_equal(ref.run(g, x), ref.run(g2, x))


def test_sf4_file_is_smaller():
    a = len(sfm.dumps(random_graph(seed=2, wbits=8)))
    b = len(sfm.dumps(random_graph(seed=2, wbits=4)))
    assert b < a


def test_payloads_are_4_byte_aligned():
    assert len(sfm.dumps(random_graph())) % 4 == 0


def test_corruption_is_detected():
    data = bytearray(sfm.dumps(random_graph()))
    data[100] ^= 0x01
    with pytest.raises(ValueError, match="checksum"):
        sfm.loads(bytes(data))


def test_truncation_is_detected():
    data = sfm.dumps(random_graph())
    with pytest.raises(ValueError):
        sfm.loads(data[:20])


def test_rejects_wrong_magic():
    import struct
    import zlib

    body = bytearray(sfm.dumps(random_graph())[:-4])
    body[:4] = b"XXXX"
    with pytest.raises(ValueError, match="not an .sfm"):
        sfm.loads(bytes(body) + struct.pack("<I", zlib.crc32(body)))
