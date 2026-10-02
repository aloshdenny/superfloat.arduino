import numpy as np
import pytest

from sfedge.tiling import CLEAR, FLAME, SMOKE, TileGrid, label_tiles, yolo_to_corners


def test_split_is_row_major_and_lossless():
    g = TileGrid(cols=3, rows=2, tile=4)
    w, h = g.size
    frame = np.arange(h * w * 3, dtype=np.int64).reshape(h, w, 3)
    tiles = g.split(frame)
    assert tiles.shape == (6, 4, 4, 3)
    np.testing.assert_array_equal(tiles[1], frame[0:4, 4:8])
    np.testing.assert_array_equal(tiles[3], frame[4:8, 0:4])


def test_split_rejects_wrong_size():
    with pytest.raises(ValueError):
        TileGrid(2, 2, 4).split(np.zeros((9, 8, 3), np.uint8))


def test_fit_resizes_to_grid():
    pytest.importorskip("PIL")
    g = TileGrid(5, 3, 16)
    out = g.fit(np.zeros((72, 128, 3), np.uint8))
    assert out.shape == (48, 80, 3)


def test_cells_cover_frame():
    g = TileGrid(4, 3)
    assert sum((c[2] - c[0]) * (c[3] - c[1]) for c in g.cells()) == pytest.approx(1.0)


def test_label_priority_and_thresholds():
    g = TileGrid(cols=2, rows=1)
    # big smoke box over both tiles, a flame inside the right tile
    boxes = [(SMOKE, 0.1, 0.1, 0.9, 0.9), (FLAME, 0.7, 0.4, 0.8, 0.6)]
    assert label_tiles(g, boxes) == [SMOKE, FLAME]


def test_small_distant_plume_counts_for_its_tile():
    g = TileGrid(cols=4, rows=4)
    tiny = (SMOKE, *yolo_to_corners(0.1, 0.1, 0.02, 0.02))  # 0.6% of a tile
    labels = label_tiles(g, [tiny])
    assert labels[0] == SMOKE and labels.count(CLEAR) == 15


def test_grazing_tiles_are_ignored():
    g = TileGrid(cols=2, rows=1)
    # box mostly in the left tile, a sliver over the boundary
    box = (SMOKE, 0.1, 0.0, 0.51, 1.0)
    assert label_tiles(g, [box]) == [SMOKE, None]
