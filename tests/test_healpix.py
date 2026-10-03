import healpix_geo.nested as hpn
import numpy as np
import pytest
from pyproj.aoi import BBox

from xpublish_tiles.healpix import (
    antimeridian_cells,
    bbox_cell_ids,
    coarse_level,
    fyx_to_nested,
    nested_to_fyx,
)


def _reference_fyx_to_nested(f, y, x, level):
    # zeus-healpix _fyx_to_healpix_idx, the writer's definition
    result = np.zeros_like(y, dtype=np.int64)
    for i in range(level):
        shift = level - 1 - i
        result |= ((y >> shift) & 1) << (2 * shift + 1)
        result |= ((x >> shift) & 1) << (2 * shift)
    return result + f * 4**level


@pytest.mark.parametrize("level", [0, 1, 5, 9])
def test_fyx_roundtrip(level):
    n = 2**level
    f, y, x = np.meshgrid(np.arange(12), np.arange(n), np.arange(n), indexing="ij")
    ids = fyx_to_nested(f.ravel(), y.ravel(), x.ravel(), level)
    assert ids.dtype == np.uint64
    np.testing.assert_array_equal(ids, _reference_fyx_to_nested(f, y, x, level).ravel())
    f2, y2, x2 = nested_to_fyx(ids, level)
    np.testing.assert_array_equal(f2, f.ravel())
    np.testing.assert_array_equal(y2, y.ravel())
    np.testing.assert_array_equal(x2, x.ravel())


def test_fyx_origin_is_south_vertex():
    # (x, y) = (0, 0) is the southernmost cell of each base face.
    level = 3
    n = 2**level
    for f in range(12):
        y, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        ids = fyx_to_nested(np.full(n * n, f), y.ravel(), x.ravel(), level)
        _, lat = hpn.healpix_to_lonlat(ids, depth=level, ellipsoid="sphere")
        assert np.argmin(np.asarray(lat)) == 0


def test_coarse_level_no_coarsening_when_small():
    bbox = BBox(west=0, south=0, east=1, north=1)
    assert coarse_level(bbox, 9, 1_500_000) == 9


def test_coarse_level_reduces_for_globe():
    globe = BBox(west=-180, south=-90, east=180, north=90)
    # 12 * 4**9 = 3_145_728 cells > 1.5M, so one level up.
    assert coarse_level(globe, 9, 1_500_000) == 8


def test_coarse_level_clamps():
    globe = BBox(west=-180, south=-90, east=180, north=90)
    assert coarse_level(globe, 5, 1) == 0


def test_bbox_cell_ids_sorted_unique_and_dilated():
    bbox = BBox(west=10, south=10, east=20, north=20)
    ids = bbox_cell_ids(bbox, 4)
    assert np.all(np.diff(ids.astype(np.int64)) > 0)
    core, _, _ = hpn.zone_coverage((10, 10, 20, 20), 4, flat=True)
    assert set(core.tolist()) < set(ids.tolist())


def test_antimeridian_cells_cached():
    assert antimeridian_cells(3) is antimeridian_cells(3)
    assert antimeridian_cells(3).size > 0
