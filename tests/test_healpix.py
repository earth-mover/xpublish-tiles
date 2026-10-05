import healpix_geo.nested as hpn
import numpy as np
import pytest
from pyproj.aoi import BBox

from xpublish_tiles.config import config
from xpublish_tiles.grids import (
    HealpixCube,
    HealpixCubeIndexer,
    UnsupportedGridError,
    find_healpix_cube_dims,
    guess_grid_metadata,
    guess_grid_system,
)
from xpublish_tiles.healpix import (
    antimeridian_cells,
    bbox_cell_ids,
    coarse_level,
    fyx_to_nested,
    nested_to_fyx,
)
from xpublish_tiles.testing.datasets import (
    GLOBAL_HEALPIX_CUBE_L3,
    GLOBAL_HEALPIX_CUBE_L5,
    _create_global_healpix,
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


def test_cube_fixture_matches_1d():
    cube = GLOBAL_HEALPIX_CUBE_L3.create()
    flat = _create_global_healpix(level=3, dtype=np.float64)
    f, y, x = nested_to_fyx(np.arange(12 * 4**3), 3)
    got = cube["foo"].isel(time=-1).values[f, y, x]
    np.testing.assert_array_equal(got, flat["foo"].values)


def test_detect_cube():
    ds = GLOBAL_HEALPIX_CUBE_L3.create()
    assert find_healpix_cube_dims(ds) == ("face", "y", "x")
    meta = guess_grid_metadata(ds)
    assert meta is not None and meta.grid_cls is HealpixCube
    grid = guess_grid_system(ds, "foo")
    assert isinstance(grid, HealpixCube)
    assert grid.dims == {"face", "y", "x"}
    assert grid.level == 3


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda ds: ds.attrs.pop("healpix_nside"), id="no-attr"),
        pytest.param(lambda ds: ds.attrs.update(healpix_nside=4), id="attr-mismatch"),
    ],
)
def test_detect_cube_negative(mutate):
    ds = GLOBAL_HEALPIX_CUBE_L3.create()
    mutate(ds)
    assert find_healpix_cube_dims(ds) is None


def test_detect_cube_not_power_of_two():
    ds = GLOBAL_HEALPIX_CUBE_L3.create().isel(y=slice(0, 6), x=slice(0, 6))
    ds.attrs["healpix_nside"] = 6
    assert find_healpix_cube_dims(ds) is None


def test_detect_cube_bad_face_ordering():
    ds = GLOBAL_HEALPIX_CUBE_L3.create()
    ds.attrs["face_ordering"] = "other"
    with pytest.raises(UnsupportedGridError):
        find_healpix_cube_dims(ds)


def test_cube_select_no_coarsen_matches_1d_ids():
    grid = guess_grid_system(GLOBAL_HEALPIX_CUBE_L3.create(), "foo")
    assert isinstance(grid, HealpixCube)
    bbox = BBox(west=-30, south=-20, east=40, north=50)
    indexers = grid.select(bbox)
    assert all(isinstance(ix, HealpixCubeIndexer) for ix in indexers)
    assert all(ix.level == 3 and ix.factor == 1 for ix in indexers)
    got = np.concatenate([ix.cell_ids for ix in indexers])
    np.testing.assert_array_equal(np.sort(got), bbox_cell_ids(bbox, 3))


def test_cube_select_coarsen_and_gather():
    ds = GLOBAL_HEALPIX_CUBE_L5.create()
    grid = guess_grid_system(ds, "foo")
    assert isinstance(grid, HealpixCube)
    globe = BBox(west=-180, south=-90, east=180, north=90)
    with config.set({"max_num_geometries": 12 * 4**3}):
        indexers = grid.select(globe)
    assert {ix.level for ix in indexers} == {3}
    assert {ix.factor for ix in indexers} == {4}
    flat = _create_global_healpix(level=5, dtype=np.float64)["foo"].values
    expected = flat.reshape(-1, 16).mean(axis=1)  # nested children are contiguous
    for ix in indexers:
        rect = ds["foo"].isel(time=-1, face=ix.face, y=ix.y, x=ix.x).load()
        got = grid.gather(rect, ix)
        assert got.dims == (grid.dim,)
        np.testing.assert_allclose(
            got.values, expected[ix.cell_ids.astype(np.int64)], atol=1e-12
        )


def test_cube_select_discrete_not_coarsened():
    ds = GLOBAL_HEALPIX_CUBE_L5.create()
    grid = guess_grid_system(ds, "foo")
    assert isinstance(grid, HealpixCube)
    globe = BBox(west=-180, south=-90, east=180, north=90)
    with config.set({"max_num_geometries": 10}):
        indexers = grid.select(globe, allow_coarsen=False)
    assert {ix.level for ix in indexers} == {5}


def test_cube_aux_var_is_rejected_like_cubed_sphere():
    ds = GLOBAL_HEALPIX_CUBE_L3.create()
    ds["bar"] = ("time", np.arange(2.0))
    grid = guess_grid_system(ds, "foo")
    assert isinstance(grid, HealpixCube)
    assert "HealpixCube" in repr(grid)
    with pytest.raises(UnsupportedGridError):
        guess_grid_system(ds, "bar")


@pytest.mark.parametrize(
    "bbox, max_cells, coarsened",
    [
        (BBox(west=170, south=-30, east=190, north=30), 10**9, False),
        (BBox(west=-30, south=10, east=60, north=70), 10**9, False),
        (BBox(west=170, south=-60, east=260, north=60), 12 * 4**3, True),
    ],
)
def test_cube_partial_bbox_gather(bbox, max_cells, coarsened):
    ds = GLOBAL_HEALPIX_CUBE_L5.create()
    grid = guess_grid_system(ds, "foo")
    assert isinstance(grid, HealpixCube)
    with config.set({"max_num_geometries": max_cells}):
        indexers = grid.select(bbox)
    (factor,) = {ix.factor for ix in indexers}
    assert (factor > 1) == coarsened
    assert any(ix.y.start > 0 or ix.x.start > 0 for ix in indexers)
    flat = _create_global_healpix(level=5, dtype=np.float64)["foo"].values
    expected = flat.reshape(-1, factor**2).mean(axis=1)
    for ix in indexers:
        rect = ds["foo"].isel(time=-1, face=ix.face, y=ix.y, x=ix.x).load()
        got = grid.gather(rect, ix)
        np.testing.assert_allclose(
            got.values, expected[ix.cell_ids.astype(np.int64)], atol=1e-12
        )


def test_bbox_cell_ids_globe_returns_all():
    ids = bbox_cell_ids(BBox(west=-180, south=-90, east=180, north=90), 3)
    np.testing.assert_array_equal(ids, np.arange(12 * 4**3))


def test_bbox_cell_ids_full_width_strip():
    ids = bbox_cell_ids(BBox(west=-180, south=10, east=180, north=40), 3)
    core, _, _ = hpn.zone_coverage((0.0, 10, 360.0 - 1e-9, 40), 3, flat=True)
    expected = hpn.kth_neighbourhood(core.astype(np.uint64), 3, ring=1).ravel()
    np.testing.assert_array_equal(ids, np.unique(expected[expected >= 0]))


@pytest.mark.parametrize(
    "bbox",
    [
        BBox(west=-180, south=85, east=180, north=90),
        BBox(west=-180, south=-90, east=180, north=-60),
    ],
)
def test_cube_select_polar_strip_respects_budget(bbox):
    grid = HealpixCube(face_dim="face", Ydim="y", Xdim="x", level=9)
    indexers = grid.select(bbox)
    (level,) = {ix.level for ix in indexers}
    ids = np.concatenate([ix.cell_ids for ix in indexers])
    assert ids.size <= config.get("max_num_geometries")
    core, _, _ = hpn.zone_coverage(
        (0.0, bbox.south, 360.0 - 1e-9, bbox.north), level, flat=True
    )
    assert np.isin(core, ids).all()


def test_cube_select_globe_matches_generic_path():
    grid = HealpixCube(face_dim="face", Ydim="y", Xdim="x", level=3)
    fast = grid.select(BBox(west=-180, south=-90, east=180, north=90))
    # height < 180 forces the bbox_cell_ids path, which still covers every cell
    slow = grid.select(BBox(west=-180, south=-89.999, east=180, north=90))
    assert len(fast) == len(slow) == 12
    for a, b in zip(fast, slow, strict=True):
        assert (a.face, a.y, a.x, a.level, a.factor) == (
            b.face,
            b.y,
            b.x,
            b.level,
            b.factor,
        )
        for name in ("indices", "antimeridian_mask", "cell_ids", "ys", "xs"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_cube_equals():
    a = HealpixCube(face_dim="face", Ydim="y", Xdim="x", level=3)
    assert a == HealpixCube(face_dim="face", Ydim="y", Xdim="x", level=3)
    assert a != HealpixCube(face_dim="face", Ydim="y", Xdim="x", level=4)
    assert a != HealpixCube(face_dim="tile", Ydim="y", Xdim="x", level=3)


def test_antimeridian_cells_read_only():
    with pytest.raises(ValueError):
        antimeridian_cells(3)[0] = 0
