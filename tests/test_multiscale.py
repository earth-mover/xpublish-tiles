import morecantile
import pytest
from pyproj import CRS

import xarray as xr
from xarray import DataTree
from xpublish_tiles.config import config
from xpublish_tiles.multiscale import (
    get_crs,
    get_dataset,
    get_resolution_level,
    scan_resolution_levels,
)
from xpublish_tiles.testing.datasets import GEOZARR_MULTISCALE, NATIVE_AT_ROOT_MULTISCALE
from xpublish_tiles.types import OverviewSelectionStrategy


def test_multiscale_has_multiple_levels():
    geozarr_tree = GEOZARR_MULTISCALE.create()
    assert len(scan_resolution_levels(geozarr_tree)) >= 2

    native_root_tree = NATIVE_AT_ROOT_MULTISCALE.create()
    assert len(scan_resolution_levels(native_root_tree)) >= 2


def test_non_multiscale_has_fewer_than_two_levels():
    air_temp_ds = xr.tutorial.load_dataset("air_temperature")
    air_temp_tree = DataTree(dataset=air_temp_ds)
    assert len(scan_resolution_levels(air_temp_tree)) < 2

    empty_tree = DataTree()
    assert len(scan_resolution_levels(empty_tree)) < 2

    simple_ds = xr.Dataset(
        {"data": (("y", "x"), [[1, 2], [3, 4]])},
        attrs={"spatial:transform": [1.0, 0, 0, 0, -1.0, 0]},
    )
    simple_tree = DataTree(dataset=simple_ds)
    assert len(scan_resolution_levels(simple_tree)) < 2


def test_scan_geozarr_finds_all_levels():
    tree = GEOZARR_MULTISCALE.create()
    levels = scan_resolution_levels(tree)

    assert len(levels) == 3

    # Should be sorted finest to coarsest
    pixel_sizes = [level.pixel_size for level in levels]
    assert pixel_sizes == sorted(pixel_sizes)


def test_scan_native_at_root_includes_root():
    tree = NATIVE_AT_ROOT_MULTISCALE.create()
    levels = scan_resolution_levels(tree)

    # Should have 3 levels: root (native) + 2 overviews
    assert len(levels) == 3

    # Root should be finest (path=None)
    finest = levels[0]
    assert finest.path is None
    assert finest.pixel_size == 0.01  # native resolution

    # The rest should be sorted finest to coarsest
    pixel_sizes = [level.pixel_size for level in levels]
    assert pixel_sizes == sorted(pixel_sizes)


def test_scan_empty_tree_returns_empty():
    tree = DataTree()
    levels = scan_resolution_levels(tree)
    assert levels == []


def test_scan_non_multiscale_dataset():
    ds = xr.tutorial.load_dataset("air_temperature")
    tree = DataTree(dataset=ds)
    levels = scan_resolution_levels(tree)
    # No spatial:transform attribute, so no levels found
    assert levels == []


@pytest.fixture
def tms():
    return morecantile.tms.get("WebMercatorQuad")


def test_get_resolution_level_high_zoom_returns_finest_geozarr(tms):
    tree = GEOZARR_MULTISCALE.create()
    level = get_resolution_level(tree, zoom=15, tms=tms)
    assert level is not None
    assert level.path == "0"
    assert level.pixel_size == 0.01


def test_get_resolution_level_low_zoom_returns_coarsest_geozarr(tms):
    tree = GEOZARR_MULTISCALE.create()
    level = get_resolution_level(tree, zoom=0, tms=tms)
    assert level is not None
    assert level.path == "2"
    assert level.pixel_size == 0.04


def test_get_resolution_level_high_zoom_returns_root_native_at_root(tms):
    tree = NATIVE_AT_ROOT_MULTISCALE.create()
    level = get_resolution_level(tree, zoom=15, tms=tms)
    assert level is not None
    assert level.path is None
    assert level.pixel_size == 0.01


def test_get_resolution_level_low_zoom_returns_coarsest_native_at_root(tms):
    tree = NATIVE_AT_ROOT_MULTISCALE.create()
    level = get_resolution_level(tree, zoom=0, tms=tms)
    assert level is not None
    assert level.path == "1"
    assert level.pixel_size == 0.04


def test_get_resolution_level_returns_none_on_empty_tree(tms):
    tree = DataTree()
    level = get_resolution_level(tree, zoom=10, tms=tms)
    assert level is None


def _pyramid_tree(*pixel_sizes: float) -> DataTree:
    return DataTree.from_dict(
        {
            str(i): xr.Dataset(
                {"data": (("y", "x"), [[1, 2], [3, 4]])},
                attrs={
                    "spatial:transform": [pixel_size, 0, 0, 0, -pixel_size, 0],
                    "proj:code": "EPSG:4326",
                },
            )
            for i, pixel_size in enumerate(pixel_sizes)
        }
    )


@pytest.fixture
def wgs84_tms():
    return morecantile.tms.get("WGS1984Quad")


def test_get_resolution_level_picks_nearest_in_log_space(wgs84_tms):
    tile_pixel_size = wgs84_tms.matrix(3).cellSize

    # Tile pixel just inside the geometric mean of a 4x pyramid: finer wins
    tree = _pyramid_tree(tile_pixel_size / 1.9, tile_pixel_size * 4 / 1.9)
    level = get_resolution_level(tree, zoom=3, tms=wgs84_tms)
    assert level is not None
    assert level.pixel_size == tile_pixel_size / 1.9

    # Just past the geometric mean: coarser wins, even though it is
    # coarser than the tile (bounded upsampling instead of 16x the data)
    tree = _pyramid_tree(tile_pixel_size / 2.1, tile_pixel_size * 4 / 2.1)
    level = get_resolution_level(tree, zoom=3, tms=wgs84_tms)
    assert level is not None
    assert level.pixel_size == tile_pixel_size * 4 / 2.1


def test_get_resolution_level_tie_prefers_coarser(wgs84_tms):
    tile_pixel_size = wgs84_tms.matrix(3).cellSize

    # Exactly at the geometric mean (2x finer vs 2x coarser): pick coarser
    tree = _pyramid_tree(tile_pixel_size / 2, tile_pixel_size * 2)
    level = get_resolution_level(tree, zoom=3, tms=wgs84_tms)
    assert level is not None
    assert level.pixel_size == tile_pixel_size * 2


def test_get_resolution_level_4x_pyramid_zoom_transitions(wgs84_tms):
    # 4x pyramid with native resolution exactly matching zoom 10 tiles:
    # the 4x overview is exact at zoom 8. Zoom 9 sits at the midpoint and
    # must use the overview, not native (16x the data for a 2x-fine tile).
    native = wgs84_tms.matrix(10).cellSize
    overview = 4 * native
    tree = _pyramid_tree(native, overview)

    for zoom, expected in [(8, overview), (9, overview), (10, native), (11, native)]:
        level = get_resolution_level(tree, zoom=zoom, tms=wgs84_tms)
        assert level is not None
        assert level.pixel_size == expected, f"zoom {zoom}"


@pytest.mark.parametrize(
    "strategy,expected_index",
    [
        (OverviewSelectionStrategy.NEAREST, 0),
        (OverviewSelectionStrategy.COARSER, 1),
        (OverviewSelectionStrategy.FINER, 0),
    ],
)
def test_get_resolution_level_strategies_closer_to_finer(
    wgs84_tms, strategy, expected_index
):
    # Tile pixel size sits between the two levels, closer to the finer one
    tile_pixel_size = wgs84_tms.matrix(3).cellSize
    pixel_sizes = [tile_pixel_size / 1.9, tile_pixel_size * 4 / 1.9]
    tree = _pyramid_tree(*pixel_sizes)

    level = get_resolution_level(tree, zoom=3, tms=wgs84_tms, strategy=strategy)
    assert level is not None
    assert level.pixel_size == pixel_sizes[expected_index]


@pytest.mark.parametrize(
    "strategy,expected_index",
    [
        (OverviewSelectionStrategy.NEAREST, 1),
        (OverviewSelectionStrategy.COARSER, 1),
        (OverviewSelectionStrategy.FINER, 0),
    ],
)
def test_get_resolution_level_strategies_closer_to_coarser(
    wgs84_tms, strategy, expected_index
):
    # Tile pixel size sits between the two levels, closer to the coarser one
    tile_pixel_size = wgs84_tms.matrix(3).cellSize
    pixel_sizes = [tile_pixel_size / 2.1, tile_pixel_size * 4 / 2.1]
    tree = _pyramid_tree(*pixel_sizes)

    level = get_resolution_level(tree, zoom=3, tms=wgs84_tms, strategy=strategy)
    assert level is not None
    assert level.pixel_size == pixel_sizes[expected_index]


@pytest.mark.parametrize("strategy", list(OverviewSelectionStrategy))
def test_get_resolution_level_strategies_exact_match(wgs84_tms, strategy):
    # A level matching the tile exactly wins under every strategy
    tile_pixel_size = wgs84_tms.matrix(3).cellSize
    tree = _pyramid_tree(tile_pixel_size / 4, tile_pixel_size, tile_pixel_size * 4)

    level = get_resolution_level(tree, zoom=3, tms=wgs84_tms, strategy=strategy)
    assert level is not None
    assert level.pixel_size == tile_pixel_size


@pytest.mark.parametrize("strategy", list(OverviewSelectionStrategy))
def test_get_resolution_level_strategies_fall_back_when_out_of_range(wgs84_tms, strategy):
    tile_pixel_size = wgs84_tms.matrix(3).cellSize

    # All levels finer than the tile: nothing coarser to fall back to
    tree = _pyramid_tree(tile_pixel_size / 8, tile_pixel_size / 4)
    level = get_resolution_level(tree, zoom=3, tms=wgs84_tms, strategy=strategy)
    assert level is not None
    assert level.pixel_size == tile_pixel_size / 4

    # All levels coarser than the tile: nothing finer to fall back to
    tree = _pyramid_tree(tile_pixel_size * 4, tile_pixel_size * 8)
    level = get_resolution_level(tree, zoom=3, tms=wgs84_tms, strategy=strategy)
    assert level is not None
    assert level.pixel_size == tile_pixel_size * 4


def test_get_resolution_level_strategy_defaults_to_config(wgs84_tms):
    tile_pixel_size = wgs84_tms.matrix(3).cellSize
    pixel_sizes = [tile_pixel_size / 1.9, tile_pixel_size * 4 / 1.9]
    tree = _pyramid_tree(*pixel_sizes)

    level = get_resolution_level(tree, zoom=3, tms=wgs84_tms)
    assert level is not None
    assert level.pixel_size == pixel_sizes[0]

    with config.set(overview_selection_strategy="coarser"):
        level = get_resolution_level(tree, zoom=3, tms=wgs84_tms)
    assert level is not None
    assert level.pixel_size == pixel_sizes[1]


def test_get_dataset_geozarr_high_zoom_returns_finest():
    tree = GEOZARR_MULTISCALE.create()
    tms = morecantile.tms.get("WebMercatorQuad")

    ds = get_dataset(tree, zoom=15, tms=tms)

    assert isinstance(ds, xr.Dataset)
    assert "data" in ds.data_vars
    assert ds.sizes["X"] == 64
    assert ds.sizes["Y"] == 64


def test_get_dataset_geozarr_low_zoom_returns_coarsest():
    tree = GEOZARR_MULTISCALE.create()
    tms = morecantile.tms.get("WebMercatorQuad")

    ds = get_dataset(tree, zoom=0, tms=tms)

    assert isinstance(ds, xr.Dataset)
    assert "data" in ds.data_vars
    assert ds.sizes["X"] == 16
    assert ds.sizes["Y"] == 16


def test_get_dataset_native_at_root_high_zoom_returns_finest():
    tree = NATIVE_AT_ROOT_MULTISCALE.create()
    tms = morecantile.tms.get("WebMercatorQuad")

    ds = get_dataset(tree, zoom=15, tms=tms)

    assert isinstance(ds, xr.Dataset)
    assert "data" in ds.data_vars
    assert ds.sizes["X"] == 64
    assert ds.sizes["Y"] == 64


def test_get_dataset_native_at_root_low_zoom_returns_coarsest():
    tree = NATIVE_AT_ROOT_MULTISCALE.create()
    tms = morecantile.tms.get("WebMercatorQuad")

    ds = get_dataset(tree, zoom=0, tms=tms)

    assert isinstance(ds, xr.Dataset)
    assert "data" in ds.data_vars
    assert ds.sizes["X_1"] == 16
    assert ds.sizes["Y_1"] == 16


def test_get_dataset_regular_returns_root():
    ds = xr.tutorial.load_dataset("air_temperature")
    tree = DataTree(dataset=ds)

    result = get_dataset(tree)

    assert isinstance(result, xr.Dataset)
    assert "air" in result.data_vars


def test_get_dataset_empty_raises():
    tree = DataTree()

    with pytest.raises(ValueError, match="no extractable dataset"):
        get_dataset(tree)


def test_get_crs_invalid_proj_code_logs_error(caplog):
    """Test that invalid proj:code logs error and falls through."""
    ds = xr.Dataset(attrs={"proj:code": "INVALID_CRS_CODE"})
    result = get_crs(ds)
    assert result is None
    assert "Failed to parse proj:code" in caplog.text


def test_get_crs_invalid_proj_wkt2_logs_error(caplog):
    """Test that invalid proj:wkt2 logs error and returns None."""
    ds = xr.Dataset(attrs={"proj:wkt2": "NOT_VALID_WKT"})
    result = get_crs(ds)
    assert result is None
    assert "Failed to parse proj:wkt2" in caplog.text


def test_get_crs_falls_back_to_wkt2():
    """Test that get_crs falls back to proj:wkt2 when proj:code not present."""
    wkt = CRS.from_epsg(4326).to_wkt()
    ds = xr.Dataset(attrs={"proj:wkt2": wkt})
    result = get_crs(ds)
    assert result is not None
    assert result.to_epsg() == 4326
