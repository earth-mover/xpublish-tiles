#!/usr/bin/env python3


import io

import cf_xarray  # noqa: F401 - Enable cf accessor
import morecantile
import numpy as np
import pandas as pd
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from morecantile import Tile
from PIL import Image
from pyproj import CRS
from pyproj.aoi import BBox

import xarray as xr
from src.xpublish_tiles.render.raster import nearest_on_uniform_grid_quadmesh
from tests import create_query_params
from xarray.testing import assert_equal
from xpublish_tiles import config
from xpublish_tiles.lib import (
    IndexingError,
    MissingParameterError,
    VariableNotFoundError,
    check_transparent_pixels,
)
from xpublish_tiles.pipeline import (
    apply_query,
    bbox_overlap,
    pipeline,
)
from xpublish_tiles.testing.datasets import (
    CURVILINEAR,
    FORECAST,
    GLOBAL_6KM,
    GLOBAL_6KM_360,
    GLOBAL_NANS,
    HRRR,
    PARA,
)
from xpublish_tiles.testing.lib import (
    assert_render_matches_snapshot,
    compare_image_buffers,
    compare_image_buffers_with_debug,
    visualize_tile,
)
from xpublish_tiles.testing.tiles import (
    PARA_TILES,
    TILES,
    WEBMERC_TMS,
)
from xpublish_tiles.types import ImageFormat, OutputBBox, OutputCRS, QueryParams


@st.composite
def bboxes(draw):
    """Generate valid bounding boxes for testing."""
    # Generate latitude bounds (must be within -90 to 90)
    south = draw(st.floats(min_value=-89.9, max_value=89.9))
    north = draw(st.floats(min_value=south + 0.1, max_value=90.0))

    # Generate longitude bounds (can be any range, including wrapped)
    west = draw(st.floats(min_value=-720.0, max_value=720.0))
    east = draw(st.floats(min_value=west + 0.1, max_value=west + 360.0))

    return BBox(west=west, south=south, east=east, north=north)


@given(
    bbox=bboxes(),
    grid_config=st.sampled_from(
        [
            (BBox(west=0.0, south=-90.0, east=360.0, north=90.0), "0-360"),
            (BBox(west=-180.0, south=-90.0, east=180.0, north=90.0), "-180-180"),
        ]
    ),
)
@example(
    bbox=BBox(west=-200.0, south=20.0, east=-190.0, north=40.0),
    grid_config=(BBox(west=0.0, south=-90.0, east=360.0, north=90.0), "0-360"),
)
@example(
    bbox=BBox(west=400.0, south=20.0, east=420.0, north=40.0),
    grid_config=(BBox(west=-180.0, south=-90.0, east=180.0, north=90.0), "-180-180"),
)
@example(
    bbox=BBox(west=-1.0, south=0.0, east=0.0, north=1.0),
    grid_config=(BBox(west=0.0, south=-90.0, east=360.0, north=90.0), "0-360"),
)
def test_bbox_overlap_detection(bbox, grid_config):
    """Test the bbox overlap detection logic handles longitude wrapping correctly."""
    grid_bbox, grid_description = grid_config
    # All valid bboxes should overlap with global grids due to longitude wrapping
    assert bbox_overlap(bbox, grid_bbox, True), (
        f"Valid bbox {bbox} should overlap with global {grid_description} grid. "
        f"Longitude wrapping should handle any longitude values."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tile,tms", TILES)
async def test_pipeline_tiles(global_datasets, tile, tms, png_snapshot, pytestconfig):
    """Test pipeline with various tiles using their native TMS CRS."""
    ds = global_datasets
    query_params = create_query_params(tile, tms)
    with config.set(rectilinear_check_min_size=0):
        result = await pipeline(ds, query_params)
    if pytestconfig.getoption("--visualize"):
        visualize_tile(result, tile)
    assert_render_matches_snapshot(
        result,
        png_snapshot,
        # we have small rasterization differences in CI
        perceptual_threshold=0.99
        if ds.attrs["name"] == "reduced_gaussian_n320"
        else None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("ds", [GLOBAL_6KM.create(), GLOBAL_6KM_360.create()])
@pytest.mark.parametrize(
    "tile",
    [
        Tile(x=0, y=1, z=2),
        Tile(x=1, y=1, z=2),
        Tile(x=0, y=2, z=2),
        Tile(x=0, y=3, z=3),
        Tile(x=0, y=6, z=4),
        Tile(x=0, y=788, z=11),
    ],
)
async def test_global_6km_regression(ds, tile, png_snapshot, pytestconfig):
    query_params = create_query_params(tile, WEBMERC_TMS)
    query_params.width = 512
    query_params.height = 512
    result = await pipeline(ds, query_params)
    if pytestconfig.getoption("--visualize"):
        visualize_tile(result, tile)
    assert_render_matches_snapshot(result, png_snapshot)


async def test_pipeline_bad_bbox(global_datasets, png_snapshot, pytestconfig):
    """Test pipeline with various tiles using their native TMS CRS."""
    ds = global_datasets
    query = QueryParams(
        variables=["foo"],
        crs=OutputCRS(CRS.from_user_input(3857)),
        # This bbox will transform to west=179.999CXXX, east=-157.XXX
        bbox=OutputBBox(
            BBox(
                west=-20037508.3428,
                south=7514065.628550399,
                east=-17532819.799950078,
                north=10018754.17140032,
            )
        ),
        selectors={},
        style="raster",
        width=256,
        height=256,
        variant="viridis",
        colorscalerange=None,
        format=ImageFormat.PNG,
    )
    result = await pipeline(ds, query)
    assert_render_matches_snapshot(
        result,
        png_snapshot,
        # we have small rasterization differences in CI
        perceptual_threshold=0.99
        if ds.attrs["name"] == "reduced_gaussian_n320"
        else None,
    )


@pytest.mark.asyncio
async def test_high_zoom_tile_global_dataset(global_datasets, png_snapshot, pytestconfig):
    ds = global_datasets
    tms = WEBMERC_TMS
    tile = morecantile.Tile(x=524288 + 2916, y=262144, z=20)
    query_params = create_query_params(tile, tms, colorscalerange=(-1, 1))
    result = await pipeline(ds, query_params)
    if pytestconfig.getoption("--visualize"):
        visualize_tile(result, tile)
    assert_render_matches_snapshot(result, png_snapshot)


async def test_projected_coordinate_data(
    projected_dataset_and_tile, png_snapshot, pytestconfig
):
    ds, tile, tms = projected_dataset_and_tile
    query_params = create_query_params(tile, tms)
    with config.set(rectilinear_check_min_size=0):
        result = await pipeline(ds, query_params)
    if pytestconfig.getoption("--visualize"):
        visualize_tile(result, tile)
    assert_render_matches_snapshot(
        result, png_snapshot, tile=tile, tms=tms, dataset_bbox=ds.attrs["bbox"]
    )


@pytest.mark.asyncio
async def test_curvilinear_data(curvilinear_dataset_and_tile, png_snapshot, pytestconfig):
    ds, tile, tms = curvilinear_dataset_and_tile
    query_params = create_query_params(tile, tms)
    result = await pipeline(ds, query_params)
    if pytestconfig.getoption("--visualize"):
        visualize_tile(result, tile)
    assert_render_matches_snapshot(result, png_snapshot, tile=tile, tms=tms)

    transposed = ds.assign_coords(lat=ds.lat.transpose(), lon=ds.lon.transpose())
    with config.set(transform_chunk_size=ds.foo.shape[-1] + 10):
        transposed_result = await pipeline(transposed, query_params)
    if pytestconfig.getoption("--visualize"):
        visualize_tile(result, tile)

    # Can't use a snapshot twice in a test sadly
    result.seek(0)
    transposed_result.seek(0)
    _, _ = compare_image_buffers_with_debug(
        result,  # Expected (original)
        transposed_result,  # Actual (transposed)
        test_name=f"test_curvilinear_data_transposed_vs_original[{tile.z}/{tile.x}/{tile.y}]",
        tile_info=(tile, tms),
        debug_visual=pytestconfig.getoption("--debug-visual"),
        debug_visual_save=pytestconfig.getoption("--debug-visual-save"),
    )


@pytest.mark.parametrize("tile,tms", PARA_TILES)
async def test_categorical_data(tile, tms, png_snapshot, pytestconfig):
    ds = PARA.create().squeeze("time")
    query_params = create_query_params(tile, tms)
    result = await pipeline(ds, query_params)
    if pytestconfig.getoption("--visualize"):
        visualize_tile(result, tile)
    assert_render_matches_snapshot(
        result, png_snapshot, tile=tile, tms=tms, dataset_bbox=ds.attrs["bbox"]
    )


async def test_categorical_data_with_custom_colormap(png_snapshot, pytestconfig):
    """Test categorical data with custom colormap renders correctly."""
    ds = PARA.create().squeeze("time")

    # PARA has 10 categories (flag_values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # Create a custom colormap with 10 distinct colors
    custom_colormap = {
        "0": "#ff0000",  # red
        "1": "#00ff00",  # green
        "2": "#0000ff",  # blue
        "3": "#ffff00",  # yellow
        "4": "#ff00ff",  # magenta
        "5": "#00ffff",  # cyan
        "6": "#800000",  # maroon
        "7": "#008000",  # dark green
        "8": "#000080",  # navy
        "9": "#808080",  # gray
    }

    tile, tms = PARA_TILES[0].values
    query_params = create_query_params(
        tile, tms, style="raster", variant="custom", colormap=custom_colormap
    )
    result = await pipeline(ds, query_params)

    if pytestconfig.getoption("--visualize"):
        visualize_tile(result, tile)
    assert_render_matches_snapshot(
        result, png_snapshot, tile=tile, tms=tms, dataset_bbox=ds.attrs["bbox"]
    )


@pytest.mark.asyncio
async def test_categorical_out_of_range_values_are_transparent(pytestconfig):
    """Test that categorical values outside flag_values render as transparent with custom colormap."""
    ds = PARA.create().squeeze("time")

    # PARA has 10 categories (flag_values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # Create a custom colormap with only the valid flag_values
    custom_colormap = {
        "0": "#ff0000",
        "1": "#00ff00",
        "2": "#0000ff",
        "3": "#ffff00",
        "4": "#ff00ff",
        "5": "#00ffff",
        "6": "#800000",
        "7": "#008000",
        "8": "#000080",
        "9": "#808080",
    }

    # Use the Belém tile (capital area) which should have good data coverage
    tile, tms = Tile(x=22, y=31, z=6), WEBMERC_TMS
    query_params = create_query_params(
        tile, tms, style="raster", variant="custom", colormap=custom_colormap
    )

    # Render unmodified data and get baseline transparency
    unmodified_result = await pipeline(ds, query_params)
    baseline_transparent_percent = check_transparent_pixels(unmodified_result.getvalue())

    # Replace all values > 2 with 99 (out of range)
    ds_modified = ds.compute()
    ds_modified["foo"].values[ds_modified["foo"].values > 2] = 99

    # Render modified data
    modified_result = await pipeline(ds_modified, query_params)
    modified_transparent_percent = check_transparent_pixels(modified_result.getvalue())

    # Assert that modified data has more transparent pixels than unmodified
    assert modified_transparent_percent > baseline_transparent_percent, (
        f"Expected transparency to increase with out-of-range values. "
        f"Baseline: {baseline_transparent_percent:.1f}%, Modified: {modified_transparent_percent:.1f}%"
    )

    if pytestconfig.getoption("--visualize"):
        visualize_tile(modified_result, tile)


@pytest.mark.asyncio
@pytest.mark.parametrize("tile,tms", GLOBAL_NANS.tiles)
async def test_global_nans_data(tile, tms, png_snapshot, pytestconfig):
    """Test pipeline with global dataset containing diagonal NaN patterns."""
    ds = GLOBAL_NANS.create()
    query_params = create_query_params(tile, tms)
    with config.set(rectilinear_check_min_size=0):
        result = await pipeline(ds, query_params)
    if pytestconfig.getoption("--visualize"):
        visualize_tile(result, tile)

    # Skip transparency validation for NaN data - just check the snapshot directly
    assert isinstance(result, io.BytesIO)
    result.seek(0)
    content = result.read()
    assert len(content) > 0
    assert content == png_snapshot


def test_apply_query_errors():
    ds = FORECAST.copy(deep=True)
    ds["foo2"] = ds["sst"] * 2

    with pytest.raises(VariableNotFoundError):
        apply_query(ds, variables=["foooooooo"], selectors={})
    with pytest.raises(IndexingError):
        apply_query(ds, variables=["sst"], selectors={"L": 123123123})

    hrrr = HRRR.create()

    with pytest.raises(IndexingError, match="Invalid selection method 'invalid'"):
        apply_query(
            hrrr, variables=["foo"], selectors={"time": "invalid::2018-01-01T01:00"}
        )

    with pytest.raises(IndexingError, match="Invalid selection method 'badmethod'"):
        apply_query(
            hrrr, variables=["foo"], selectors={"time": "badmethod::2018-01-01T01:00"}
        )


def test_apply_query_selectors():
    ds = FORECAST.copy(deep=True)
    ds["foo2"] = ds["sst"] * 2

    result = apply_query(ds, variables=["sst"], selectors={})
    assert result["sst"].da.dims == ("Y", "X")
    assert len(result) == 1

    result = apply_query(ds, variables=["sst", "foo2"], selectors={})
    assert len(result) == 2
    assert result["sst"].grid.equals(result["foo2"].grid)

    result = apply_query(
        ds,
        variables=["sst"],
        selectors={"L": 0, "S": "1960-02-01 00:00:00"},
    )
    assert_equal(
        result["sst"].da,
        FORECAST.sst.sel(L=0, S="1960-02-01 00:00:00").isel(M=-1, S=-1),
    )

    curvilinear_ds = CURVILINEAR.create()
    result = apply_query(curvilinear_ds, variables=["foo"], selectors={})
    assert_equal(result["foo"].da, curvilinear_ds.foo.sel(s_rho=0, method="nearest"))

    hrrr = HRRR.create()
    hrrr.time.attrs = {"standard_name": "time"}
    result = apply_query(hrrr, variables=["foo"], selectors={"time": "2018-01-01"})
    expected = hrrr.foo.isel(time=0, step=-1)
    assert_equal(result["foo"].da, expected)

    # second time coordinate with same CF-compliant attrs
    hrrr.coords["time2"] = hrrr.time.copy()
    result = apply_query(hrrr, variables=["foo"], selectors={"time": "2018-01-01"})
    assert_equal(result["foo"].da, expected.assign_coords(time2=hrrr.time.data.item()))

    # out of order Z coordinates
    ds = ds.isel(M=[1, 0, 2])
    ds["M"].attrs["axis"] = "Z"
    ds.attrs["_xpublish_id"] = "foo"
    with pytest.raises(MissingParameterError):
        apply_query(ds, variables=["sst"], selectors={})


def test_apply_query_selectors_method_nearest():
    hrrr = HRRR.create().reindex(
        time=pd.date_range("2018-01-01", "2018-01-02", freq="6h")
    )
    actual = apply_query(
        hrrr,
        variables=["foo"],
        selectors={
            "time": "nearest::2018-01-01T01:15",
        },
    )
    expected = hrrr.sel(time="2018-01-01T01:15", method="nearest").isel(step=-1)
    xr.testing.assert_equal(actual["foo"].da, expected["foo"])

    # Test with multiple selectors (using existing step dimension)
    actual_multi = apply_query(
        hrrr,
        variables=["foo"],
        selectors={
            "time": "nearest::2018-01-01T01:15",
            "step": "nearest::45min",
        },
    )
    expected_multi = hrrr.sel(time="2018-01-01T01:15", method="nearest").sel(
        step="45min", method="nearest"
    )
    xr.testing.assert_equal(actual_multi["foo"].da, expected_multi["foo"])

    # Test ffill
    target_time = "2018-01-01T04:00"  # Between first two timestamps
    actual_ffill = apply_query(
        hrrr, variables=["foo"], selectors={"time": f"ffill::{target_time}"}
    )
    expected_ffill = hrrr.sel(time=target_time, method="ffill").isel(step=-1)
    xr.testing.assert_equal(actual_ffill["foo"].da, expected_ffill["foo"])

    # Test bfill
    actual_bfill = apply_query(
        hrrr, variables=["foo"], selectors={"time": f"bfill::{target_time}"}
    )
    expected_bfill = hrrr.sel(time=target_time, method="bfill").isel(step=-1)
    xr.testing.assert_equal(actual_bfill["foo"].da, expected_bfill["foo"])

    # Test backfill (alias for bfill)
    actual_backfill = apply_query(
        hrrr, variables=["foo"], selectors={"time": f"backfill::{target_time}"}
    )
    expected_backfill = hrrr.sel(time=target_time, method="backfill").isel(step=-1)
    xr.testing.assert_equal(actual_backfill["foo"].da, expected_backfill["foo"])

    target_time = "2018-01-01T04:00"
    actual = apply_query(
        hrrr, variables=["foo"], selectors={"time": f"pad::{target_time}"}
    )
    expected = hrrr.sel(time=target_time, method="pad").isel(step=-1)
    xr.testing.assert_equal(actual["foo"].da, expected["foo"])


def test_apply_query_selectors_exact():
    """Test exact selection method (explicit and implicit)"""
    hrrr = HRRR.create().reindex(
        time=pd.date_range("2018-01-01", "2018-01-02", freq="6h")
    )

    # Implicit exact (no method prefix)
    actual_implicit = apply_query(
        hrrr, variables=["foo"], selectors={"time": "2018-01-01T00:00"}
    )
    expected = hrrr.sel(time="2018-01-01T00:00").isel(step=-1)
    xr.testing.assert_equal(actual_implicit["foo"].da, expected["foo"])

    # Explicit exact
    actual_explicit = apply_query(
        hrrr, variables=["foo"], selectors={"time": "exact::2018-01-01T00:00"}
    )
    xr.testing.assert_equal(actual_explicit["foo"].da, expected["foo"])


def test_apply_query_with_string_selectors():
    ds = xr.Dataset(
        data_vars={
            "foo": (
                ("band", "band2", "offset", "x", "y"),
                np.arange(3240).reshape(3, 3, 3, 30, 4),
            )
        },
        coords={
            "x": (["x"], np.arange(30), {"axis": "X", "standard_name": "longitude"}),
            "y": (["y"], np.arange(4), {"axis": "Y", "standard_name": "latitude"}),
            "offset": (
                ["offset"],
                np.array(
                    [
                        np.timedelta64(i, "ns")
                        for i in [0, 3_600_000_000_000, 3_600_000_000_000 * 2]
                    ]
                ),
                {"standard_name": "offset"},
            ),
            "band": (["band"], [1, 2, 3], {"standard_name": "wavelength"}),
            "band2": (["band2"], ["1", "2", "3"], {"standard_name": "wavelength"}),
        },
    )

    selectors = {"band": "1", "band2": 2, "offset": "1 hours"}
    result = apply_query(ds, variables=["foo"], selectors=selectors)
    assert_equal(result["foo"].da, ds.foo.sel(band=1, band2="2", offset="1 hours"))


def test_datashader_nearest_regridding():
    ds = xr.Dataset(
        {"foo": (("x", "y"), np.arange(120).reshape(30, 4))},
        coords={"x": np.arange(30), "y": np.arange(4)},
    ).drop_indexes(("x", "y"))
    res = nearest_on_uniform_grid_quadmesh(ds.foo, "x", "y")
    assert_equal(ds.foo, res.astype(ds.foo.dtype).transpose(*ds.foo.dims))


@pytest.mark.parametrize("data_type", ["discrete", "continuous"])
@pytest.mark.parametrize("size", [1, 2, 4, 8])
@pytest.mark.parametrize("kind", ["u", "i", "f"])
async def test_datashader_casting(data_type, size, kind, pytestconfig):
    """
    For all dtypes, we render a bbox that will contain NaNs.
    Ensure that output is identical to that of rendering a float64 input.
    """
    if kind == "f" and size == 1:
        pytest.skip()
    if data_type == "discrete":
        attrs = {
            "flag_values": [0, 1, 2, 3],
            "flag_meanings": "a b c d",
        }
    else:
        attrs = {"valid_min": 0, "valid_max": 3}
    ds = xr.Dataset(
        {
            "foo": (
                ("x", "y"),
                np.array([[1, 2, 3], [0, 1, 2]], dtype=f"{kind}{size}"),
                attrs,
            )
        },
        coords={
            "x": ("x", [1, 2], {"standard_name": "longitude"}),
            "y": ("y", [1, 2, 3], {"standard_name": "latitude"}),
        },
    )
    query = QueryParams(
        variables=["foo"],
        crs=OutputCRS(CRS.from_user_input(4326)),
        bbox=OutputBBox(BBox(west=-5, east=5, south=-5, north=5)),
        selectors={},
        style="raster",
        width=256,
        height=256,
        variant="viridis",
        colorscalerange=None,
        format=ImageFormat.PNG,
    )
    actual = await pipeline(ds, query)
    if pytestconfig.getoption("--visualize"):
        visualize_tile(actual, morecantile.Tile(0, 0, 0))
    expected = await pipeline(ds.astype(np.float64), query)
    assert compare_image_buffers(expected, actual)


async def test_bad_latitude_coordinates(png_snapshot, pytestconfig):
    """
    Regression test for https://github.com/holoviz/datashader/issues/1431
    IMPORTANT: This only fails on linux with datshader < 0.18.2
    """
    lon = -179.875 + 0.25 * np.arange(1440)
    lat = 89.875 - 0.25 * np.arange(720)
    ds = xr.DataArray(
        np.ones(shape=(720, 1440), dtype="f4"),
        dims=("lat", "lon"),
        coords={
            "lon": ("lon", lon, {"standard_name": "longitude"}),
            "lat": ("lat", lat, {"standard_name": "latitude"}),
        },
        attrs={"valid_min": 0, "valid_max": 2},
        name="foo",
    ).to_dataset()
    tile = morecantile.Tile(x=8, y=8, z=4)
    query = create_query_params(tms=WEBMERC_TMS, tile=tile)
    render = await pipeline(ds, query)
    if pytestconfig.getoption("--visualize"):
        visualize_tile(render, tile)
    assert_render_matches_snapshot(render, png_snapshot)


async def test_transparent_tile_no_coverage(pytestconfig):
    """Test that a fully transparent PNG is returned when there's no bbox coverage."""
    # Create dataset bounded between lon=0 to 90, lat=0 to 90
    lon = np.linspace(0, 90, 91)
    lat = np.linspace(0, 90, 91)
    ds = xr.DataArray(
        np.ones(shape=(91, 91), dtype="f4"),
        dims=("lat", "lon"),
        coords={
            "lon": ("lon", lon, {"standard_name": "longitude"}),
            "lat": ("lat", lat, {"standard_name": "latitude"}),
        },
        attrs={"valid_min": 0, "valid_max": 1},
        name="foo",
    ).to_dataset()

    # Request a tile in bbox (-180 to -10, -80 to -70) - no overlap with dataset
    query = QueryParams(
        variables=["foo"],
        crs=OutputCRS(CRS.from_user_input(4326)),
        bbox=OutputBBox(BBox(west=-180, south=-80, east=-10, north=-70)),
        selectors={},
        style="raster",
        width=256,
        height=256,
        variant="viridis",
        colorscalerange=None,
        format=ImageFormat.PNG,
    )

    result = await pipeline(ds, query)

    # Verify the tile is fully transparent
    result.seek(0)
    img = Image.open(result)
    img = img.convert("RGBA")

    # Check that all pixels are fully transparent (alpha = 0)
    alpha_channel = np.array(img)[:, :, 3]
    assert np.all(alpha_channel == 0), "All pixels should be fully transparent"

    # Also check image dimensions
    assert img.size == (256, 256), "Image should have correct dimensions"


@pytest.mark.asyncio
@pytest.mark.parametrize("tile,tms", HRRR.tiles)
async def test_hrrr_multiple_vs_hrrr_rendering(tile, tms, pytestconfig):
    """Test that HRRR_MULTIPLE renders identically to HRRR for the same tiles."""
    from xpublish_tiles.testing.datasets import HRRR_MULTIPLE

    # Create both datasets
    hrrr_ds = HRRR.create()
    hrrr_multiple_ds = HRRR_MULTIPLE.create()

    # Create query params for the tile
    query_params = create_query_params(tile, tms)

    # Render both datasets with the same parameters
    hrrr_result = await pipeline(hrrr_ds, query_params)
    hrrr_multiple_result = await pipeline(hrrr_multiple_ds, query_params)

    if pytestconfig.getoption("--visualize"):
        visualize_tile(hrrr_result, tile)
        visualize_tile(hrrr_multiple_result, tile)

    # Compare the rendered images
    images_similar, _ = compare_image_buffers_with_debug(
        hrrr_result,
        hrrr_multiple_result,
        test_name="hrrr_multiple",
        tile_info=(tile, tms),
        debug_visual=pytestconfig.getoption("--debug-visual", default=False),
        debug_visual_save=pytestconfig.getoption("--debug-visual-save", default=False),
        mode="perceptual",
        perceptual_threshold=0.99,
    )
    assert images_similar, (
        f"HRRR_MULTIPLE should render identically to HRRR for tile {tile} "
        f"but images differ"
    )
