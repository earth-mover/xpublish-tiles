#!/usr/bin/env python3


import cf_xarray  # noqa: F401 - Enable cf accessor
import morecantile
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from PIL import Image
from pyproj import CRS
from pyproj.aoi import BBox

import xarray as xr
from src.xpublish_tiles.render.raster import nearest_on_uniform_grid_quadmesh
from tests import create_query_params
from xarray.testing import assert_equal
from xpublish_tiles.pipeline import (
    apply_query,
    bbox_overlap,
    pipeline,
)
from xpublish_tiles.testing.datasets import (
    CURVILINEAR,
    FORECAST,
    PARA,
    create_global_dataset,
)
from xpublish_tiles.testing.lib import (
    assert_render_matches_snapshot,
    compare_image_buffers,
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
    assert_render_matches_snapshot(result, png_snapshot)


@pytest.mark.asyncio
async def test_high_zoom_tile_global_dataset(png_snapshot, pytestconfig):
    ds = create_global_dataset()
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
        selectors={"L": 0, "forecast_reference_time": "1960-02-01 00:00:00"},
    )
    assert_equal(
        result["sst"].da, FORECAST.sst.sel(L=0, S="1960-02-01 00:00:00").isel(M=-1, S=-1)
    )

    curvilinear_ds = CURVILINEAR.create()
    result = apply_query(curvilinear_ds, variables=["foo"], selectors={})
    assert_equal(result["foo"].da, curvilinear_ds.foo.sel(s_rho=0, method="nearest"))


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
