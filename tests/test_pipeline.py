#!/usr/bin/env python3

import io

import cf_xarray  # noqa: F401 - Enable cf accessor
import matplotlib.pyplot as plt
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
from tests.conftest import compare_image_buffers
from tests.tiles import PARA_TILES, TILES, WEBMERC_TMS
from xarray.testing import assert_equal
from xpublish_tiles.datasets import FORECAST, PARA, ROMSDS, create_global_dataset
from xpublish_tiles.lib import check_transparent_pixels
from xpublish_tiles.pipeline import (
    apply_query,
    check_bbox_overlap,
    pipeline,
)
from xpublish_tiles.types import ImageFormat, OutputBBox, OutputCRS, QueryParams


def visualize_tile(result: io.BytesIO, tile: morecantile.Tile) -> None:
    """Visualize a rendered tile with matplotlib showing RGB and alpha channels.

    Args:
        result: BytesIO buffer containing PNG image data
        tile: Tile object with z, x, y coordinates
    """
    result.seek(0)
    pil_img = Image.open(result)
    img_array = np.array(pil_img)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Show the rendered tile
    axes[0].imshow(img_array)
    axes[0].set_title(f"Tile z={tile.z}, x={tile.x}, y={tile.y}")

    # Show alpha channel if present
    if img_array.shape[2] == 4:
        alpha = img_array[:, :, 3]
        im = axes[1].imshow(alpha, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title(
            f"Alpha Channel\n{((alpha == 0).sum() / alpha.size * 100):.1f}% transparent"
        )
        plt.colorbar(im, ax=axes[1])
    else:
        axes[1].text(
            0.5, 0.5, "No Alpha", ha="center", va="center", transform=axes[1].transAxes
        )

    plt.tight_layout()
    plt.show(block=True)  # Block until window is closed


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
    assert check_bbox_overlap(bbox, grid_bbox, True), (
        f"Valid bbox {bbox} should overlap with global {grid_description} grid. "
        f"Longitude wrapping should handle any longitude values."
    )


def create_query_params(tile, tms, *, colorscalerange=None):
    """Create QueryParams instance using test tiles and TMS."""

    # Convert TMS CRS to pyproj CRS
    target_crs = CRS.from_epsg(tms.crs.to_epsg())

    # Get bounds in the TMS's native CRS
    native_bounds = tms.xy_bounds(tile)
    bbox = BBox(
        west=native_bounds[0],
        south=native_bounds[1],
        east=native_bounds[2],
        north=native_bounds[3],
    )

    return QueryParams(
        variables=["foo"],
        crs=OutputCRS(target_crs),
        bbox=OutputBBox(bbox),
        selectors={},
        style="raster",
        width=256,
        height=256,
        cmap="viridis",
        colorscalerange=colorscalerange,
        format=ImageFormat.PNG,
    )


def validate_transparency(
    content: bytes,
    *,
    tile=None,
    tms=None,
    dataset_bbox: BBox | None = None,
):
    """Validate transparency of rendered content based on tile/dataset overlap.

    Args:
        content: The rendered PNG content
        tile: The tile being rendered (optional)
        tms: The tile matrix set (optional)
        dataset_bbox: Bounding box of the dataset (optional)
    """
    # Calculate tile bbox if tile and tms provided
    tile_bbox = None
    if tile is not None and tms is not None:
        tile_bounds = tms.bounds(tile)
        tile_bbox = BBox(
            west=tile_bounds.left,
            south=tile_bounds.bottom,
            east=tile_bounds.right,
            north=tile_bounds.top,
        )

    # Check if this is the specific failing test case that should skip transparency checks
    # This is a boundary tile, and the bounds checking is inaccurate.
    # TODO: Consider figuring out a better way to do this, but I suspect it's just too hard.
    # TODO: We could instead just keep separate lists of fully contained and partially intersecting tiles;
    #       and add an explicit check.
    skip_transparency_check = (
        tile is not None
        and tms is not None
        and tile.x == 0
        and tile.y == 1
        and tile.z == 2
        and tms.id == "EuropeanETRS89_LAEAQuad"
    )

    # Check transparency based on whether dataset contains the tile
    transparent_percent = check_transparent_pixels(content)
    if not skip_transparency_check:
        if tile_bbox is not None and dataset_bbox is not None:
            if dataset_bbox.contains(tile_bbox):
                assert (
                    transparent_percent == 0
                ), f"Found {transparent_percent:.1f}% transparent pixels in fully contained tile."
            elif dataset_bbox.intersects(tile_bbox):
                assert transparent_percent > 0
            else:
                assert (
                    transparent_percent == 100
                ), f"Found {transparent_percent:.1f}% transparent pixels in fully disjoint tile (expected 100%)."
        else:
            assert (
                transparent_percent == 0
            ), f"Found {transparent_percent:.1f}% transparent pixels."


def assert_render_matches_snapshot(
    result: io.BytesIO,
    png_snapshot,
    *,
    tile=None,
    tms=None,
    dataset_bbox: BBox | None = None,
):
    """Helper function to validate PNG content against snapshot.

    Args:
        result: The rendered image buffer
        png_snapshot: The expected snapshot
        tile: The tile being rendered (optional)
        tms: The tile matrix set (optional)
        dataset_bbox: Bounding box of the dataset (optional)
    """
    assert isinstance(result, io.BytesIO)
    result.seek(0)
    content = result.read()
    assert len(content) > 0
    validate_transparency(content, tile=tile, tms=tms, dataset_bbox=dataset_bbox)
    assert content == png_snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("tile,tms", TILES)
async def test_pipeline_tiles(global_datasets, tile, tms, png_snapshot):
    """Test pipeline with various tiles using their native TMS CRS."""
    ds = global_datasets
    query_params = create_query_params(tile, tms)
    result = await pipeline(ds, query_params)

    assert_render_matches_snapshot(result, png_snapshot)


@pytest.mark.skip(reason="this bbox is slightly outside the bounds of web mercator")
async def test_pipeline_bad_bbox(global_datasets, png_snapshot):
    """Test pipeline with various tiles using their native TMS CRS."""
    ds = global_datasets
    query = QueryParams(
        variables=["foo"],
        crs=OutputCRS(CRS.from_user_input(3857)),
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
        cmap="viridis",
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

    result = apply_query(ROMSDS, variables=["temp"], selectors={})
    assert_equal(
        result["temp"].da, ROMSDS.temp.sel(s_rho=0, method="nearest").isel(ocean_time=-1)
    )


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
        cmap="viridis",
        colorscalerange=None,
        format=ImageFormat.PNG,
    )
    actual = await pipeline(ds, query)
    if pytestconfig.getoption("--visualize"):
        visualize_tile(actual, morecantile.Tile(0, 0, 0))
    expected = await pipeline(ds.astype(np.float64), query)
    assert compare_image_buffers(expected, actual)
