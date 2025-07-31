#!/usr/bin/env python3

import io

import cf_xarray  # noqa: F401 - Enable cf accessor
import morecantile
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from PIL import Image
from pyproj import CRS
from pyproj.aoi import BBox

from tests.tiles import TILES, WEBMERC_TMS
from xpublish_tiles.datasets import create_global_dataset
from xpublish_tiles.pipeline import (
    check_bbox_overlap,
    pipeline,
)
from xpublish_tiles.types import ImageFormat, OutputBBox, OutputCRS, QueryParams, Style


def is_png(buffer: io.BytesIO) -> bool:
    """Check if a BytesIO buffer contains valid PNG data."""
    buffer.seek(0)
    header = buffer.read(8)
    buffer.seek(0)
    # PNG signature: 89 50 4E 47 0D 0A 1A 0A
    return header == b"\x89PNG\r\n\x1a\n"


def check_transparent_pixels(image_bytes):
    """Check the percentage of transparent pixels in a PNG image."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    arr = np.array(img)
    transparent_mask = arr[:, :, 3] == 0
    transparent_count = np.sum(transparent_mask)
    total_pixels = arr.shape[0] * arr.shape[1]

    return (transparent_count / total_pixels) * 100


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
        style=Style.RASTER,
        width=256,
        height=256,
        cmap="viridis",
        colorscalerange=colorscalerange,
        format=ImageFormat.PNG,
    )


def assert_render_matches_snapshot(result: io.BytesIO, png_snapshot):
    """Helper function to validate PNG content against snapshot."""
    assert isinstance(result, io.BytesIO)
    result.seek(0)
    content = result.read()

    assert len(content) > 0

    # Check for transparent pixels - there should be none with bbox padding
    transparent_percent = check_transparent_pixels(content)
    assert (
        transparent_percent == 0
    ), f"Found {transparent_percent:.1f}% transparent pixels."

    assert content == png_snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("tile,tms", TILES)
async def test_pipeline_tiles(global_datasets, tile, tms, png_snapshot):
    """Test pipeline with various tiles using their native TMS CRS."""
    ds = global_datasets
    query_params = create_query_params(tile, tms)
    result = await pipeline(ds, query_params)
    assert_render_matches_snapshot(result, png_snapshot)


@pytest.mark.asyncio
async def test_high_zoom_tile_global_dataset(png_snapshot):
    ds = create_global_dataset()
    tms = WEBMERC_TMS

    tile = morecantile.Tile(x=524288 + 2916, y=262144, z=20)

    query_params = create_query_params(tile, tms, colorscalerange=(-1, 1))
    # Run the full pipeline
    result = await pipeline(ds, query_params)
    assert_render_matches_snapshot(result, png_snapshot)
