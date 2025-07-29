#!/usr/bin/env python3

import io

import cf_xarray  # noqa: F401 - Enable cf accessor
import numpy as np
import pytest
from PIL import Image
from pyproj import CRS
from pyproj.aoi import BBox

from tests.tiles import TILES
from xpublish_tiles.pipeline import pipeline
from xpublish_tiles.types import ImageFormat, OutputBBox, OutputCRS, QueryParams, Style


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


def create_query_params(tile, tms):
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
        colorscalerange=None,
        format=ImageFormat.PNG,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tile,tms", TILES)
async def test_pipeline_tiles(global_datasets, tile, tms, png_snapshot):
    """Test pipeline with various tiles using their native TMS CRS."""
    ds = global_datasets
    query_params = create_query_params(tile, tms)
    result = await pipeline(ds, query_params)
    assert isinstance(result, io.BytesIO)
    result.seek(0)
    content = result.read()
    assert len(content) > 0

    # Check for transparent pixels - there should be very few (< 2% tolerance for edge effects)
    transparent_percent = check_transparent_pixels(content)
    assert transparent_percent < 2.0, (
        f"Found {transparent_percent:.1f}% transparent pixels in tile "
        f"{tile} with TMS {tms.id}. This indicates a data transformation issue."
    )

    assert content == png_snapshot
