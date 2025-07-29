#!/usr/bin/env python3

import io

import cf_xarray  # noqa: F401 - Enable cf accessor
import pytest
from pyproj import CRS
from pyproj.aoi import BBox

from tests.tiles import TILES
from xpublish_tiles.pipeline import pipeline
from xpublish_tiles.types import OutputBBox, OutputCRS, QueryParams, Style


def create_query_params(tile, tms):
    """Create QueryParams instance using test tiles and TMS."""

    # Convert TMS CRS to pyproj CRS
    target_crs = CRS.from_epsg(tms.crs.to_epsg())

    # Get bounds in the TMS's native CRS
    if tms.crs.to_epsg() == 4326:
        # For WGS84 TMS, bounds() gives WGS84 coordinates
        geo_bounds = tms.bounds(tile)
        bbox = BBox(
            west=geo_bounds[0],
            south=geo_bounds[1],
            east=geo_bounds[2],
            north=geo_bounds[3],
        )
    else:
        # For projected TMS, get bounds in the native projection
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
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tile,tms", TILES)
async def test_pipeline_tiles(global_datasets, tile, tms, snapshot):
    """Test pipeline with various tiles using their native TMS CRS."""
    ds = global_datasets
    query_params = create_query_params(tile, tms)
    result = await pipeline(ds, query_params)
    assert isinstance(result, io.BytesIO)
    result.seek(0)
    content = result.read()
    assert len(content) > 0
    assert content == snapshot
