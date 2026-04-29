"""Smoke tests for the vector tile renderer (MVT + GeoJSON)."""

import gzip
import json

import pytest
from morecantile import Tile
from pyproj import CRS
from pyproj.aoi import BBox

from xpublish_tiles import config
from xpublish_tiles.pipeline import pipeline
from xpublish_tiles.testing.datasets import create_global_dataset
from xpublish_tiles.testing.tiles import WEBMERC_TMS
from xpublish_tiles.types import ImageFormat, OutputBBox, OutputCRS, QueryParams


def _vector_query(tile, tms, *, format: ImageFormat) -> QueryParams:
    epsg_code = tms.crs.to_epsg()
    target_crs = (
        CRS.from_epsg(epsg_code)
        if epsg_code is not None
        else CRS.from_user_input(tms.crs)
    )
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
        style="vector",
        width=256,
        height=256,
        variant="default",
        colorscalerange=(-1.0, 1.0),
        format=format,
    )


@pytest.mark.asyncio
async def test_vector_mvt_smoke():
    """End-to-end: vector style + MVT format yields a gzipped non-empty protobuf
    that starts with the Tile.layers wire tag (field=3, type=2 → 0x1a)."""
    ds = create_global_dataset(nlat=180, nlon=361)
    tile = Tile(x=0, y=0, z=0)
    query = _vector_query(tile, WEBMERC_TMS, format=ImageFormat.MVT)
    with config.set(rectilinear_check_min_size=0):
        result = await pipeline(ds, query)
    raw = result.getvalue()
    assert raw, "Expected non-empty MVT response"

    decompressed = gzip.decompress(raw)
    assert decompressed, "Decompressed MVT should have at least one layer"
    assert decompressed[0] == 0x1A, (
        f"Expected first byte to be the Tile.layers tag (0x1a), got {decompressed[0]:#04x}"
    )


@pytest.mark.asyncio
async def test_vector_geojson_smoke():
    """End-to-end: vector style + GeoJSON format yields a parseable
    FeatureCollection of polygons."""
    ds = create_global_dataset(nlat=180, nlon=361)
    tile = Tile(x=0, y=0, z=0)
    query = _vector_query(tile, WEBMERC_TMS, format=ImageFormat.GEOJSON)
    with config.set(rectilinear_check_min_size=0):
        result = await pipeline(ds, query)
    fc = json.loads(result.getvalue().decode("utf-8"))
    assert fc["type"] == "FeatureCollection"
    assert isinstance(fc["features"], list)
    assert len(fc["features"]) > 0, "Expected at least one feature for a global tile"

    sample = fc["features"][0]
    assert sample["type"] == "Feature"
    assert sample["geometry"]["type"] == "Polygon"
    ring = sample["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1], "Polygon ring must be closed"
    assert len(ring) >= 4, "Polygon ring must have at least 4 coordinates"
    # GeoJSON coordinates must be in WGS84 lon/lat per RFC 7946.
    for lon, lat in ring:
        assert -180.0 <= lon <= 360.0
        assert -90.0 <= lat <= 90.0


@pytest.mark.asyncio
async def test_vector_max_features_per_side_clamping():
    """The query param's max_features_per_side is clamped to the server cap."""
    from xpublish_tiles.lib import max_render_shape

    # Server config sets the absolute hard cap.
    with config.set(vector_max_features_per_side=512):
        # Below cap: client value is honored.
        assert max_render_shape(style="vector", max_features_per_side=128) == (128, 128)
        # Above cap: clamped down to the server cap.
        assert max_render_shape(style="vector", max_features_per_side=4096) == (512, 512)
        # None: defaults to the server cap (max detail).
        assert max_render_shape(style="vector", max_features_per_side=None) == (
            512,
            512,
        )
        # Zero / negative is clamped up to 1.
        assert max_render_shape(style="vector", max_features_per_side=0) == (1, 1)


@pytest.mark.asyncio
async def test_vector_format_aliases():
    """`f=mvt`, `f=application/vnd.mapbox-vector-tile`, and the StrEnum value
    all map to ``ImageFormat.MVT``."""
    from xpublish_tiles.validators import validate_image_format

    assert validate_image_format("mvt") is ImageFormat.MVT
    assert validate_image_format("application/vnd.mapbox-vector-tile") is ImageFormat.MVT
    assert validate_image_format("geojson") is ImageFormat.GEOJSON
    assert validate_image_format("application/geo+json") is ImageFormat.GEOJSON
    # Backwards-compat with the legacy `image/<sub>` form
    assert validate_image_format("image/png") is ImageFormat.PNG
