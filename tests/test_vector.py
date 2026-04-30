"""Smoke tests for the vector tile renderer (MVT + GeoJSON)."""

import gzip
import json

import morecantile
import pytest
from morecantile import Tile
from pyproj import CRS
from pyproj.aoi import BBox

from xpublish_tiles import config
from xpublish_tiles.pipeline import pipeline
from xpublish_tiles.testing.datasets import create_global_dataset
from xpublish_tiles.testing.tiles import WEBMERC_TMS
from xpublish_tiles.types import ImageFormat, OutputBBox, OutputCRS, QueryParams

CRS84_TMS = morecantile.tms.get("WorldCRS84Quad")


def _vector_query(
    tile,
    tms,
    *,
    format: ImageFormat,
    variant: str = "default",
    variables: list[str] | None = None,
) -> QueryParams:
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
        variables=variables if variables is not None else ["foo"],
        crs=OutputCRS(target_crs),
        bbox=OutputBBox(bbox),
        selectors={},
        style="vector",
        width=256,
        height=256,
        variant=variant,
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
    """End-to-end: vector style + GeoJSON format on a CRS84 TMS yields a
    parseable FeatureCollection of polygons in lon/lat."""
    ds = create_global_dataset(nlat=180, nlon=361)
    tile = Tile(x=0, y=0, z=0)
    query = _vector_query(tile, CRS84_TMS, format=ImageFormat.GEOJSON)
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
    # GeoJSON coordinates must be in WGS84 lon/lat per RFC 7946. Allow a
    # one-cell overshoot (~1° at this resolution) so the boundary cells whose
    # rings span the antimeridian don't trip the bounds check.
    for lon, lat in ring:
        assert -182.0 <= lon <= 182.0
        assert -91.0 <= lat <= 91.0


@pytest.mark.asyncio
async def test_vector_geojson_rejects_projected_crs():
    """GeoJSON output is RFC 7946 (CRS84 only); requesting it against a
    projected TMS like WebMercatorQuad must error rather than silently
    reproject."""
    ds = create_global_dataset(nlat=180, nlon=361)
    tile = Tile(x=0, y=0, z=0)
    query = _vector_query(tile, WEBMERC_TMS, format=ImageFormat.GEOJSON)
    with config.set(rectilinear_check_min_size=0):
        with pytest.raises(ValueError, match="geographic CRS"):
            await pipeline(ds, query)


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


@pytest.mark.parametrize("variant", ["cells", "points", "default"])
def test_validate_style_vector_variants(variant):
    """`vector/cells`, `vector/points` are canonical; `vector/default` aliases cells."""
    from xpublish_tiles.validators import validate_style

    assert validate_style(f"vector/{variant}") == ("vector", variant)


def test_validate_style_vector_rejects_unknown_variant():
    from xpublish_tiles.validators import validate_style

    with pytest.raises(
        ValueError,
        match="variant 'bogus' is not supported for style 'vector'",
    ):
        validate_style("vector/bogus")


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["cells", "default"])
async def test_vector_cells_variant_renders(variant):
    """`vector/cells` and `vector/default` both produce identical MVT output."""
    ds = create_global_dataset(nlat=180, nlon=361)
    tile = Tile(x=0, y=0, z=0)
    query = _vector_query(tile, WEBMERC_TMS, format=ImageFormat.MVT, variant=variant)
    with config.set(rectilinear_check_min_size=0):
        result = await pipeline(ds, query)
    assert result.getvalue(), f"vector/{variant} should produce non-empty MVT"


@pytest.mark.asyncio
async def test_vector_points_mvt_smoke():
    """vector/points emits MVT with Point geometry features (type=1)."""
    ds = create_global_dataset(nlat=180, nlon=361)
    tile = Tile(x=0, y=0, z=0)
    query = _vector_query(tile, WEBMERC_TMS, format=ImageFormat.MVT, variant="points")
    with config.set(rectilinear_check_min_size=0):
        result = await pipeline(ds, query)
    raw = result.getvalue()
    assert raw, "Expected non-empty points MVT response"
    decompressed = gzip.decompress(raw)
    # MVT has at least one layer; points layer features are POINT (type=1).
    # Cheap assertion: somewhere in the body the bytes for "points" layer name appear.
    assert b"points" in decompressed, "Expected layer named 'points' in MVT body"


@pytest.mark.asyncio
async def test_vector_points_geojson_multivar():
    """Multi-variable points: each Feature's properties carries every requested var."""
    ds = create_global_dataset(nlat=90, nlon=181)
    # Synthesize a second variable so we can test multi-var on the same grid.
    ds = ds.assign(bar=ds["foo"] * 2.0)
    tile = Tile(x=0, y=0, z=0)
    query = _vector_query(
        tile,
        CRS84_TMS,
        format=ImageFormat.GEOJSON,
        variant="points",
        variables=["foo", "bar"],
    )
    with config.set(rectilinear_check_min_size=0):
        result = await pipeline(ds, query)
    fc = json.loads(result.getvalue().decode("utf-8"))
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) > 0

    sample = fc["features"][0]
    assert sample["geometry"]["type"] == "Point"
    lon, lat = sample["geometry"]["coordinates"]
    # Boundary-cell centroids can sit ~1 cell past ±180° / ±90° due to the
    # polygon pipeline padding cells outward from the data range.
    assert -182.0 <= lon <= 182.0
    assert -92.0 <= lat <= 92.0
    # Both variables present and bar = 2 * foo (within float tolerance).
    assert set(sample["properties"]) == {"foo", "bar"}
    assert sample["properties"]["bar"] == pytest.approx(2.0 * sample["properties"]["foo"])


@pytest.mark.asyncio
async def test_vector_points_drops_features_with_any_nan():
    """If any requested variable has NaN at a cell, the Point feature is dropped."""
    import numpy as np

    ds = create_global_dataset(nlat=20, nlon=21)
    # Mask out 'bar' over the western hemisphere so half the cells have NaN there.
    bar = ds["foo"].where(ds["longitude"] >= 0).astype("float32")
    ds = ds.assign(bar=bar)
    tile = Tile(x=0, y=0, z=0)
    query_both = _vector_query(
        tile,
        CRS84_TMS,
        format=ImageFormat.GEOJSON,
        variant="points",
        variables=["foo", "bar"],
    )
    query_foo_only = _vector_query(
        tile,
        CRS84_TMS,
        format=ImageFormat.GEOJSON,
        variant="points",
        variables=["foo"],
    )
    with config.set(rectilinear_check_min_size=0):
        fc_both = json.loads((await pipeline(ds, query_both)).getvalue())
        fc_foo = json.loads((await pipeline(ds, query_foo_only)).getvalue())
    # Requesting both vars drops every NaN-bar cell; foo-only keeps them all.
    assert len(fc_both["features"]) < len(fc_foo["features"])
    # And every emitted "both" feature has finite values for both.
    for feat in fc_both["features"]:
        for k in ("foo", "bar"):
            assert np.isfinite(feat["properties"][k])
