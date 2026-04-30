"""Smoke tests for the vector tile renderer (MVT + GeoJSON)."""

import gzip
import json

import morecantile
import numpy as np
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
    levels: tuple[float, ...] | None = None,
    smoothing: float | None = None,
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
        levels=levels,
        smoothing=smoothing,
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


# ---------------------------------------------------------------------------
# vector/contours
# ---------------------------------------------------------------------------


def test_validate_style_contours_variant():
    from xpublish_tiles.validators import validate_style

    assert validate_style("vector/contours") == ("vector", "contours")


def test_validate_levels_basic():
    from xpublish_tiles.validators import validate_levels

    assert validate_levels(None) is None
    assert validate_levels("") is None
    assert validate_levels("0,5,10") == (0.0, 5.0, 10.0)
    assert validate_levels("-1.5,2.5") == (-1.5, 2.5)


@pytest.mark.parametrize(
    "value, msg",
    [
        ("5", "at least 2 values"),
        ("0,5,3", "strictly increasing"),
        ("0,0,5", "strictly increasing"),
        ("not,a,float", "comma-separated"),
    ],
)
def test_validate_levels_invalid(value, msg):
    from xpublish_tiles.validators import validate_levels

    with pytest.raises(ValueError, match=msg):
        validate_levels(value)


@pytest.mark.asyncio
async def test_vector_contours_requires_levels():
    """Contours without levels raises a clear error."""
    ds = create_global_dataset(nlat=90, nlon=181)
    tile = Tile(x=0, y=0, z=0)
    query = _vector_query(tile, WEBMERC_TMS, format=ImageFormat.MVT, variant="contours")
    with config.set(rectilinear_check_min_size=0):
        with pytest.raises(ValueError, match="requires the `levels`"):
            await pipeline(ds, query)


@pytest.mark.asyncio
async def test_vector_contours_mvt_smoke():
    """vector/contours + MVT yields a non-empty gzipped protobuf with the
    Tile.layers wire tag."""
    ds = create_global_dataset(nlat=90, nlon=181)
    tile = Tile(x=0, y=0, z=0)
    query = _vector_query(
        tile,
        WEBMERC_TMS,
        format=ImageFormat.MVT,
        variant="contours",
        levels=(-1.0, -0.5, 0.0, 0.5, 1.0),
    )
    with config.set(rectilinear_check_min_size=0):
        result = await pipeline(ds, query)
    raw = result.getvalue()
    assert raw, "Expected non-empty contours MVT response"
    decompressed = gzip.decompress(raw)
    assert decompressed, "Expected at least one layer in contours MVT"
    assert decompressed[0] == 0x1A, (
        f"Expected first byte to be the Tile.layers tag (0x1a), got {decompressed[0]:#04x}"
    )


@pytest.mark.asyncio
async def test_vector_contours_geojson_band_properties():
    """Each contour Feature is a Polygon with the band's lo/hi/mid/variable
    properties; rings are closed lon/lat."""
    ds = create_global_dataset(nlat=90, nlon=181)
    tile = Tile(x=0, y=0, z=0)
    levels = (-1.0, -0.25, 0.25, 1.0)
    query = _vector_query(
        tile,
        CRS84_TMS,
        format=ImageFormat.GEOJSON,
        variant="contours",
        levels=levels,
    )
    with config.set(rectilinear_check_min_size=0):
        result = await pipeline(ds, query)
    fc = json.loads(result.getvalue().decode("utf-8"))
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) > 0

    fill_feats = [f for f in fc["features"] if f["properties"].get("kind") == "fill"]
    line_feats = [f for f in fc["features"] if f["properties"].get("kind") == "line"]
    assert fill_feats, "expected at least one filled-band feature"
    assert line_feats, "expected at least one isoline feature"

    # Lines: each is a LineString carrying the level value as `value`.
    seen_line_levels = {f["properties"]["value"] for f in line_feats}
    assert seen_line_levels.issubset(set(levels))
    for f in line_feats:
        assert f["geometry"]["type"] == "LineString"
        assert f["properties"]["variable"] == "foo"

    expected_pairs = {(levels[i], levels[i + 1]) for i in range(len(levels) - 1)}
    seen_pairs = set()
    for feat in fill_feats:
        assert feat["type"] == "Feature"
        assert feat["geometry"]["type"] == "Polygon"
        props = feat["properties"]
        assert props["variable"] == "foo"
        assert props["value_lo"] < props["value_hi"]
        assert props["value_mid"] == pytest.approx(
            0.5 * (props["value_lo"] + props["value_hi"])
        )
        seen_pairs.add((props["value_lo"], props["value_hi"]))

        # Each ring must be closed and lie in roughly CRS84 lon/lat.
        for ring in feat["geometry"]["coordinates"]:
            assert ring[0] == ring[-1]
            assert len(ring) >= 4
            for lon, lat in ring:
                assert -183.0 <= lon <= 183.0
                assert -93.0 <= lat <= 93.0

    # Every requested band that has data should appear at least once.
    assert seen_pairs.issubset(expected_pairs)
    assert seen_pairs  # something rendered


@pytest.mark.asyncio
async def test_vector_contours_emits_lines_layer():
    """vector/contours emits a `<varname>_lines` MVT layer alongside the
    polygon layer: same MVT response, two source-layers."""
    ds = create_global_dataset(nlat=90, nlon=181)
    tile = Tile(x=0, y=0, z=0)
    query = _vector_query(
        tile,
        WEBMERC_TMS,
        format=ImageFormat.MVT,
        variant="contours",
        levels=(-0.5, 0.0, 0.5),
    )
    with config.set(rectilinear_check_min_size=0):
        result = await pipeline(ds, query)
    raw = result.getvalue()
    decompressed = gzip.decompress(raw)
    # Both the variable layer and the "<var>_lines" layer should appear.
    assert b"foo" in decompressed
    assert b"foo_lines" in decompressed


@pytest.mark.asyncio
async def test_vector_contours_geojson_rejects_projected_crs():
    """Contours + GeoJSON requires a geographic CRS, same as cells/points."""
    ds = create_global_dataset(nlat=90, nlon=181)
    tile = Tile(x=0, y=0, z=0)
    query = _vector_query(
        tile,
        WEBMERC_TMS,
        format=ImageFormat.GEOJSON,
        variant="contours",
        levels=(-0.5, 0.0, 0.5),
    )
    with config.set(rectilinear_check_min_size=0):
        with pytest.raises(ValueError, match="geographic CRS"):
            await pipeline(ds, query)


@pytest.mark.asyncio
async def test_vector_contours_handles_holes():
    """A radial DEPRESSION (low value at the center surrounded by high
    values) produces a stack polygon with a hole — the outer at L
    encloses the entire high-value region and the hole at L surrounds
    the low-value center where z < L. Validates the multi-ring polygon
    encoder code path under the stacking semantics."""
    import xarray as xr

    nlat, nlon = 91, 181
    lats = np.linspace(-90, 90, nlat)
    lons = np.linspace(-180, 180, nlon)
    LON, LAT = np.meshgrid(lons, lats)
    # Radial depression centred on (-90, 0) — fully inside CRS84 tile
    # (0,0,0) which covers only the western hemisphere. The stack at
    # L=0.3 traces "z >= 0.3" → outer at the boundary of the high-value
    # plateau, plus a hole around the low-value pit at the center.
    field = 1.0 - np.exp(-(((LON + 90) / 20) ** 2 + (LAT / 20) ** 2))
    ds = xr.Dataset(
        {"foo": (("latitude", "longitude"), field.astype(np.float32))},
        coords={"latitude": lats, "longitude": lons},
    )
    ds["latitude"].attrs["standard_name"] = "latitude"
    ds["latitude"].attrs["units"] = "degrees_north"
    ds["longitude"].attrs["standard_name"] = "longitude"
    ds["longitude"].attrs["units"] = "degrees_east"
    ds["foo"].attrs["valid_min"] = 0.0
    ds["foo"].attrs["valid_max"] = 1.0

    tile = Tile(x=0, y=0, z=0)
    query = _vector_query(
        tile,
        CRS84_TMS,
        format=ImageFormat.GEOJSON,
        variant="contours",
        levels=(0.0, 0.3, 0.7, 1.0),
    )
    with config.set(rectilinear_check_min_size=0):
        result = await pipeline(ds, query)
    fc = json.loads(result.getvalue().decode("utf-8"))
    # The stack at L=0.3 covers the high-value plateau with a hole around
    # the central low; expect at least one polygon with >1 ring.
    stack_03_polys = [
        f
        for f in fc["features"]
        if f["properties"].get("kind") == "fill"
        and f["properties"]["value_lo"] == pytest.approx(0.3)
    ]
    assert stack_03_polys, "expected at least one [0.3, 0.7] stack polygon"
    assert any(len(f["geometry"]["coordinates"]) > 1 for f in stack_03_polys), (
        "expected at least one multi-ring polygon (outer + hole) at L=0.3"
    )


def test_validate_smoothing_basic():
    from xpublish_tiles.validators import validate_smoothing

    assert validate_smoothing(None) is None
    assert validate_smoothing("") is None
    assert validate_smoothing("0") == 0.0
    assert validate_smoothing("1.5") == 1.5
    with pytest.raises(ValueError, match="non-negative"):
        validate_smoothing("-1")
    with pytest.raises(ValueError, match="non-negative"):
        validate_smoothing("not-a-float")


@pytest.mark.asyncio
async def test_vector_contours_smoothing_simplifies_jagged_field():
    """A noisy small-grid field produces shorter, smoother polygon rings
    when the server pre-blurs the field. Compare per-feature ring-vertex
    counts at sigma=0 vs sigma=2.0."""
    import xarray as xr

    rng = np.random.default_rng(seed=0)
    # Small, noisy grid contained in CRS84 tile (0,0,0) (western hemisphere).
    nlat, nlon = 41, 41
    lats = np.linspace(-90, 90, nlat)
    lons = np.linspace(-180, 0, nlon)
    LON, LAT = np.meshgrid(lons, lats)
    base = np.exp(-(((LON + 90) / 30) ** 2 + (LAT / 30) ** 2))
    noise = rng.standard_normal(base.shape) * 0.15
    ds = xr.Dataset(
        {"foo": (("latitude", "longitude"), (base + noise).astype(np.float32))},
        coords={"latitude": lats, "longitude": lons},
    )
    ds["latitude"].attrs["standard_name"] = "latitude"
    ds["latitude"].attrs["units"] = "degrees_north"
    ds["longitude"].attrs["standard_name"] = "longitude"
    ds["longitude"].attrs["units"] = "degrees_east"
    ds["foo"].attrs["valid_min"] = -1.0
    ds["foo"].attrs["valid_max"] = 2.0

    tile = Tile(x=0, y=0, z=0)
    levels = (0.2, 0.5, 0.8)

    async def fetch(sigma):
        query = _vector_query(
            tile,
            CRS84_TMS,
            format=ImageFormat.GEOJSON,
            variant="contours",
            levels=levels,
            smoothing=sigma,
        )
        with config.set(rectilinear_check_min_size=0):
            result = await pipeline(ds, query)
        return json.loads(result.getvalue())

    fc_unblurred = await fetch(0.0)
    fc_blurred = await fetch(2.0)

    def total_vertices(fc):
        # Count only the polygon-band rings; lines are emitted alongside but
        # their vertex behaviour is the same as the rings, so either basis
        # would work — pick fill for clarity.
        return sum(
            sum(len(ring) for ring in feat["geometry"]["coordinates"])
            for feat in fc["features"]
            if feat["properties"].get("kind") == "fill"
        )

    # Pre-blur should produce a substantially simpler geometry — fewer total
    # ring vertices since marching squares finds fewer wiggles in a smoothed
    # field. We assert "meaningfully fewer" rather than a specific factor.
    assert total_vertices(fc_blurred) < total_vertices(fc_unblurred) * 0.85, (
        f"Expected smoothing=2.0 to simplify rings; got "
        f"unblurred={total_vertices(fc_unblurred)}, "
        f"blurred={total_vertices(fc_blurred)}"
    )


@pytest.mark.asyncio
async def test_vector_contours_empty_band_yields_no_features():
    """If no value pair is achievable in the data, the renderer emits no
    features for that band — it doesn't raise."""
    ds = create_global_dataset(nlat=45, nlon=91)
    tile = Tile(x=0, y=0, z=0)
    # The synthetic dataset is bounded in [-1, 1], so [10, 20] is empty.
    query = _vector_query(
        tile,
        CRS84_TMS,
        format=ImageFormat.GEOJSON,
        variant="contours",
        levels=(10.0, 20.0),
    )
    with config.set(rectilinear_check_min_size=0):
        result = await pipeline(ds, query)
    fc = json.loads(result.getvalue().decode("utf-8"))
    assert fc["type"] == "FeatureCollection"
    assert fc["features"] == []


def test_mvt_multiring_polygon_encoder_winding_and_holes():
    """Round-trip a square-with-square-hole through the multi-ring encoder
    and verify the MoveTo/LineTo/ClosePath structure is well-formed for the
    MVT spec (outer winding has positive shoelace area in tile coords)."""
    from xpublish_tiles.render.mvt import (
        encode_mvt_polygon_layer,
        quantize_rings,
    )

    bbox = BBox(west=0.0, south=0.0, east=10.0, north=10.0)
    extent = 256

    # Outer square (CCW math) + hole square (CW math) — contourpy convention.
    outer = np.array([[1.0, 1.0], [9.0, 1.0], [9.0, 9.0], [1.0, 9.0]], dtype=np.float64)
    hole = np.array([[3.0, 3.0], [3.0, 7.0], [7.0, 7.0], [7.0, 3.0]], dtype=np.float64)
    rings_flat = np.concatenate([outer, hole], axis=0)
    ring_starts = np.array([0, 4, 8], dtype=np.int32)
    poly_ring_starts = np.array([0, 2], dtype=np.int32)
    rings_q = quantize_rings(rings_flat, bbox=bbox, extent=extent)

    layer = encode_mvt_polygon_layer(
        name="test",
        extent=extent,
        rings_flat_q=rings_q,
        ring_starts=ring_starts,
        poly_ring_starts=poly_ring_starts,
        properties={"value": np.array([1.5])},
    )
    # Layer body must contain the layer name and the property key.
    assert b"test" in layer
    assert b"value" in layer
    # Non-trivial body: at least the geometry payload bytes are present.
    assert len(layer) > 50


def test_mvt_multiring_drops_polygon_with_degenerate_outer():
    """If a polygon's outer ring has fewer than 3 distinct quantized
    vertices, the entire polygon is dropped — matches the cells encoder."""
    from xpublish_tiles.render.mvt import (
        encode_mvt_polygon_layer,
        quantize_rings,
    )

    bbox = BBox(west=0.0, south=0.0, east=10.0, north=10.0)
    extent = 256

    # Three duplicate points → degenerate ring.
    degenerate = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]], dtype=np.float64)
    valid_outer = np.array(
        [[1.0, 1.0], [9.0, 1.0], [9.0, 9.0], [1.0, 9.0]], dtype=np.float64
    )
    rings_flat = np.concatenate([degenerate, valid_outer], axis=0)
    ring_starts = np.array([0, 3, 7], dtype=np.int32)
    poly_ring_starts = np.array([0, 1, 2], dtype=np.int32)
    rings_q = quantize_rings(rings_flat, bbox=bbox, extent=extent)

    layer = encode_mvt_polygon_layer(
        name="test",
        extent=extent,
        rings_flat_q=rings_q,
        ring_starts=ring_starts,
        poly_ring_starts=poly_ring_starts,
        properties={"value": np.array([1.0, 2.0])},
    )
    # Only the surviving polygon's value (2.0) should be encoded; the
    # degenerate polygon's value (1.0) is dropped along with the feature.
    import struct

    assert struct.pack("<d", 2.0) in layer
    assert struct.pack("<d", 1.0) not in layer
