import io
import json
import sys

import cf_xarray.geometry
import datashader as dsh
import morecantile
import numpy as np
import numpy.testing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
import spatialpandas
import xvec  # noqa: F401
from morecantile import Tile
from PIL import Image
from pyiceberg.catalog.sql import SqlCatalog
from pyproj import CRS
from pyproj.aoi import BBox

import xarray as xr
import zarr
import zarr.storage
from tests import create_query_params
from xpublish_tiles import config
from xpublish_tiles.cli.main import (
    get_dataset_for_name,
    open_geoparquet,
    open_iceberg_features,
)
from xpublish_tiles.features import features_to_multipolygons, split_at_antimeridian
from xpublish_tiles.grids import (
    Features,
    GeometryIndexer,
    PointFeatures,
    Rectilinear,
    detect_grids,
    guess_grid_metadata,
    guess_grid_system,
)
from xpublish_tiles.lib import (
    PointQueryNotSupportedError,
    TileTooBigError,
    UnsupportedGridError,
    UnsupportedStyleError,
    unwrap_seam_vertices,
)
from xpublish_tiles.pipeline import bbox_overlap, pipeline
from xpublish_tiles.projections import transformer_from_crs
from xpublish_tiles.testing.datasets import (
    FEATURE_GEOMS,
    FEATURES_POINTS,
    FEATURES_POINTS_XVEC,
    FEATURES_POLYGONS,
    FEATURES_POLYGONS_XVEC,
    GLOBAL_6KM,
    POINT_GEOMS,
    create_global_dataset,
)
from xpublish_tiles.testing.tiles import WEBMERC_TMS
from xpublish_tiles.tiles_lib import get_min_zoom

W = 2 * 20037508.342789244
T = transformer_from_crs(CRS(4326), CRS(3857))


def _feature_table() -> tuple[pa.Table, dict]:
    """(test only) Arrow table of FEATURE_GEOMS as WKB, plus its GeoParquet ``geo`` metadata."""
    n = len(FEATURE_GEOMS)
    table = pa.table(
        {
            "geometry": shapely.to_wkb(np.array(FEATURE_GEOMS, dtype=object)),
            "value": np.arange(n, dtype=np.float64),
            "count": np.arange(n, dtype=np.int64),
            "name": [f"f{i}" for i in range(n)],
        }
    )
    geo = {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "geometry_types": []}},
    }
    return table, geo


def _check_feature_dataset(ds: xr.Dataset) -> None:
    """(test only) Asserts for a dataset built from ``_feature_table``."""
    assert set(ds.data_vars) == {"value", "count"}
    assert ds["name"].dims == ("feature",)
    grid = guess_grid_system(ds, "value")
    assert isinstance(grid, Features)
    assert grid.crs == CRS("OGC:CRS84")
    assert grid.bbox == BBox(west=-180, east=180, south=0, north=60)


def _iceberg_table(tmp_path, properties: dict[str, str]):
    """(test only) ``_feature_table`` written to a local SQLite Iceberg catalog."""
    catalog = SqlCatalog(
        "local",
        uri=f"sqlite:///{tmp_path}/catalog.db",
        warehouse=f"file://{tmp_path}",
    )
    catalog.create_namespace("ns")
    arrow, _ = _feature_table()
    table = catalog.create_table("ns.t", arrow.schema, properties=properties)
    table.append(arrow)
    return table


def test_open_geoparquet(tmp_path):
    table, geo = _feature_table()
    table = table.replace_schema_metadata({"geo": json.dumps(geo)})
    path = tmp_path / "features.parquet"
    pq.write_table(table, path)

    ds = open_geoparquet(str(path))
    assert ds.attrs["_xpublish_id"] == f"geoparquet+file:{path}"
    _check_feature_dataset(ds)


def test_open_iceberg_features(tmp_path):
    _, geo = _feature_table()
    table = _iceberg_table(tmp_path, {"geo": json.dumps(geo)})
    ds = open_iceberg_features(table, "arraylake+table:o/ns.t")
    assert ds.attrs["_xpublish_id"] == "arraylake+table:o/ns.t"
    _check_feature_dataset(ds)


def test_open_iceberg_features_needs_geo(tmp_path):
    table = _iceberg_table(tmp_path, {})
    with pytest.raises(ValueError, match="no 'geo' property"):
        open_iceberg_features(table, "arraylake+table:o/ns.t")


def test_arraylake_table_rejects_branch():
    with pytest.raises(ValueError, match="--branch"):
        get_dataset_for_name("arraylake+table://o/ns.t", branch="dev")


@pytest.mark.parametrize("fixture", [FEATURES_POLYGONS, FEATURES_POLYGONS_XVEC])
def test_detect_features(fixture):
    ds = fixture.create()
    meta = guess_grid_metadata(ds)
    assert meta is not None
    assert meta.grid_cls is Features
    grid = guess_grid_system(ds, "foo")
    assert isinstance(grid, Features)
    assert grid.dim == "feature"
    _assert_split(grid.geometries[3])
    assert (shapely.get_num_geometries(grid.geometries[[0, 1, 4, 5]]) == 1).all()
    assert grid.bbox == BBox(west=-180, east=180, south=0, north=60)
    assert not grid.is_structural_variable(ds["foo"])


def _assert_split(geom) -> None:
    """(test only) ``geom`` is split at 180°: parts on both sides, all in [-180, 180]."""
    parts = shapely.get_parts(geom)
    bounds = shapely.bounds(parts)
    assert len(parts) >= 2
    assert bounds[:, 0].min() == -180 and bounds[:, 2].max() == 180
    assert (bounds[:, 0] >= -180).all() and (bounds[:, 2] <= 180).all()


def test_detect_features_cf_without_geometry_attr():
    ds = FEATURES_POLYGONS.create()
    del ds["foo"].attrs["geometry"]
    ds.attrs["_xpublish_id"] = "features-no-geometry-attr"
    assert isinstance(guess_grid_system(ds, "foo"), Features)


def test_features_structural_variables_not_layers():
    ds = FEATURES_POLYGONS.create()
    assert set(detect_grids(ds)) == {"foo", "bar"}


def test_features_and_gridded_variable_in_one_dataset():
    ds = xr.merge(
        [FEATURES_POLYGONS.create(), GLOBAL_6KM.create()[["foo"]].rename(foo="t2m")],
        join="outer",
    )
    grids = detect_grids(ds)
    assert isinstance(grids["foo"], Features)
    assert not isinstance(grids["t2m"], Features)


def test_features_crossing_ignores_wide_feature():
    geoms = [shapely.segmentize(shapely.box(-100, 0, 100, 10), 100), FEATURE_GEOMS[3]]
    ds = xr.Dataset(coords={"feature": geoms}).xvec.set_geom_indexes("feature", crs=4326)
    ds["foo"] = ("feature", [1.0, 2.0])
    ds.attrs["_xpublish_id"] = "features-wide"
    grid = guess_grid_system(ds, "foo")
    assert isinstance(grid, Features)
    assert shapely.get_num_geometries(grid.geometries[0]) == 1
    _assert_split(grid.geometries[1])


@pytest.mark.parametrize(
    "geoms",
    [
        pytest.param([shapely.LineString([(0, 0), (1, 1)])], id="line"),
        pytest.param([shapely.Point(0, 0), shapely.box(0, 0, 1, 1)], id="mixed"),
    ],
)
def test_features_unsupported_geometry(geoms):
    ds = xr.Dataset(coords={"feature": geoms}).xvec.set_geom_indexes("feature", crs=4326)
    ds["foo"] = ("feature", np.ones(len(geoms)))
    ds.attrs["_xpublish_id"] = f"features-unsupported-{len(geoms)}"
    with pytest.raises(UnsupportedGridError):
        guess_grid_metadata(ds)
    with pytest.raises(UnsupportedGridError, match="only polygons or only points"):
        guess_grid_system(ds, "foo")


def _features_and_gridded(xpublish_id: str) -> xr.Dataset:
    ds = xr.merge(
        [FEATURES_POLYGONS.create(), GLOBAL_6KM.create()[["foo"]].rename(foo="t2m")],
        join="outer",
    )
    ds.attrs["_xpublish_id"] = xpublish_id
    return ds


def _block_shapely(monkeypatch):
    for module in ("shapely", "cf_xarray.geometry"):
        monkeypatch.setitem(sys.modules, module, None)
    # force a fresh import so the module's import guard runs
    monkeypatch.delitem(sys.modules, "xpublish_tiles.features", raising=False)


def test_features_cf_without_shapely(monkeypatch):
    ds = _features_and_gridded("features-no-shapely")
    _block_shapely(monkeypatch)
    assert isinstance(guess_grid_system(ds, "t2m"), Rectilinear)
    with pytest.raises(ImportError, match=r"xpublish-tiles\[vector\]"):
        guess_grid_system(ds, "foo")


def test_features_cf_line_container_without_shapely(monkeypatch):
    ds = _features_and_gridded("features-cf-line")
    ds["geometry_container"].attrs["geometry_type"] = "line"
    _block_shapely(monkeypatch)
    with pytest.raises(UnsupportedGridError, match="unsupported geometry_type"):
        guess_grid_system(ds, "foo")
    grids = detect_grids(ds)
    assert set(grids) == {"t2m"}
    assert isinstance(grids["t2m"], Rectilinear)


def test_features_presplit_multipolygon_not_crossing():
    geoms = [
        shapely.MultiPolygon(
            [shapely.box(170, 50, 180, 55), shapely.box(-180, 50, -170, 55)]
        )
    ]
    ds = xr.Dataset(coords={"feature": geoms}).xvec.set_geom_indexes("feature", crs=4326)
    ds["foo"] = ("feature", [1.0])
    ds.attrs["_xpublish_id"] = "features-presplit"
    grid = guess_grid_system(ds, "foo")
    assert isinstance(grid, Features)
    assert shapely.get_num_geometries(grid.geometries[0]) == 2
    assert grid.bbox == BBox(west=-180, east=180, south=50, north=55)
    assert shapely.total_bounds(grid.geometries).tolist() == [-180, 50, 180, 55]


@pytest.mark.parametrize(
    "bbox, expected",
    [
        (BBox(west=-105, south=25, east=-85, north=45), [0]),
        (BBox(west=-70, south=38, east=-68, north=42), []),
        (BBox(west=-70, south=38, east=-62, north=42), [1]),
        (BBox(west=-100, south=25, east=-30, north=55), [0, 1, 2]),
        (BBox(west=170, south=45, east=190, north=65), [3]),
        (BBox(west=-178, south=45, east=-170, north=65), [3]),
        (BBox(west=-30, south=-10, east=-20, north=-5), []),
    ],
)
def test_features_sel(bbox, expected):
    grid = guess_grid_system(FEATURES_POLYGONS.create(), "foo")
    (indexer,) = grid.sel(bbox=bbox)["feature"]
    assert isinstance(indexer, GeometryIndexer)
    assert indexer.indices.tolist() == expected


@pytest.mark.parametrize("fixture", [FEATURES_POLYGONS, FEATURES_POLYGONS_XVEC])
def test_features_locate(fixture):
    grid = guess_grid_system(fixture.create(), "foo")
    assert isinstance(grid, Features)
    inside = shapely.point_on_surface(grid.geometries)
    x, y = shapely.get_x(inside), shapely.get_y(inside)
    (positions,) = grid.locate(x=x, y=y).values()
    assert positions.tolist() == list(range(len(FEATURE_GEOMS)))

    points = {
        "hole": (-70, 40),
        "east of antimeridian": (-179, 55),
        "west of antimeridian": (179, 55),
        "wrapped": (181, 55),
        "nothing": (0, -45),
    }
    x, y = np.array(list(points.values()), dtype=float).T
    (positions,) = grid.locate(x=x, y=y).values()
    assert dict(zip(points, positions.tolist(), strict=True)) == {
        "hole": -1,
        "east of antimeridian": 3,
        "west of antimeridian": 3,
        "wrapped": 3,
        "nothing": -1,
    }


async def test_features_tile_with_no_feature_is_transparent():
    tile = Tile(x=6, y=7, z=4)
    tms = morecantile.tms.get("WebMercatorQuad")
    ds = FEATURES_POLYGONS.create()
    grid = guess_grid_system(ds, "foo")
    left, bottom, right, top = tms.bounds(tile)
    tile_bbox = BBox(west=left, south=bottom, east=right, north=top)
    assert bbox_overlap(tile_bbox, grid.bbox, True)
    (indexer,) = grid.sel(bbox=tile_bbox)["feature"]
    assert isinstance(indexer, GeometryIndexer)
    assert indexer.is_empty

    query_params = create_query_params(tile, tms, style="polygons")
    result = await pipeline(ds, query_params)
    alpha = np.asarray(Image.open(io.BytesIO(result.getvalue())).convert("RGBA"))[..., 3]
    assert (alpha == 0).all()


def test_unwrap_seam_vertices():
    x = np.array([10.0, -10.0, 5.0, -5.0])
    unwrap_seam_vertices(x, np.array([0, 1]), width=100.0)
    assert x.tolist() == [10.0, 90.0, 5.0, -5.0]


def test_features_to_multipolygons():
    g = np.array([shapely.box(0, 0, 10, 10)], dtype=object)
    polys = features_to_multipolygons(g, T)
    assert len(polys) == 1
    (x0, x1), (y0, y1) = T.transform([0, 10], [0, 10])
    npt.assert_allclose(polys.bounds[0], [x0, y0, x1, y1])


def test_split_at_antimeridian():
    crosser = shapely.Polygon([(175, 50), (-175, 50), (-175, 60), (175, 60)])
    other = shapely.box(0, 0, 10, 10)
    split = split_at_antimeridian(np.array([crosser, other], dtype=object))
    _assert_split(split[0])
    npt.assert_allclose(shapely.area(split[0]), 100)
    assert shapely.equals(split[1], other)


def test_split_at_antimeridian_crosses_0_and_180():
    # -10° to 190°, sampled every 50°: the ring runs ...-10, 40, 90, 140, -170 (=190)...
    box = shapely.segmentize(shapely.box(-10, 0, 190, 10), 50)
    wrapped = shapely.transform(box, lambda c: np.where(c > 180, c - [360, 0], c))
    (out,) = split_at_antimeridian(np.array([wrapped], dtype=object))
    _assert_split(out)
    npt.assert_allclose(shapely.area(out), 200 * 10)
    for lon in (-5, 100, -175):
        assert out.contains(shapely.Point(lon, 5))
    assert not out.contains(shapely.Point(-100, 5))


async def test_features_vertex_limit_raises():
    ds = FEATURES_POLYGONS.create()
    query = create_query_params(Tile(x=0, y=0, z=0), WEBMERC_TMS, style="polygons")
    with config.set({"max_renderable_vertices": 10}), pytest.raises(TileTooBigError):
        await pipeline(ds, query)


def test_features_min_zoom_uses_vertex_limit():
    ds = FEATURES_POLYGONS.create()
    grid = guess_grid_system(ds, "foo")
    assert isinstance(grid, Features)
    tms = morecantile.tms.get("WebMercatorQuad")
    limit = grid.vertex_count(np.arange(6)) - 1
    with config.set({"max_renderable_vertices": limit}):
        assert get_min_zoom(grid=grid, tms=tms, da=ds["foo"], style="polygons") > 0


async def test_features_single_feature_renders():
    ds = FEATURES_POLYGONS_XVEC.create().isel(feature=[0])
    ds.attrs["_xpublish_id"] = "features-single"
    query = create_query_params(
        Tile(x=0, y=0, z=0), WEBMERC_TMS, style="polygons", colorscalerange=(0, 100)
    )
    await pipeline(ds, query)


async def test_features_raster_style_rejected():
    ds = FEATURES_POLYGONS.create()
    query = create_query_params(Tile(x=0, y=0, z=0), WEBMERC_TMS, style="raster")
    with pytest.raises(ValueError, match="only supports style='polygons'"):
        await pipeline(ds, query)


def test_features_crossing_hole_oriented_and_empty():
    poly = shapely.Polygon(
        [(170, 40), (-170, 40), (-170, 60), (170, 60)],
        [[(175, 45), (178, 45), (178, 55), (175, 55)]],
    )
    ds = xr.Dataset(coords={"feature": [poly]}).xvec.set_geom_indexes("feature", crs=4326)
    ds["foo"] = ("feature", [1.0])
    ds.attrs["_xpublish_id"] = "features-crossing-hole"
    grid = guess_grid_system(ds, "foo")
    assert isinstance(grid, Features)
    (geom,) = grid.geometries
    _assert_split(geom)
    for part in shapely.get_parts(geom):
        assert all(part.exterior.is_ccw != ring.is_ccw for ring in part.interiors)

    polys = features_to_multipolygons(
        grid.geometries, transformer_from_crs(CRS(4326), CRS(4326))
    )
    gdf = spatialpandas.GeoDataFrame({"geometry": polys, "data": np.ones(1)})
    cvs = dsh.Canvas(plot_width=40, plot_height=40, x_range=(-180, 180), y_range=(40, 60))
    agg = cvs.polygons(gdf, "geometry", agg=dsh.mean("data"))
    east = dsh.Canvas(plot_width=40, plot_height=40, x_range=(170, 180), y_range=(40, 60))
    agg_east = east.polygons(gdf, "geometry", agg=dsh.mean("data"))
    assert np.isnan(agg_east.sel(x=176.5, y=50, method="nearest").item())
    assert agg_east.sel(x=172, y=50, method="nearest").item() == 1
    assert agg.sel(x=-175, y=50, method="nearest").item() == 1


def test_detect_features_with_latlon_coords():
    ds = FEATURES_POLYGONS.create()
    n = ds.sizes["feature"]
    ds = ds.assign_coords(
        lat=("feature", np.linspace(0, 50, n), {"standard_name": "latitude"}),
        lon=("feature", np.linspace(-170, 170, n), {"standard_name": "longitude"}),
    )
    ds.attrs["_xpublish_id"] = "features-latlon-coords"
    assert isinstance(guess_grid_system(ds, "foo"), Features)


def test_detect_features_cf_decode_coords_all():
    ds = FEATURES_POLYGONS.create()
    store = zarr.storage.MemoryStore()
    ds.to_zarr(store, zarr_format=3, consolidated=False)
    opened = xr.open_zarr(store, consolidated=False, decode_coords="all")
    opened.attrs["_xpublish_id"] = "features-decode-coords-all"
    grid = guess_grid_system(opened, "foo")
    expected = guess_grid_system(ds, "foo")
    assert isinstance(grid, Features) and isinstance(expected, Features)
    assert shapely.equals(grid.geometries, expected.geometries).all()
    assert grid.structural_names == expected.structural_names


async def test_features_discrete_render_keeps_flag_values():
    # Under a mean reduction the overlap averages flags 0 and 2 into flag 1.
    geoms = [shapely.box(0, 0, 20, 20), shapely.box(10, 0, 30, 20)]
    ds = xr.Dataset(coords={"feature": geoms}).xvec.set_geom_indexes("feature", crs=4326)
    ds["foo"] = (
        "feature",
        np.array([0, 2], dtype=np.uint8),
        {"flag_values": [0, 1, 2], "flag_meanings": "a b c"},
    )
    ds.attrs["_xpublish_id"] = "features-discrete-overlap"
    colormap = {"0": "#ff0000", "1": "#00ff00", "2": "#0000ff"}
    query = create_query_params(
        Tile(x=0, y=0, z=0),
        WEBMERC_TMS,
        style="polygons",
        variant="custom",
        colormap=colormap,
    )
    result = await pipeline(ds, query)
    rgba = np.asarray(Image.open(io.BytesIO(result.getvalue())).convert("RGBA"))
    colors = {tuple(c) for c in rgba[rgba[..., 3] > 0][:, :3].tolist()}
    assert colors == {(255, 0, 0), (0, 0, 255)}


@pytest.mark.parametrize("fixture", [FEATURES_POINTS, FEATURES_POINTS_XVEC])
def test_detect_point_features(fixture):
    ds = fixture.create()
    meta = guess_grid_metadata(ds)
    assert meta is not None
    assert meta.grid_cls is PointFeatures
    grid = guess_grid_system(ds, "foo")
    assert isinstance(grid, PointFeatures)
    assert grid.dim == "feature"
    assert shapely.equals(grid.geometries, np.array(POINT_GEOMS)).all()
    assert grid.counts.tolist() == [1, 1, 1, 3, 1, 1]
    assert grid.bbox == BBox(west=-95, east=179.9, south=0, north=55)
    assert set(detect_grids(ds)) == {"foo", "bar"}


def test_detect_point_features_cf_without_node_count():
    points = [shapely.Point(10, 5), shapely.Point(370, 6)]
    ds = cf_xarray.geometry.shapely_to_cf(
        xr.DataArray(np.array(points, dtype=object), dims="feature")
    )
    # cf_xarray puts the node coordinates on a dim of their own
    assert "node_count" not in ds and ds["x"].dims == ("node",)
    crs = CRS.from_epsg(4326)
    ds = ds.assign_coords(spatial_ref=xr.DataArray(0, attrs=crs.to_cf()))
    ds["geometry_container"].attrs["grid_mapping"] = "spatial_ref"
    ds["foo"] = ("feature", [1.0, 2.0])
    ds = ds.set_coords(["x", "y", "crd_x", "crd_y"])
    ds.attrs["_xpublish_id"] = "point-features-no-node-count"
    grid = guess_grid_system(ds, "foo")
    assert isinstance(grid, PointFeatures)
    assert grid.dim == "feature"
    assert shapely.get_x(grid.geometries).tolist() == [10, 10]


@pytest.mark.parametrize(
    "bbox,expected",
    [
        (BBox(west=-71, south=39, east=-69, north=41), [1, 2]),
        (BBox(west=-46, south=34, east=-44, north=36), [3]),
        (BBox(west=170, south=50, east=190, north=60), [4]),
        (BBox(west=-185, south=50, east=-170, north=60), [4]),
        (BBox(west=-30, south=-10, east=-20, north=-5), []),
    ],
)
def test_point_features_sel(bbox, expected):
    grid = guess_grid_system(FEATURES_POINTS.create(), "foo")
    (indexer,) = grid.sel(bbox=bbox)["feature"]
    assert isinstance(indexer, GeometryIndexer)
    assert indexer.indices.tolist() == expected


def test_point_features_locate_unsupported():
    grid = guess_grid_system(FEATURES_POINTS.create(), "foo")
    assert not grid.supports_locate
    with pytest.raises(PointQueryNotSupportedError):
        grid.locate(x=np.array([-95.0]), y=np.array([35.0]))


@pytest.mark.parametrize(
    "ds,style,match",
    [
        pytest.param(
            FEATURES_POINTS, "polygons", "only supports style='points'", id="points"
        ),
        pytest.param(
            FEATURES_POLYGONS, "points", "only supports style='polygons'", id="polygons"
        ),
    ],
)
async def test_point_features_style_mismatch(ds, style, match):
    query = create_query_params(Tile(x=0, y=0, z=0), WEBMERC_TMS, style=style)
    with pytest.raises(UnsupportedStyleError, match=match):
        await pipeline(ds.create(), query)


async def test_points_style_needs_point_features():
    ds = create_global_dataset(nlat=19, nlon=37)
    query = create_query_params(Tile(x=0, y=0, z=0), WEBMERC_TMS, style="points")
    with pytest.raises(UnsupportedStyleError, match="needs point features"):
        await pipeline(ds, query)


def _alpha(buffer: io.BytesIO) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGBA"))[..., 3]


def _xvec_points(points: list, values, xpublish_id: str, **attrs) -> xr.Dataset:
    ds = xr.Dataset(coords={"feature": points}).xvec.set_geom_indexes("feature", crs=4326)
    ds["foo"] = ("feature", np.asarray(values), attrs)
    ds.attrs["_xpublish_id"] = xpublish_id
    return ds


async def test_point_marker_reaches_across_tile_edges():
    ds = _xvec_points([shapely.Point(0, 0)], [1.0], "point-on-tile-corner")
    # each z1 tile holds a quarter of the marker, in the corner at (0, 0)
    corners = {(0, 0): (255, 255), (1, 0): (255, 0), (0, 1): (0, 255), (1, 1): (0, 0)}
    for (x, y), (row, col) in corners.items():
        query = create_query_params(
            Tile(x=x, y=y, z=1), WEBMERC_TMS, style="points", colorscalerange=(0, 2)
        )
        rows, cols = np.nonzero(_alpha(await pipeline(ds, query)))
        assert rows.size
        assert (np.abs(rows - row) <= 3).all() and (np.abs(cols - col) <= 3).all()


async def test_point_marker_wraps_at_antimeridian_on_world_tile():
    ds = _xvec_points([shapely.Point(179.9, 0)], [1.0], "point-by-antimeridian")
    query = create_query_params(
        Tile(x=0, y=0, z=0), WEBMERC_TMS, style="points", colorscalerange=(0, 2)
    )
    _, cols = np.nonzero(_alpha(await pipeline(ds, query)))
    assert cols.min() == 0 and cols.max() == 255


async def test_point_discrete_render_keeps_flag_values():
    # one lone point, and two in one pixel whose flags 0 and 2 would average to 1
    points = [shapely.Point(-40, 0), shapely.Point(40, 0), shapely.Point(40.01, 0)]
    ds = _xvec_points(
        points,
        np.array([0, 2, 0], dtype=np.uint8),
        "point-features-discrete",
        flag_values=[0, 1, 2],
        flag_meanings="a b c",
    )
    query = create_query_params(
        Tile(x=0, y=0, z=0),
        WEBMERC_TMS,
        style="points",
        variant="custom",
        colormap={"0": "#ff0000", "1": "#00ff00", "2": "#0000ff"},
    )
    result = await pipeline(ds, query)
    rgba = np.asarray(Image.open(io.BytesIO(result.getvalue())).convert("RGBA"))
    colors = {tuple(c) for c in rgba[rgba[..., 3] > 0][:, :3].tolist()}
    assert colors == {(255, 0, 0), (0, 0, 255)}


async def test_point_features_vertex_limit_raises():
    query = create_query_params(Tile(x=0, y=0, z=0), WEBMERC_TMS, style="points")
    with config.set({"max_renderable_vertices": 5}), pytest.raises(TileTooBigError):
        await pipeline(FEATURES_POINTS.create(), query)
