import io
import sys

import datashader as dsh
import morecantile
import numpy as np
import numpy.testing as npt
import pytest
import shapely
import spatialpandas
import xvec  # noqa: F401
from morecantile import Tile
from PIL import Image
from pyproj import CRS
from pyproj.aoi import BBox

import xarray as xr
from tests import create_query_params
from xpublish_tiles import config
from xpublish_tiles.grids import (
    Features,
    GeometryIndexer,
    Rectilinear,
    detect_grids,
    guess_grid_metadata,
    guess_grid_system,
)
from xpublish_tiles.lib import (
    TileTooBigError,
    UnsupportedGridError,
    features_to_multipolygons,
    unwrap_seam_vertices,
)
from xpublish_tiles.pipeline import bbox_overlap, pipeline
from xpublish_tiles.projections import transformer_from_crs
from xpublish_tiles.testing.datasets import (
    FEATURE_GEOMS,
    FEATURES_POLYGONS,
    FEATURES_POLYGONS_XVEC,
    GLOBAL_6KM,
)
from xpublish_tiles.testing.tiles import WEBMERC_TMS
from xpublish_tiles.tiles_lib import get_min_zoom

W = 2 * 20037508.342789244
T = transformer_from_crs(CRS(4326), CRS(3857))


@pytest.mark.parametrize("fixture", [FEATURES_POLYGONS, FEATURES_POLYGONS_XVEC])
def test_detect_features(fixture):
    ds = fixture.create()
    meta = guess_grid_metadata(ds)
    assert meta is not None
    assert meta.grid_cls is Features
    grid = guess_grid_system(ds, "foo")
    assert isinstance(grid, Features)
    assert grid.dim == "feature"
    assert grid.crossing.tolist() == [False, False, False, True, False, False]
    assert grid.bbox == BBox(west=-180, east=180, south=0, north=60)
    assert not grid.is_structural_variable(ds["foo"])


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
    assert grid.crossing.tolist() == [False, True]


def test_features_point_geometry_unsupported():
    ds = xr.Dataset(coords={"feature": [shapely.Point(0, 0)]}).xvec.set_geom_indexes(
        "feature", crs=4326
    )
    ds["foo"] = ("feature", [1.0])
    ds.attrs["_xpublish_id"] = "features-point"
    with pytest.raises(UnsupportedGridError):
        guess_grid_metadata(ds)
    with pytest.raises(UnsupportedGridError, match="non-polygonal geometries"):
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


def test_features_cf_without_shapely(monkeypatch):
    ds = _features_and_gridded("features-no-shapely")
    _block_shapely(monkeypatch)
    assert isinstance(guess_grid_system(ds, "t2m"), Rectilinear)
    with pytest.raises(ImportError, match=r"xpublish-tiles\[vector\]"):
        guess_grid_system(ds, "foo")


def test_features_cf_point_container_without_shapely(monkeypatch):
    ds = _features_and_gridded("features-cf-point")
    ds["geometry_container"].attrs["geometry_type"] = "point"
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
    assert grid.crossing.tolist() == [False]
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


def test_features_to_multipolygons_no_crossing():
    g = np.array([shapely.box(0, 0, 10, 10)], dtype=object)
    polys, src = features_to_multipolygons(g, np.array([False]), T)
    assert src.tolist() == [0] and len(polys) == 1


def test_features_to_multipolygons_crossing_copies():
    g = np.array([shapely.box(175, 50, 185, 60)], dtype=object)  # build-time shifted form
    polys, src = features_to_multipolygons(g, np.array([True]), T)
    assert src.tolist() == [0, 0, 0]
    b = polys.bounds  # (3, 4): xmin, ymin, xmax, ymax
    assert (b[:, 2] - b[:, 0] < W / 2).all()
    assert b[0, 0] > 0
    npt.assert_allclose(b[1, [0, 2]], b[0, [0, 2]] - W)
    npt.assert_allclose(b[2, [0, 2]], b[0, [0, 2]] + W)


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
    assert grid.crossing.tolist() == [True]
    (geom,) = grid.geometries
    assert geom.exterior.is_ccw != geom.interiors[0].is_ccw

    polys, src = features_to_multipolygons(
        grid.geometries, grid.crossing, transformer_from_crs(CRS(4326), CRS(4326))
    )
    gdf = spatialpandas.GeoDataFrame({"geometry": polys, "data": np.ones(src.size)})
    cvs = dsh.Canvas(plot_width=40, plot_height=40, x_range=(170, 190), y_range=(40, 60))
    agg = cvs.polygons(gdf, "geometry", agg=dsh.mean("data"))
    assert np.isnan(agg.sel(x=176.5, y=50, method="nearest").item())
    assert agg.sel(x=172, y=50, method="nearest").item() == 1
