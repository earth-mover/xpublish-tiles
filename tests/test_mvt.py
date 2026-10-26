import mapbox_vector_tile
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pytest
import shapely
import xvec  # noqa: F401
from morecantile import Tile

import xarray as xr
from tests import create_query_params
from xpublish_tiles.lib import UnsupportedStyleError
from xpublish_tiles.pipeline import pipeline
from xpublish_tiles.render.mvt import (
    BUFFER,
    EXTENT,
    _field,
    encode_geometries,
    encode_layer,
    prepare_geometries,
)
from xpublish_tiles.testing.datasets import FEATURES_POLYGONS, FEATURES_POLYGONS_XVEC
from xpublish_tiles.testing.tiles import WEBMERC_TMS
from xpublish_tiles.types import TileFormat

Z0 = Tile(x=0, y=0, z=0)


def decode(raw: bytes) -> dict:
    return mapbox_vector_tile.decode(raw, default_options={"y_coord_down": True})


async def render(ds, tile=Z0, **kwargs) -> dict:
    kwargs.setdefault("style", "polygons")
    query = create_query_params(tile, WEBMERC_TMS, format=TileFormat.MVT, **kwargs)
    return decode((await pipeline(ds, query)).getvalue())


def ring_area_signs(geometry: dict) -> list[list[int]]:
    """Shoelace sign of each ring in tile coordinates, per polygon."""
    polygons = geometry["coordinates"]
    if geometry["type"] == "Polygon":
        polygons = [polygons]
    return [
        [int(np.sign(shapely.LinearRing(ring).is_ccw - 0.5)) for ring in polygon]
        for polygon in polygons
    ]


def xvec_dataset(geoms, values, **attrs) -> xr.Dataset:
    ds = xr.Dataset(coords={"feature": geoms}).xvec.set_geom_indexes("feature", crs=4326)
    ds["foo"] = ("feature", np.asarray(values), attrs)
    return ds


def test_encode_layer_roundtrip():
    shell = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
    hole = [(20, 20), (20, 80), (80, 80), (80, 20), (20, 20)]
    geoms = np.array(
        [
            shapely.Polygon(shell, [hole]),
            shapely.MultiPolygon(
                [shapely.box(200, 200, 300, 300), shapely.box(400, 400, 500, 450)]
            ),
            shapely.box(1000, 2000, 1010, 2003),
        ]
    )
    geoms = prepare_geometries(geoms)
    raw = encode_layer(
        "layer",
        encode_geometries(geoms),
        np.array([1.5, -2.0, 3.0]),
        ids=np.array([7, 8, 9]),
    )
    layer = decode(_field(3, raw))["layer"]
    assert layer["extent"] == EXTENT
    features = layer["features"]
    assert [f["id"] for f in features] == [7, 8, 9]
    assert [f["properties"] for f in features] == [
        {"value": 1.5},
        {"value": -2.0},
        {"value": 3.0},
    ]
    decoded = [shapely.geometry.shape(f["geometry"]) for f in features]
    for got, expected in zip(decoded, geoms, strict=True):
        assert shapely.equals(got, expected)
    assert len(decoded[0].interiors) == 1
    # MVT: exteriors wind positive in tile coordinates (y down), holes negative.
    assert [ring_area_signs(f["geometry"]) for f in features] == [
        [[1, -1]],
        [[1], [1]],
        [[1]],
    ]


@pytest.mark.parametrize(
    "dataset", [FEATURES_POLYGONS, FEATURES_POLYGONS_XVEC], ids=lambda d: d.name
)
async def test_features_mvt(dataset):
    ds = dataset.create().isel(time=0)
    layer = (await render(ds, colorscalerange=(0, 100)))["foo"]
    features = layer["features"]
    assert [f["id"] for f in features] == list(range(6))

    values = [f["properties"]["value"] for f in features]
    npt_values = ds.foo.values.tolist()
    assert values == npt_values

    viridis = mpl.colormaps["viridis"]
    assert [f["properties"]["color"] for f in features] == [
        mcolors.to_hex(viridis(v / 100)) for v in npt_values
    ]

    geoms = {f["id"]: f["geometry"] for f in features}
    for geometry in geoms.values():
        for signs in ring_area_signs(geometry):
            assert signs == [1] + [-1] * (len(signs) - 1)
    assert geoms[1]["type"] == "Polygon" and len(geoms[1]["coordinates"]) == 2
    # a multipolygon, and the antimeridian crosser split in two
    assert geoms[2]["type"] == geoms[3]["type"] == "MultiPolygon"
    assert len(geoms[3]["coordinates"]) == 2


async def test_mvt_clips_to_buffered_tile():
    ds = FEATURES_POLYGONS.create().isel(time=0)
    # inside the holed square (-80..-60, 30..50) but not its hole
    tile = WEBMERC_TMS.tile(-78, 48, 6)
    (feature,) = (await render(ds, tile=tile))["foo"]["features"]
    assert feature["id"] == 1
    coords = np.array(feature["geometry"]["coordinates"][0])
    assert coords.min() == -BUFFER and coords.max() == EXTENT + BUFFER


async def test_mvt_empty_tile():
    ds = FEATURES_POLYGONS.create().isel(time=0)
    query = create_query_params(
        Tile(x=6, y=7, z=4), WEBMERC_TMS, style="polygons", format=TileFormat.MVT
    )
    assert (await pipeline(ds, query)).getvalue() == b""


async def test_mvt_without_colorscale_has_no_color():
    ds = FEATURES_POLYGONS.create().isel(time=0)
    features = (await render(ds))["foo"]["features"]
    assert all(set(f["properties"]) == {"value"} for f in features)


async def test_mvt_out_of_range_colors():
    ds = FEATURES_POLYGONS.create().isel(time=0)
    features = (
        await render(
            ds,
            colorscalerange=(0, 35),
            abovemaxcolor="transparent",
            belowmincolor="red",
        )
    )["foo"]["features"]
    colors = {f["properties"]["value"]: f["properties"]["color"] for f in features}
    assert colors[40] == colors[60] == "#00000000"
    assert colors[10] == mcolors.to_hex(mpl.colormaps["viridis"](10 / 35))


async def test_mvt_discrete_values_and_colors():
    ds = xvec_dataset(
        [shapely.box(0, 0, 20, 20), shapely.box(30, 0, 50, 20)],
        np.array([0, 2], dtype=np.uint8),
        flag_values=[0, 1, 2],
        flag_meanings="a b c",
    )
    features = (
        await render(
            ds,
            variant="custom",
            colormap={"0": "#ff0000", "1": "#00ff00", "2": "#0000ff"},
        )
    )["foo"]["features"]
    assert [f["properties"] for f in features] == [
        {"value": 0, "color": "#ff0000"},
        {"value": 2, "color": "#0000ff"},
    ]


async def test_mvt_drops_nan_features():
    ds = xvec_dataset(
        [shapely.box(0, 0, 20, 20), shapely.box(30, 0, 50, 20)], [1.0, np.nan]
    )
    features = (await render(ds))["foo"]["features"]
    assert [(f["id"], f["properties"]["value"]) for f in features] == [(0, 1.0)]


async def test_mvt_layer_per_variable():
    ds = FEATURES_POLYGONS.create().isel(time=0)
    tile = await render(ds, variables=("foo", "bar"))
    assert set(tile) == {"foo", "bar"}
    assert [f["properties"]["value"] for f in tile["bar"]["features"]] == list(range(6))


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"style": "raster"}, id="raster"),
        pytest.param({"variant": "rgb"}, id="rgb"),
        pytest.param({"edgecolor": "red"}, id="edgecolor"),
    ],
)
async def test_mvt_unsupported_styles(kwargs):
    ds = FEATURES_POLYGONS.create().isel(time=0)
    with pytest.raises(UnsupportedStyleError):
        await render(ds, **kwargs)
