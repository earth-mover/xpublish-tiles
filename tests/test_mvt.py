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
from xpublish_tiles.render import RenderRegistry
from xpublish_tiles.render.mvt import (
    BUFFER,
    EXTENT,
    MVTRenderer,
    _field,
    encode_geometries,
    encode_layer,
    prepare_geometries,
)
from xpublish_tiles.render.polygons import PolygonsRenderer
from xpublish_tiles.testing.datasets import FEATURES_POLYGONS
from xpublish_tiles.testing.tiles import WEBMERC_TMS
from xpublish_tiles.types import TileFormat

Z0 = Tile(x=0, y=0, z=0)
VIRIDIS = mpl.colormaps["viridis"]


@pytest.fixture(scope="module")
def features() -> xr.Dataset:
    return FEATURES_POLYGONS.create().isel(time=0)


def decode(raw: bytes) -> dict:
    return mapbox_vector_tile.decode(raw, default_options={"y_coord_down": True})


async def render(ds, tile=Z0, **kwargs) -> dict:
    kwargs.setdefault("style", "polygons")
    query = create_query_params(tile, WEBMERC_TMS, format=TileFormat.MVT, **kwargs)
    return decode((await pipeline(ds, query)).getvalue())


def xvec_dataset(geoms, values, **attrs) -> xr.Dataset:
    ds = xr.Dataset(coords={"feature": geoms}).xvec.set_geom_indexes("feature", crs=4326)
    ds["foo"] = ("feature", np.asarray(values), attrs)
    return ds


def test_mvt_renderer_is_chosen_by_format():
    query = create_query_params(Z0, WEBMERC_TMS, style="polygons", format=TileFormat.MVT)
    assert isinstance(query.get_renderer(), MVTRenderer)
    assert RenderRegistry.get("polygons") is PolygonsRenderer


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
    for f, expected in zip(features, geoms, strict=True):
        assert shapely.equals(shapely.geometry.shape(f["geometry"]), expected)

    # MVT: exteriors wind positive in tile coordinates (y down), holes negative.
    def ccw(geometry: dict) -> list[list[bool]]:
        polygons = geometry["coordinates"]
        if geometry["type"] == "Polygon":
            polygons = [polygons]
        return [[shapely.LinearRing(r).is_ccw for r in p] for p in polygons]

    assert [ccw(f["geometry"]) for f in features] == [
        [[True, False]],
        [[True], [True]],
        [[True]],
    ]


def test_prepare_geometries_drops_clipped_slivers():
    edge = EXTENT + BUFFER
    crossing = shapely.box(edge - 100, 0, edge + 100, 100)
    # touches the clip edge only along a line, which clipping leaves behind
    touching = shapely.box(edge, 200, edge + 100, 300)
    (clipped,) = prepare_geometries(
        np.array([shapely.MultiPolygon([crossing, touching])])
    )
    assert shapely.equals(clipped, shapely.box(edge - 100, 0, edge, 100))


def test_encode_layer_bool_values():
    geoms = prepare_geometries(np.array([shapely.box(0, 0, 10, 10)] * 2))
    raw = encode_layer("layer", encode_geometries(geoms), np.array([True, False]))
    features = decode(_field(3, raw))["layer"]["features"]
    assert [f["properties"]["value"] for f in features] == [True, False]


async def test_features_mvt(features):
    feats = (await render(features))["foo"]["features"]
    assert [f["id"] for f in feats] == list(range(6))
    assert [f["properties"]["value"] for f in feats] == features.foo.values.tolist()

    geoms = {f["id"]: f["geometry"] for f in feats}
    assert geoms[1]["type"] == "Polygon" and len(geoms[1]["coordinates"]) == 2
    # a multipolygon, and the antimeridian crosser split in two
    assert geoms[2]["type"] == geoms[3]["type"] == "MultiPolygon"
    assert len(geoms[3]["coordinates"]) == 2


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        pytest.param(
            {"colorscalerange": (0, 100)},
            {v: mcolors.to_hex(VIRIDIS(v / 100)) for v in range(10, 70, 10)},
            id="range",
        ),
        pytest.param({}, dict.fromkeys(range(10, 70, 10)), id="no-range"),
        pytest.param(
            {
                "colorscalerange": (15, 35),
                "abovemaxcolor": "transparent",
                "belowmincolor": "red",
            },
            {
                10: "#ff0000",
                20: mcolors.to_hex(VIRIDIS(5 / 20)),
                30: mcolors.to_hex(VIRIDIS(15 / 20)),
                40: "#00000000",
                50: "#00000000",
                60: "#00000000",
            },
            id="out-of-range",
        ),
    ],
)
async def test_mvt_colors(features, kwargs, expected):
    feats = (await render(features, **kwargs))["foo"]["features"]
    colors = {f["properties"]["value"]: f["properties"].get("color") for f in feats}
    assert colors == expected


async def test_mvt_clips_to_buffered_tile(features):
    # inside the holed square (-80..-60, 30..50) but not its hole
    tile = WEBMERC_TMS.tile(-78, 48, 6)
    (feature,) = (await render(features, tile=tile))["foo"]["features"]
    assert feature["id"] == 1
    coords = np.array(feature["geometry"]["coordinates"][0])
    assert coords.min() == -BUFFER and coords.max() == EXTENT + BUFFER


async def test_mvt_empty_tile(features):
    assert await render(features, tile=Tile(x=6, y=7, z=4)) == {}


async def test_mvt_discrete_values_and_colors():
    ds = xvec_dataset(
        [shapely.box(0, 0, 20, 20), shapely.box(30, 0, 50, 20)],
        np.array([0, 2], dtype=np.uint8),
        flag_values=[0, 1, 2],
        flag_meanings="a b c",
    )
    feats = (
        await render(
            ds,
            variant="custom",
            colormap={"0": "#ff0000", "1": "#00ff00", "2": "#0000ff"},
        )
    )["foo"]["features"]
    assert [f["properties"] for f in feats] == [
        {"value": 0, "color": "#ff0000"},
        {"value": 2, "color": "#0000ff"},
    ]


async def test_mvt_drops_nan_features():
    boxes = [shapely.box(0, 0, 20, 20), shapely.box(30, 0, 50, 20)]
    feats = (await render(xvec_dataset(boxes, [1.0, np.nan])))["foo"]["features"]
    assert [(f["id"], f["properties"]["value"]) for f in feats] == [(0, 1.0)]
    assert await render(xvec_dataset(boxes, [np.nan, np.nan])) == {}


async def test_mvt_layer_per_variable(features):
    tile = await render(features, variables=("foo", "bar"))
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
async def test_mvt_unsupported_styles(features, kwargs):
    with pytest.raises(UnsupportedStyleError):
        await render(features, **kwargs)
