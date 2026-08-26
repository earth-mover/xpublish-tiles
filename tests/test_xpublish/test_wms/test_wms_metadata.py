"""Snapshot and smoke tests for WMS endpoints across diverse grid types"""

import xml.etree.ElementTree as ET

import morecantile
import pytest
import xpublish
from fastapi.testclient import TestClient
from syrupy.extensions.json import JSONSnapshotExtension

from xpublish_tiles.testing.datasets import (
    CUBED_SPHERE,
    ERA5,
    FVCOM,
    GEOSTATIONARY,
    GEOZARR_MULTISCALE,
    GLOBAL_HEALPIX_L3,
    HRRR,
    IFS,
    NATIVE_AT_ROOT_MULTISCALE,
    RADAR,
    REGIONAL_HEALPIX_NA,
)
from xpublish_tiles.xpublish.wms import WMSPlugin

WEBMERC_TMS = morecantile.tms.get("WebMercatorQuad")


def _make_client(fixture) -> TestClient:
    # Publish under the fixture's name so each parametrized run gets a unique
    # ``_xpublish_id`` — otherwise the grid cache (keyed on xpublish_id +
    # dim names) returns the previous fixture's grid for this one.
    ds = fixture.create()
    rest = xpublish.Rest({fixture.name: ds}, plugins={"wms": WMSPlugin()})
    return TestClient(rest.app)


def _normalize_for_snapshot(obj):
    """Normalize a GetCapabilities response so snapshots are stable across platforms.

    Bounding-box values come from pyproj reprojections that diverge between
    PROJ versions on macOS vs Linux, style ordering follows entry-point load
    order, and dimension value lists are huge — elide/sort them and keep the
    rest of the structure.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("bounding_box", "ex_geographic_bounding_box"):
                out[k] = "<elided>"
            elif k == "styles" and isinstance(v, list):
                out[k] = sorted(s["name"] for s in v)
            elif k == "values" and isinstance(v, str):
                out[k] = f"<{len(v.split(','))} values>" if v else v
            else:
                out[k] = _normalize_for_snapshot(v)
        return out
    if isinstance(obj, list):
        return [_normalize_for_snapshot(x) for x in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


@pytest.mark.parametrize(
    "fixture",
    [
        pytest.param(ERA5, id="era5"),
        pytest.param(HRRR, id="hrrr"),
        pytest.param(IFS, id="ifs"),
        pytest.param(GLOBAL_HEALPIX_L3, id="global_healpix_l3"),
        pytest.param(REGIONAL_HEALPIX_NA, id="regional_healpix_na"),
        pytest.param(CUBED_SPHERE, id="cubed_sphere"),
        pytest.param(GEOSTATIONARY, id="geostationary"),
        pytest.param(GEOZARR_MULTISCALE, id="geozarr_multiscale"),
        pytest.param(NATIVE_AT_ROOT_MULTISCALE, id="native_at_root_multiscale"),
        pytest.param(FVCOM, id="fvcom"),
        pytest.param(RADAR, id="radar"),
    ],
)
def test_wms_capabilities_snapshot(fixture, snapshot):
    """Snapshot the GetCapabilities response across diverse grid types."""
    client = _make_client(fixture)
    params = {"service": "WMS", "version": "1.3.0", "request": "GetCapabilities"}

    xml_response = client.get(f"/datasets/{fixture.name}/wms", params=params)
    assert xml_response.status_code == 200
    root = ET.fromstring(xml_response.content)
    assert root.tag == "{http://www.opengis.net/wms}WMS_Capabilities"

    json_response = client.get(
        f"/datasets/{fixture.name}/wms", params={**params, "format": "json"}
    )
    assert json_response.status_code == 200
    assert _normalize_for_snapshot(json_response.json()) == snapshot.use_extension(
        JSONSnapshotExtension
    )


@pytest.mark.parametrize(
    "fixture,tile",
    [
        pytest.param(ERA5, morecantile.Tile(x=2, y=1, z=2), id="era5"),
        pytest.param(HRRR, morecantile.Tile(x=0, y=1, z=2), id="hrrr"),
        pytest.param(IFS, morecantile.Tile(x=2, y=1, z=2), id="ifs"),
        pytest.param(
            GLOBAL_HEALPIX_L3, morecantile.Tile(x=0, y=0, z=1), id="global_healpix_l3"
        ),
        pytest.param(
            REGIONAL_HEALPIX_NA,
            morecantile.Tile(x=1, y=1, z=2),
            id="regional_healpix_na",
        ),
        pytest.param(CUBED_SPHERE, morecantile.Tile(x=1, y=1, z=2), id="cubed_sphere"),
        pytest.param(GEOSTATIONARY, morecantile.Tile(x=1, y=2, z=2), id="geostationary"),
        pytest.param(
            GEOZARR_MULTISCALE, morecantile.Tile(x=15, y=10, z=5), id="geozarr_multiscale"
        ),
        pytest.param(
            NATIVE_AT_ROOT_MULTISCALE,
            morecantile.Tile(x=15, y=10, z=5),
            id="native_at_root_multiscale",
        ),
        pytest.param(FVCOM, morecantile.Tile(x=160, y=184, z=9), id="fvcom"),
        pytest.param(RADAR, morecantile.Tile(x=4, y=5, z=4), id="radar"),
    ],
)
def test_wms_getmap_across_grids(fixture, tile):
    """GetMap renders every advertised grid type with its advertised styles."""
    client = _make_client(fixture)

    capabilities = client.get(
        f"/datasets/{fixture.name}/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetCapabilities",
            "format": "json",
        },
    )
    assert capabilities.status_code == 200
    layers = capabilities.json()["capability"]["layer"]["layers"]
    assert len(layers) > 0
    layer = layers[0]
    style_names = sorted(style["name"] for style in layer["styles"])
    style = next(
        (name for name in style_names if name == "raster/default"), style_names[0]
    )

    bounds = WEBMERC_TMS.xy_bounds(tile)
    response = client.get(
        f"/datasets/{fixture.name}/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetMap",
            "layers": layer["name"],
            "styles": style,
            "crs": "EPSG:3857",
            "bbox": f"{bounds.left},{bounds.bottom},{bounds.right},{bounds.top}",
            "width": 256,
            "height": 256,
        },
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
