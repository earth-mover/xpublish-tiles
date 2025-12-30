import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from xpublish import Rest
from xpublish_tiles.xpublish.tiles import TilesPlugin


def _make_level_dataset(level: int, lat_size: int, lon_size: int) -> xr.Dataset:
    lat = np.linspace(-85, 85, lat_size)
    lon = np.linspace(-180, 180, lon_size, endpoint=False)
    data = np.full((lat_size, lon_size), fill_value=float(level), dtype=np.float32)

    ds = xr.Dataset(
        {
            "foo": (
                ("lat", "lon"),
                data,
                {"valid_min": float(data.min()), "valid_max": float(data.max())},
            )
        },
        coords={
            "lat": (
                "lat",
                lat,
                {
                    "axis": "Y",
                    "standard_name": "latitude",
                    "units": "degrees_north",
                },
            ),
            "lon": (
                "lon",
                lon,
                {
                    "axis": "X",
                    "standard_name": "longitude",
                    "units": "degrees_east",
                },
            ),
        },
        attrs={"title": f"Precomputed level {level}"},
    )

    return ds


@pytest.fixture(scope="session")
def multiscale_datatree():
    levels = {
        str(level): _make_level_dataset(level, 8 * (2**level), 16 * (2**level))
        for level in (0, 1, 2)
    }
    return xr.DataTree.from_dict(levels, name="pyramid")


@pytest.fixture(scope="session")
def multiscale_client(multiscale_datatree):
    rest = Rest(
        datatrees={"pyramid": multiscale_datatree},
        plugins={"precomputed_tiles": TilesPlugin()},
    )
    return TestClient(rest.app)


@pytest.mark.parametrize("zoom", [0, 1, 2])
def test_tilejson_uses_requested_level_metadata(multiscale_client, zoom):
    response = multiscale_client.get(
        "/datatrees/pyramid/tiles/WebMercatorQuad/tilejson.json"
        f"?variables=foo&width=256&height=256&tileMatrix={zoom}"
    )

    assert response.status_code == 200
    tilejson = response.json()

    assert tilejson["name"] == f"Precomputed level {zoom}"
    assert any("/datatrees/pyramid/tiles/WebMercatorQuad" in url for url in tilejson["tiles"])


@pytest.mark.parametrize("zoom", [0, 1, 2])
def test_tiles_resolve_precomputed_levels(multiscale_client, zoom):
    response = multiscale_client.get(
        f"/datatrees/pyramid/tiles/WebMercatorQuad/{zoom}/0/0"
        "?variables=foo&width=256&height=256"
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_missing_level_returns_404(multiscale_client):
    tilejson_response = multiscale_client.get(
        "/datatrees/pyramid/tiles/WebMercatorQuad/tilejson.json"
        "?variables=foo&width=256&height=256&tileMatrix=5"
    )
    assert tilejson_response.status_code == 404

    tile_response = multiscale_client.get(
        "/datatrees/pyramid/tiles/WebMercatorQuad/5/0/0"
        "?variables=foo&width=256&height=256"
    )
    assert tile_response.status_code == 404


def test_dataset_tiles_plugin_still_works_with_precomputed_plugin(
    multiscale_datatree, air_dataset
):
    rest = Rest(
        {"air": air_dataset},
        datatrees={"pyramid": multiscale_datatree},
        plugins={"tiles": TilesPlugin()},
    )
    client = TestClient(rest.app)

    response = client.get(
        "/datasets/air/tiles/WebMercatorQuad/tilejson.json"
        "?variables=air&width=256&height=256"
    )

    assert response.status_code == 200
