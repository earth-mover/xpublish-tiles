import pytest
import xpublish
from fastapi.testclient import TestClient

from xpublish_tiles.xpublish.tiles import TilesPlugin


@pytest.fixture(scope="session")
def xpublish_app(air_dataset):
    rest = xpublish.Rest({"air": air_dataset}, plugins={"tiles": TilesPlugin()})
    return rest.app


@pytest.fixture(scope="session")
def xpublish_client(xpublish_app):
    app = xpublish_app
    return TestClient(app)


def test_app_router(xpublish_client):
    response = xpublish_client.get("/datasets/air/tiles")
    assert response.status_code == 200

    data = response.json()
    assert data["title"] == "Dataset Tiles"
    assert data["description"] == "Tiles for this dataset"
    assert len(data["links"]) == 1
    assert data["links"][0]["href"] == "./WebMercatorQuad"
    assert data["links"][0]["rel"] == "item"
    assert data["links"][0]["type"] == "application/json"
    assert data["links"][0]["title"] == "WebMercatorQuad tileset"
