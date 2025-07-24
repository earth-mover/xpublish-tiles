import pytest
import xpublish
from fastapi.testclient import TestClient

from xpublish_tiles.xpublish.wms import WMSPlugin


@pytest.fixture(scope="session")
def xpublish_app(air_dataset):
    rest = xpublish.Rest({"air": air_dataset}, plugins={"wms": WMSPlugin()})
    return rest.app


@pytest.fixture(scope="session")
def xpublish_client(xpublish_app):
    app = xpublish_app
    return TestClient(app)


def test_app_router(xpublish_client):
    response = xpublish_client.get("/datasets/air/wms")
    assert response.status_code == 422
    assert response.json() == {"message": "Hello, WMS!"}
