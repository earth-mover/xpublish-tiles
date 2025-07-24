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
    # Test GetCapabilities request
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={"service": "WMS", "version": "1.3.0", "request": "GetCapabilities"},
    )
    assert response.status_code == 200
    assert response.json() == {"message": "GetCapabilities"}

    # Test GetMap request
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetMap",
            "layers": "air",
            "styles": "raster/default",
            "crs": "EPSG:3857",
            "bbox": "0,0,1,1",
            "width": 256,
            "height": 256,
            "autoscale": "true",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"message": "GetMap"}

    # Test invalid request (no parameters)
    response = xpublish_client.get("/datasets/air/wms")
    assert response.status_code == 422
