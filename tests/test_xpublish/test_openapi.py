import warnings

import pytest
import xpublish
from fastapi.testclient import TestClient
from pydantic.json_schema import PydanticJsonSchemaWarning

from xpublish_tiles.xpublish.tiles import TilesPlugin
from xpublish_tiles.xpublish.wms import WMSPlugin


@pytest.fixture(scope="session")
def openapi_client(air_dataset):
    rest = xpublish.Rest(
        {"air": air_dataset},
        plugins={"tiles": TilesPlugin(), "wms": WMSPlugin()},
    )
    return TestClient(rest.app)


def test_openapi_schema_generation(openapi_client):
    """The full OpenAPI schema must be generatable with all plugins registered.

    Guards against query model fields whose types pydantic cannot render to
    JSON Schema (e.g. raw pyproj.CRS), which would 500 every /openapi.json
    and /docs request.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", PydanticJsonSchemaWarning)
        response = openapi_client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    assert any("/wms" in path for path in schema["paths"])
    assert any("/tiles" in path for path in schema["paths"])

    for model in ("WMSGetMapQuery", "WMSGetFeatureInfoQuery"):
        crs = schema["components"]["schemas"][model]["properties"]["crs"]
        assert crs["type"] == "string"
        assert crs["default"] == "EPSG:4326"
