import xml.etree.ElementTree as ET

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


def test_get_capabilities_xml(xpublish_client):
    """Test GetCapabilities request returns valid XML by default."""
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={"service": "WMS", "version": "1.3.0", "request": "GetCapabilities"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/xml; charset=utf-8"

    # Parse XML to ensure it's valid
    root = ET.fromstring(response.content)
    assert root.tag.endswith("WMS_Capabilities")
    assert root.get("version") == "1.3.0"

    # Check for required elements
    service = root.find(".//{http://www.opengis.net/wms}Service")
    assert service is not None

    capability = root.find(".//{http://www.opengis.net/wms}Capability")
    assert capability is not None

    # Check for layers
    layers = root.findall(".//{http://www.opengis.net/wms}Layer")
    assert len(layers) > 0


def test_get_capabilities_json(xpublish_client):
    """Test GetCapabilities request returns JSON when requested."""
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetCapabilities",
            "format": "json",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    # Parse JSON to ensure it's valid
    data = response.json()
    assert "version" in data
    assert data["version"] == "1.3.0"
    assert "service" in data
    assert "capability" in data

    # Check service information
    service = data["service"]
    assert service["name"] == "WMS"
    assert "title" in service

    # Check capability information
    capability = data["capability"]
    assert "request" in capability
    assert "layer" in capability


def test_get_capabilities_content_negotiation(xpublish_client):
    """Test content negotiation via Accept header."""
    # Test JSON via Accept header
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={"service": "WMS", "version": "1.3.0", "request": "GetCapabilities"},
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    # Test XML via Accept header (should be default anyway)
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={"service": "WMS", "version": "1.3.0", "request": "GetCapabilities"},
        headers={"Accept": "application/xml"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/xml; charset=utf-8"


def test_get_capabilities_layers(xpublish_client):
    """Test that GetCapabilities includes dataset layers."""
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetCapabilities",
            "format": "json",
        },
    )
    assert response.status_code == 200

    data = response.json()
    root_layer = data["capability"]["layer"]

    # Should have child layers for data variables
    assert "layers" in root_layer
    child_layers = root_layer["layers"]
    assert len(child_layers) > 0

    # Check that air variable is included as a layer
    layer_names = [layer["name"] for layer in child_layers if "name" in layer]
    assert "air" in layer_names

    # Check layer properties
    air_layer = next(layer for layer in child_layers if layer.get("name") == "air")
    assert "title" in air_layer
    assert "crs" in air_layer
    assert "EPSG:4326" in air_layer["crs"]
    assert "EPSG:3857" in air_layer["crs"]


def test_get_capabilities_dimensions(xpublish_client):
    """Test that GetCapabilities includes dataset dimensions."""
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetCapabilities",
            "format": "json",
        },
    )
    assert response.status_code == 200

    data = response.json()
    root_layer = data["capability"]["layer"]
    child_layers = root_layer["layers"]

    # Find the air layer
    air_layer = next(layer for layer in child_layers if layer.get("name") == "air")

    # Check for dimensions
    if air_layer.get("dimensions"):
        dimensions = air_layer["dimensions"]
        dimension_names = [dim["name"] for dim in dimensions]

        # The air dataset should have time dimension
        assert "time" in dimension_names

        # Check time dimension properties
        time_dim = next(dim for dim in dimensions if dim["name"] == "time")
        assert "units" in time_dim
        assert "values" in time_dim


def test_app_router(xpublish_client):
    """Test basic WMS routing functionality."""
    # Test GetCapabilities request (now returns actual capabilities)
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={"service": "WMS", "version": "1.3.0", "request": "GetCapabilities"},
    )
    assert response.status_code == 200
    # Should return XML by default
    assert "xml" in response.headers["content-type"]

    # Test GetMap request
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetMap",
            "layers": "air",
            "styles": "raster/magma",
            "crs": "EPSG:3857",
            "bbox": "-8766409.899970, 5009377.085697, -7514065.628546, 6261721.357122",
            "width": 256,
            "height": 256,
            "time": "2013-01-01T00:00:00",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    # Test invalid request (no parameters)
    response = xpublish_client.get("/datasets/air/wms")
    assert response.status_code == 422
