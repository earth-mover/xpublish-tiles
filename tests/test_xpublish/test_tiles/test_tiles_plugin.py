import numpy as np
import pytest
import xpublish
from fastapi.testclient import TestClient

import xarray as xr
from xpublish_tiles.xpublish.tiles import TilesPlugin


@pytest.fixture(scope="session")
def xpublish_app(air_dataset):
    rest = xpublish.Rest({"air": air_dataset}, plugins={"tiles": TilesPlugin()})
    return rest.app


@pytest.fixture(scope="session")
def xpublish_client(xpublish_app):
    app = xpublish_app
    return TestClient(app)


def test_tilesets_list_endpoint(xpublish_client):
    """Test the enhanced tilesets list endpoint at /tiles/"""
    response = xpublish_client.get("/datasets/air/tiles/")
    assert response.status_code == 200

    data = response.json()
    assert "tilesets" in data
    assert len(data["tilesets"]) >= 1

    # Check the first tileset
    tileset = data["tilesets"][0]
    assert "title" in tileset
    assert "crs" in tileset
    assert "dataType" in tileset
    assert tileset["dataType"] == "map"
    assert "links" in tileset
    assert len(tileset["links"]) >= 2  # self and tiling-scheme links

    # Check for enhanced fields
    assert "tileMatrixSetURI" in tileset
    assert "tileMatrixSetLimits" in tileset
    assert isinstance(tileset["tileMatrixSetLimits"], list)
    assert len(tileset["tileMatrixSetLimits"]) > 0

    # Check tile matrix set limits structure
    limit = tileset["tileMatrixSetLimits"][0]
    assert "tileMatrix" in limit
    assert "minTileRow" in limit
    assert "maxTileRow" in limit
    assert "minTileCol" in limit
    assert "maxTileCol" in limit

    # Check layers if present
    if tileset.get("layers"):
        layer = tileset["layers"][0]
        assert "id" in layer
        assert "dataType" in layer
        assert "links" in layer


def test_tilesets_list_with_metadata():
    """Test that dataset metadata is properly included in the tilesets response"""
    # Create a dataset with rich metadata
    data = xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.random.randn(90, 180),
                dims=["lat", "lon"],
                coords={
                    "lat": np.linspace(-90, 90, 90),
                    "lon": np.linspace(-180, 180, 180),
                },
                attrs={
                    "long_name": "Surface Temperature",
                    "description": "Global surface temperature data",
                    "units": "degC",
                },
            )
        },
        attrs={
            "title": "Global Climate Data",
            "description": "Sample global climate dataset",
            "keywords": "climate, temperature, global",
            "attribution": "Test Data Corporation",
            "license": "CC-BY-4.0",
            "version": "1.0.0",
            "contact": "data@example.com",
        },
    )

    # Create app with the metadata-rich dataset
    rest = xpublish.Rest({"climate": data}, plugins={"tiles": TilesPlugin()})
    client = TestClient(rest.app)

    # Test the endpoint
    response = client.get("/datasets/climate/tiles/")
    assert response.status_code == 200

    data = response.json()
    tileset = data["tilesets"][0]

    # Check that metadata fields are populated
    assert tileset["title"] == "Global Climate Data - WebMercatorQuad"
    assert tileset["description"] == "Sample global climate dataset"
    assert tileset["keywords"] == ["climate", "temperature", "global"]
    assert tileset["attribution"] == "Test Data Corporation"
    assert tileset["license"] == "CC-BY-4.0"
    assert tileset["version"] == "1.0.0"
    assert tileset["pointOfContact"] == "data@example.com"
    assert tileset["mediaTypes"] == ["image/png", "image/jpeg"]

    # Check bounding box
    assert "boundingBox" in tileset
    bbox = tileset["boundingBox"]
    assert bbox["lowerLeft"] == [-180.0, -90.0]
    assert bbox["upperRight"] == [180.0, 90.0]
    assert "crs" in bbox

    # Check layers
    assert "layers" in tileset
    assert len(tileset["layers"]) == 1
    layer = tileset["layers"][0]
    assert layer["id"] == "temperature"
    assert layer["title"] == "Surface Temperature"
    assert layer["description"] == "Global surface temperature data"
