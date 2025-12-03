import io
import json
import urllib.parse

import numpy as np
import pandas as pd
import pytest
import xpublish
from fastapi.testclient import TestClient
from PIL import Image

import xarray as xr
from xpublish_tiles.testing.datasets import EU3035, IFS, PARA_HIRES, REDGAUSS_N320
from xpublish_tiles.xpublish.tiles import TilesPlugin
from xpublish_tiles.xpublish.tiles.tile_matrix import extract_dimension_extents

CUSTOM_COLORMAP = urllib.parse.quote(json.dumps({"0": "#000000", "255": "#ffffff"}))


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
    assert tileset["dataType"] in ["map", "vector", "coverage"]
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
    assert limit["minTileRow"] == 0
    assert "maxTileRow" in limit
    assert limit["maxTileRow"] == 359
    assert "minTileCol" in limit
    assert limit["minTileCol"] == 0
    assert "maxTileCol" in limit
    assert limit["maxTileCol"] == 179

    # Check layers if present
    if tileset.get("layers"):
        layer = tileset["layers"][0]
        assert "id" in layer
        assert "dataType" in layer
        assert "links" in layer


def test_tilesets_list_with_metadata():
    """Test that dataset metadata is properly included in the tilesets response"""
    # Create a dataset with rich metadata including time dimension
    import pandas as pd

    time_coords = pd.date_range("2020-01-01", periods=12, freq="MS")

    data = xr.Dataset(
        {
            "scalar": ((), 0, {"foo": "bar"}),
            "temperature": xr.DataArray(
                np.random.randn(12, 90, 180),
                dims=["time", "lat", "lon"],
                coords={
                    "time": (
                        ["time"],
                        time_coords,
                        {"axis": "T", "standard_name": "time"},
                    ),
                    "lat": (
                        ["lat"],
                        np.linspace(-90, 90, 90),
                        {
                            "axis": "Y",
                            "standard_name": "latitude",
                            "units": "degrees_north",
                        },
                    ),
                    "lon": (
                        ["lon"],
                        np.linspace(-180, 180, 180),
                        {
                            "axis": "X",
                            "standard_name": "longitude",
                            "units": "degrees_east",
                        },
                    ),
                },
                attrs={
                    "long_name": "Surface Temperature",
                    "description": "Global surface temperature data",
                    "units": "degC",
                },
            ),
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
    # The title should contain the dataset title and a TMS name
    assert tileset["title"].startswith("Global Climate Data - ")
    assert " - " in tileset["title"]  # Should have format "Dataset Title - TMS_ID"
    assert tileset["description"] == "Sample global climate dataset"
    assert tileset["keywords"] == ["climate", "temperature", "global"]
    assert tileset["attribution"] == "Test Data Corporation"
    assert tileset["license"] == "CC-BY-4.0"
    assert tileset["version"] == "1.0.0"
    assert tileset["pointOfContact"] == "data@example.com"
    assert tileset["mediaTypes"] == ["image/png", "image/jpeg"]

    # Check layers
    assert "layers" in tileset
    assert len(tileset["layers"]) == 1
    layer = tileset["layers"][0]
    assert layer["id"] == "temperature"
    assert layer["title"] == "Surface Temperature"
    assert layer["description"] == "Global surface temperature data"

    # Check that dimensions are no longer in layers (moved to tileset level)
    assert "dimensions" not in layer or layer["dimensions"] is None

    # Check bounding box
    assert "boundingBox" in layer
    bbox = layer["boundingBox"]
    assert bbox["lowerLeft"] == [-180.0, -90.0]
    assert bbox["upperRight"] == [180.0, 90.0]
    assert "crs" in bbox

    # Check that extents are now in the layer
    assert "extents" in layer
    assert layer["extents"] is not None
    assert "time" in layer["extents"]

    time_extent = layer["extents"]["time"]
    assert "interval" in time_extent
    assert len(time_extent["interval"]) == 2
    assert time_extent["interval"][0] == "2020-01-01T00:00:00"
    assert time_extent["interval"][1] == "2020-12-01T00:00:00"

    # Test the tileset metadata endpoint - extents should no longer be at tileset level
    metadata_response = client.get("/datasets/climate/tiles/WebMercatorQuad")
    assert metadata_response.status_code == 200

    metadata = metadata_response.json()
    assert "extents" not in metadata


def test_one_dimensional_dataset():
    ds = REDGAUSS_N320.create().isel(point=slice(2000))
    rest = xpublish.Rest(
        {"n320": ds},
        plugins={"tiles": TilesPlugin()},
    )
    client = TestClient(rest.app)

    response = client.get("/datasets/n320/tiles/")
    assert response.status_code == 200
    response_data = response.json()
    tileset = next(iter(response_data["tilesets"]))
    assert "layers" in tileset
    assert len(tileset["layers"]) == 1

    layer = next(iter(tileset["layers"]))
    assert "extents" in layer
    assert len(layer["extents"]) == 0

    response = client.get(
        "/datasets/n320/tiles/WebMercatorQuad/tilejson.json"
        "?variables=foo&style=raster/custom&width=512&height=512&colorscalerange=-3,3"
    )
    assert response.status_code == 200
    tilejson = response.json()

    # Zoom levels should be valid
    assert tilejson["minzoom"] == 0
    assert tilejson["maxzoom"] == 24

    rest = xpublish.Rest(
        {
            "n320": ds.expand_dims(
                {"time": pd.date_range("2001-01-01", periods=5, freq="D")}
            )
        },
        plugins={"tiles": TilesPlugin()},
    )
    client = TestClient(rest.app)

    response = client.get("/datasets/n320/tiles/")
    assert response.status_code == 200
    response_data = response.json()
    tileset = next(iter(response_data["tilesets"]))
    assert "layers" in tileset
    assert len(tileset["layers"]) == 1

    layer = next(iter(tileset["layers"]))
    assert "extents" in layer
    assert layer["extents"] == {
        "time": {
            "default": "2001-01-05T00:00:00",
            "interval": ["2001-01-01T00:00:00", "2001-01-05T00:00:00"],
        }
    }

    response = client.get(
        "/datasets/n320/tiles/WebMercatorQuad/tilejson.json"
        "?variables=foo&style=raster/custom&width=512&height=512&colorscalerange=-3,3"
    )
    assert response.status_code == 200
    tilejson = response.json()

    # Zoom levels should be valid
    assert tilejson["minzoom"] == 0
    assert tilejson["maxzoom"] == 24


def test_multi_dimensional_dataset():
    """Test dataset with multiple dimension types (time, elevation, custom)"""
    import pandas as pd

    # Create a dataset with multiple dimensions
    time_coords = pd.date_range("2020-01-01", periods=6, freq="MS")
    elevation_coords = [0, 100, 500, 1000, 2000]
    scenario_coords = ["RCP45", "RCP85", "Historical"]

    data = xr.Dataset(
        {
            "scalar": ((), 0, {"foo": "bar"}),
            "temperature": xr.DataArray(
                np.random.randn(6, 5, 3, 90, 180),
                dims=["time", "elevation", "scenario", "lat", "lon"],
                coords={
                    "time": (
                        ["time"],
                        time_coords,
                        {"axis": "T", "standard_name": "time"},
                    ),
                    "elevation": (
                        ["elevation"],
                        elevation_coords,
                        {
                            "units": "meters",
                            "long_name": "Elevation above sea level",
                            "axis": "Z",
                            "positive": "up",
                        },
                    ),
                    "scenario": (
                        ["scenario"],
                        scenario_coords,
                        {"long_name": "Climate scenario"},
                    ),
                    "lat": (
                        ["lat"],
                        np.linspace(-90, 90, 90),
                        {
                            "axis": "Y",
                            "standard_name": "latitude",
                            "units": "degrees_north",
                        },
                    ),
                    "lon": (
                        ["lon"],
                        np.linspace(-180, 180, 180),
                        {
                            "axis": "X",
                            "standard_name": "longitude",
                            "units": "degrees_east",
                        },
                    ),
                },
                attrs={
                    "long_name": "Air Temperature",
                    "description": "Multi-dimensional temperature data",
                    "units": "degC",
                },
            ),
        },
        attrs={
            "title": "Multi-dimensional Climate Data",
            "description": "Climate dataset with multiple dimensions",
        },
    )

    # Create app with the multi-dimensional dataset
    rest = xpublish.Rest({"climate": data}, plugins={"tiles": TilesPlugin()})
    client = TestClient(rest.app)

    # Test the endpoint
    response = client.get("/datasets/climate/tiles/")
    assert response.status_code == 200

    response_data = response.json()
    tileset = response_data["tilesets"][0]
    layer = tileset["layers"][0]

    # Check that dimensions are no longer in layers (moved to tileset level)
    assert "dimensions" not in layer or layer["dimensions"] is None

    # Check that extents are now in the layer
    assert "extents" in layer
    assert layer["extents"] is not None
    assert len(layer["extents"]) == 3  # time, elevation, scenario

    # Check time extent
    assert "time" in layer["extents"]
    time_extent = layer["extents"]["time"]
    assert "interval" in time_extent
    assert len(time_extent["interval"]) == 2

    # Test the tileset metadata endpoint - extents should no longer be at tileset level
    metadata_response = client.get("/datasets/climate/tiles/WebMercatorQuad")
    assert metadata_response.status_code == 200

    metadata = metadata_response.json()
    assert "extents" not in metadata
    assert time_extent["interval"][0] == "2020-01-01T00:00:00"
    assert time_extent["interval"][1] == "2020-06-01T00:00:00"

    # Check elevation extent (now in layer)
    assert "elevation" in layer["extents"]
    elevation_extent = layer["extents"]["elevation"]
    assert "interval" in elevation_extent
    assert "units" in elevation_extent
    assert elevation_extent["units"] == "meters"
    assert "description" in elevation_extent
    assert elevation_extent["description"] == "Elevation above sea level"
    assert elevation_extent["interval"] == [0.0, 2000.0]

    # Check scenario extent (custom, now in layer)
    assert "scenario" in layer["extents"]
    scenario_extent = layer["extents"]["scenario"]
    assert "interval" in scenario_extent
    assert "description" in scenario_extent
    assert scenario_extent["description"] == "Climate scenario"
    assert scenario_extent["interval"] == ["RCP45", "Historical"]


async def test_dimension_extraction_utilities():
    """Test the dimension extraction utility functions directly"""

    # Create test data array with various dimension types
    time_coords = pd.date_range("2021-01-01", periods=4, freq="D")

    data_array = xr.DataArray(
        np.random.randn(4, 3, 10, 20),
        dims=["time", "depth", "lat", "lon"],
        coords={
            "time": (["time"], time_coords, {"axis": "T", "standard_name": "time"}),
            "depth": (
                ["depth"],
                [0, 10, 50],
                {
                    "units": "m",
                    "long_name": "Ocean depth",
                    "axis": "Z",
                    "positive": "down",
                },
            ),
            "lat": (
                ["lat"],
                np.linspace(-5, 5, 10),
                {"axis": "Y", "standard_name": "latitude", "units": "degrees_north"},
            ),
            "lon": (
                ["lon"],
                np.linspace(-10, 10, 20),
                {"axis": "X", "standard_name": "longitude", "units": "degrees_east"},
            ),
        },
    )

    ds = data_array.to_dataset(name="foo")
    ds["scalar"] = ((), 0, {"foo": "bar"})
    dimensions = await extract_dimension_extents(ds, "foo")

    # Should extract time and depth, but not lat/lon (spatial)
    assert len(dimensions) == 2

    # Check time dimension
    time_dim = next(d for d in dimensions if d.name == "time")
    assert time_dim.type.value == "temporal"
    assert len(time_dim.values) == 4
    assert time_dim.extent[0] == "2021-01-01T00:00:00"
    assert time_dim.extent[1] == "2021-01-04T00:00:00"
    assert time_dim.default == "2021-01-04T00:00:00"

    # Check depth dimension
    depth_dim = next(d for d in dimensions if d.name == "depth")
    assert depth_dim.type.value == "vertical"
    assert depth_dim.units == "m"
    assert depth_dim.description == "Ocean depth"
    assert depth_dim.extent == [0.0, 50.0]
    assert depth_dim.default == 0.0


def test_no_dimensions_dataset():
    """Test dataset with only spatial dimensions"""
    data = xr.Dataset(
        {
            "scalar": ((), 0, {"foo": "bar"}),
            "temperature": xr.DataArray(
                np.random.randn(90, 180),
                dims=["lat", "lon"],
                coords={
                    "lat": (
                        ["lat"],
                        np.linspace(-90, 90, 90),
                        {
                            "axis": "Y",
                            "standard_name": "latitude",
                            "units": "degrees_north",
                        },
                    ),
                    "lon": (
                        ["lon"],
                        np.linspace(-180, 180, 180),
                        {
                            "axis": "X",
                            "standard_name": "longitude",
                            "units": "degrees_east",
                        },
                    ),
                },
                attrs={"long_name": "Temperature"},
            ),
        }
    )

    rest = xpublish.Rest({"simple": data}, plugins={"tiles": TilesPlugin()})
    client = TestClient(rest.app)

    response = client.get("/datasets/simple/tiles/")
    assert response.status_code == 200

    response_data = response.json()
    tileset = response_data["tilesets"][0]
    layer = tileset["layers"][0]

    # Should have no dimensions (or dimensions should be None/empty)
    dimensions = layer.get("dimensions")
    assert dimensions is None or len(dimensions) == 0


async def test_cf_axis_detection():
    """Test that CF axis detection works correctly"""
    # Create dataset with non-standard dimension names but proper CF attributes
    time_coords = pd.date_range("2022-01-01", periods=3, freq="ME")

    data_array = xr.DataArray(
        np.random.randn(3, 2, 5, 8),
        dims=["month", "level", "y_coord", "x_coord"],  # Non-standard names
        coords={
            "month": (["month"], time_coords, {"axis": "T", "standard_name": "time"}),
            "level": (
                ["level"],
                [1000, 500],
                {"axis": "Z", "units": "hPa", "positive": "down"},
            ),
            "y_coord": (
                ["y_coord"],
                np.linspace(40, 50, 5),
                {"axis": "Y", "standard_name": "latitude"},
            ),
            "x_coord": (
                ["x_coord"],
                np.linspace(-10, 0, 8),
                {"axis": "X", "standard_name": "longitude"},
            ),
        },
    )

    ds = data_array.to_dataset(name="foo")
    ds["scalar"] = ((), 0, {"foo": "bar"})
    dimensions = await extract_dimension_extents(ds, "foo")

    # Should detect temporal and vertical dimensions despite non-standard names
    assert len(dimensions) == 2

    # Check that CF axis detection worked
    dim_names = {d.name for d in dimensions}
    assert "month" in dim_names  # Detected as temporal via CF axis T
    assert "level" in dim_names  # Detected as vertical via CF axis Z

    # Verify types are correctly assigned
    month_dim = next(d for d in dimensions if d.name == "month")
    level_dim = next(d for d in dimensions if d.name == "level")

    assert month_dim.type.value == "temporal"
    assert level_dim.type.value == "vertical"
    assert level_dim.units == "hPa"


async def test_helper_functions(air_dataset):
    """Test the helper functions for extracting bounds and generating limits"""
    from xpublish_tiles.xpublish.tiles.tile_matrix import (
        get_all_tile_matrix_set_ids,
        get_tile_matrix_limits,
    )

    # Test getting all TMS IDs
    tms_ids = get_all_tile_matrix_set_ids()
    assert isinstance(tms_ids, list)
    assert "WebMercatorQuad" in tms_ids
    assert len(tms_ids) >= 1

    # Test tile matrix limits generation with dataset
    limits = await get_tile_matrix_limits(
        "WebMercatorQuad", air_dataset, range(3)
    )  # Just 0-2
    assert len(limits) == 3
    assert limits[0].tileMatrix == "0"
    assert limits[1].tileMatrix == "1"
    assert limits[2].tileMatrix == "2"


def test_no_bbox_overlap_transparent_png():
    """Test that requesting a tile with no overlap with dataset bounds returns transparent PNG"""
    # EU3035 covers Europe, so we'll request a tile that's far away (e.g., in Pacific Ocean)
    rest = xpublish.Rest({"europe": EU3035.create()}, plugins={"tiles": TilesPlugin()})
    client = TestClient(rest.app)

    # Request a tile that's completely outside the dataset bounds
    # This tile at zoom 9 should be in the Pacific Ocean, far from Europe
    response = client.get(
        "/datasets/europe/tiles/WebMercatorQuad/9/10/200"
        "?variables=foo&style=raster/viridis&width=512&height=512"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    # Verify the returned image is fully transparent
    img = Image.open(io.BytesIO(response.content))
    img = img.convert("RGBA")

    # Check that all pixels are fully transparent (alpha = 0)
    alpha_channel = np.array(img)[:, :, 3]
    assert np.all(alpha_channel == 0), "All pixels should be fully transparent"

    # Also check image dimensions
    assert img.size == (512, 512), "Image should have correct dimensions"


def test_para_hires_zoom_level_2_size_limit():
    """Test that requesting zoom level 2 for PARA_HIRES dataset triggers size limit error"""
    rest = xpublish.Rest(
        {"para_hires": PARA_HIRES.create()}, plugins={"tiles": TilesPlugin()}
    )
    client = TestClient(rest.app)

    response = client.get(
        "/datasets/para_hires/tiles/WebMercatorQuad/2/1/1"
        "?variables=foo&style=raster/viridis&width=256&height=256"
    )
    assert response.status_code == 413
    error_detail = response.json()["detail"]
    assert "WebMercatorQuad/2/1/1" in error_detail
    assert "request too big" in error_detail
    assert "Please choose a higher zoom level" in error_detail


def test_tilejson_endpoint():
    """Test the TileJSON endpoint functionality comprehensively"""
    import pandas as pd

    # Create dataset with rich metadata and time dimension
    time_coords = pd.date_range("2020-01-01", periods=3, freq="MS")
    data = xr.Dataset(
        {
            "scalar": ((), 0, {"foo": "bar"}),
            "temperature": xr.DataArray(
                np.random.randn(3, 90, 180),
                dims=["time", "lat", "lon"],
                coords={
                    "time": (
                        ["time"],
                        time_coords,
                        {"axis": "T", "standard_name": "time"},
                    ),
                    "lat": (
                        ["lat"],
                        np.linspace(-90, 90, 90),
                        {
                            "axis": "Y",
                            "standard_name": "latitude",
                            "units": "degrees_north",
                        },
                    ),
                    "lon": (
                        ["lon"],
                        np.linspace(-180, 180, 180),
                        {
                            "axis": "X",
                            "standard_name": "longitude",
                            "units": "degrees_east",
                        },
                    ),
                },
                attrs={"long_name": "Temperature"},
            ),
        },
        attrs={
            "title": "Global Temperature Data",
            "description": "Sample temperature dataset",
            "attribution": "Test Data Corp",
            "version": "2.0.1",
        },
    )

    rest = xpublish.Rest({"temp": data}, plugins={"tiles": TilesPlugin()})
    client = TestClient(rest.app)

    # Test TileJSON endpoint with dimension selectors and colormap
    response = client.get(
        "/datasets/temp/tiles/WebMercatorQuad/tilejson.json"
        "?variables=temperature&style=raster/custom&width=512&height=512&time=2020-02-01&"
        f"colorscalerange=-3,3&colormap={CUSTOM_COLORMAP}"
    )
    assert response.status_code == 200

    tilejson = response.json()

    # Check required TileJSON fields
    assert tilejson["tilejson"] == "3.0.0"
    assert "tiles" in tilejson
    assert len(tilejson["tiles"]) == 1

    # Check tile URL template format
    tile_url = tilejson["tiles"][0]
    print(tile_url)
    assert "{z}" in tile_url
    assert "{y}" in tile_url
    assert "{x}" in tile_url
    assert "variables=temperature" in tile_url
    assert "style=raster/custom" in tile_url
    assert "width=512" in tile_url
    assert "height=512" in tile_url
    assert "time=2020-02-01" in tile_url  # Dimension selector preserved
    assert "colorscalerange=-3,3" in tile_url  # Additional param preserved
    assert "render_errors=false" in tile_url  # Default param included
    assert f"colormap={CUSTOM_COLORMAP}" in tile_url  # Additional param preserved

    # Check optional fields
    assert tilejson["scheme"] == "xyz"
    assert "bounds" in tilejson
    assert "minzoom" in tilejson
    assert "maxzoom" in tilejson

    # Check that dataset metadata is included
    assert tilejson["name"] == "Global Temperature Data"
    assert tilejson["description"] == "Sample temperature dataset"
    assert tilejson["attribution"] == "Test Data Corp"
    assert tilejson["version"] == "2.0.1"

    # Bounds should be a 4-element array [west, south, east, north]
    if tilejson["bounds"] is not None:
        assert len(tilejson["bounds"]) == 4
        west, south, east, north = tilejson["bounds"]
        assert west < east
        assert south < north

    # Zoom levels should be valid
    if tilejson["minzoom"] is not None:
        assert 0 <= tilejson["minzoom"] <= 30
    if tilejson["maxzoom"] is not None:
        assert tilejson["maxzoom"] == 24
        if tilejson["minzoom"] is not None:
            assert tilejson["minzoom"] <= tilejson["maxzoom"]


def test_tilejson_invalid_tile_matrix_set():
    """Test TileJSON endpoint returns 404 for invalid tile matrix set"""
    rest = xpublish.Rest({"air": xr.Dataset()}, plugins={"tiles": TilesPlugin()})
    client = TestClient(rest.app)

    response = client.get(
        "/datasets/air/tiles/InvalidTMS/tilejson.json"
        "?variables=air&style=raster/viridis&width=256&height=256"
    )
    assert response.status_code == 404
    assert "Tile matrix set not found" in response.json()["detail"]


def test_tilejson_missing_variables():
    """Test TileJSON endpoint handles validation errors for missing required fields"""
    rest = xpublish.Rest({"air": xr.Dataset()}, plugins={"tiles": TilesPlugin()})
    client = TestClient(rest.app)

    # Test missing variables parameter (required field)
    response = client.get(
        "/datasets/air/tiles/WebMercatorQuad/tilejson.json"
        "?style=raster/viridis&width=256&height=256"
    )
    assert response.status_code == 422
    # Check that validation error mentions missing variables field
    detail = response.json()["detail"]
    assert any(error.get("loc") == ["query", "variables"] for error in detail)


def test_tilejson_bounds_normalized_from_0_360_global():
    """TileJSON bounds should normalize 0..360 longitudes to [-180, 180]."""
    data = xr.Dataset(
        {
            "scalar": ((), 0, {"foo": "bar"}),
            "temperature": xr.DataArray(
                np.random.randn(90, 180),
                dims=["lat", "lon"],
                coords={
                    "lat": (
                        ["lat"],
                        np.linspace(-90, 90, 90),
                        {"axis": "Y", "standard_name": "latitude"},
                    ),
                    "lon": (
                        ["lon"],
                        np.linspace(0, 360, 180),
                        {"axis": "X", "standard_name": "longitude"},
                    ),
                },
            ),
        }
    )

    rest = xpublish.Rest({"globe": data}, plugins={"tiles": TilesPlugin()})
    client = TestClient(rest.app)

    resp = client.get(
        "/datasets/globe/tiles/WebMercatorQuad/tilejson.json"
        "?variables=temperature&style=raster/viridis&width=256&height=256&f=png"
    )
    assert resp.status_code == 200
    tj = resp.json()
    assert tj["bounds"][0] == -180.0
    assert tj["bounds"][2] == 180.0


def test_tilejson_bounds_dateline_crossing_0_360():
    """For dateline-crossing 0..360 datasets, use full world longitudes in TileJSON."""
    data = xr.Dataset(
        {
            "scalar": ((), 0, {"foo": "bar"}),
            "temperature": xr.DataArray(
                np.random.randn(10, 20),
                dims=["lat", "lon"],
                coords={
                    "lat": (
                        ["lat"],
                        np.linspace(10, -10, 10),
                        {"axis": "Y", "standard_name": "latitude"},
                    ),
                    "lon": (
                        ["lon"],
                        np.linspace(10, 350, 20),
                        {"axis": "X", "standard_name": "longitude"},
                    ),
                },
            ),
        }
    )

    rest = xpublish.Rest({"cross": data}, plugins={"tiles": TilesPlugin()})
    client = TestClient(rest.app)

    resp = client.get(
        "/datasets/cross/tiles/WebMercatorQuad/tilejson.json"
        "?variables=temperature&style=raster/viridis&width=256&height=256&f=png"
    )
    assert resp.status_code == 200
    tj = resp.json()
    bounds = tj.get("bounds")
    if bounds is not None:
        assert bounds[0] == -180.0
        assert bounds[2] == 180.0


def test_selection_method_dsl():
    ds = IFS.create()
    t0 = ds.time.data[0]
    ds = ds.reindex(time=pd.date_range(t0, periods=ds.sizes["time"] + 3, freq="6h"))

    rest = xpublish.Rest({"ifs": ds}, plugins={"tiles": TilesPlugin()})
    client = TestClient(rest.app)
    resp = client.get(
        "/datasets/ifs/tiles/WebMercatorQuad/tilejson.json"
        "?variables=foo&time=nearest::2000-01-01T04:00"
        "&style=raster/viridis&width=256&height=256&f=png"
    )
    assert resp.status_code == 200

    resp = client.get(
        "/datasets/ifs/tiles/WebMercatorQuad/tilejson.json"
        "?variables=foo&time=nearest::2000-01-01T04:00&step=3h"
        "&style=raster/viridis&width=256&height=256&f=png"
    )
    assert resp.status_code == 200


def test_tilejson_bounds_with_decreasing_lat_lon():
    """Bounds should normalize correctly when lat and lon coords decrease."""
    data = xr.Dataset(
        {
            "scalar": ((), 0, {"foo": "bar"}),
            "temperature": xr.DataArray(
                np.random.randn(90, 180),
                dims=["lat", "lon"],
                coords={
                    "lat": (
                        ["lat"],
                        np.linspace(90, -90, 90),
                        {"axis": "Y", "standard_name": "latitude"},
                    ),
                    "lon": (
                        ["lon"],
                        np.linspace(0, 360, 180, endpoint=False),
                        {"axis": "X", "standard_name": "longitude"},
                    ),
                },
            ),
        }
    )

    rest = xpublish.Rest({"dec": data}, plugins={"tiles": TilesPlugin()})
    client = TestClient(rest.app)

    resp = client.get(
        "/datasets/dec/tiles/WebMercatorQuad/tilejson.json"
        "?variables=temperature&style=raster/viridis&width=256&height=256&f=png"
    )
    assert resp.status_code == 200
    tj = resp.json()
    bounds = tj.get("bounds")
    if bounds is not None:
        assert bounds == [-180.0, -90.0, 180.0, 90.0]


def test_colormap_tile_endpoint(xpublish_client):
    """Test that tiles can be generated with custom colormap using raster/custom style."""
    from urllib.parse import quote

    colormap = {"0": "#ff0000", "255": "#0000ff"}  # Red to blue
    colormap_json = json.dumps(colormap)
    colormap_encoded = quote(colormap_json)

    # Must specify style=raster/custom when using colormap
    url = (
        f"/datasets/air/tiles/WebMercatorQuad/0/0/0"
        f"?variables=air&width=256&height=256&style=raster/custom&colormap={colormap_encoded}"
    )
    response = xpublish_client.get(url)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    # Verify it's a valid PNG image
    image_data = io.BytesIO(response.content)
    img = Image.open(image_data)
    assert img.format == "PNG"
    assert img.size == (256, 256)

    # Test that invalid colormap keys are rejected
    bad_colormap = {"0": "#ff0000", "2": "#0000ff"}
    colormap_encoded = quote(json.dumps(bad_colormap))
    url = (
        f"/datasets/air/tiles/WebMercatorQuad/0/0/0"
        f"?variables=air&width=256&height=256&style=raster/custom&colormap={colormap_encoded}"
    )
    response = xpublish_client.get(url)
    assert response.status_code == 422


def test_colormap_with_style_parameter_succeeds(xpublish_client):
    """Test that colormap requires raster/custom style, other styles return 422."""
    from urllib.parse import quote

    colormap = {"0": "#ff0000", "255": "#0000ff"}
    colormap_json = json.dumps(colormap)
    colormap_encoded = quote(colormap_json)

    # Test that colormap with raster/custom succeeds
    response = xpublish_client.get(
        f"/datasets/air/tiles/WebMercatorQuad/0/0/0"
        f"?variables=air&width=256&height=256&colormap={colormap_encoded}&style=raster/custom"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    # Verify it's a valid PNG image
    image_data = io.BytesIO(response.content)
    img = Image.open(image_data)
    assert img.format == "PNG"
    assert img.size == (256, 256)

    # Test that colormap with raster/viridis fails with 422
    response = xpublish_client.get(
        f"/datasets/air/tiles/WebMercatorQuad/0/0/0"
        f"?variables=air&width=256&height=256&colormap={colormap_encoded}&style=raster/viridis"
    )
    assert response.status_code == 422

    # Test that colormap with raster/default fails with 422
    response = xpublish_client.get(
        f"/datasets/air/tiles/WebMercatorQuad/0/0/0"
        f"?variables=air&width=256&height=256&colormap={colormap_encoded}&style=raster/default"
    )
    assert response.status_code == 422
