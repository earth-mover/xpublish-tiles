import numpy as np
import pytest
import xpublish
from fastapi.testclient import TestClient

import xarray as xr
from xpublish_tiles.datasets import EU3035
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
    # Create a dataset with rich metadata including time dimension
    import pandas as pd

    time_coords = pd.date_range("2020-01-01", periods=12, freq="MS")

    data = xr.Dataset(
        {
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

    # Check that dimensions are no longer in layers (moved to tileset level)
    assert "dimensions" not in layer or layer["dimensions"] is None

    # Check that extents are now in the layer
    assert "extents" in layer
    assert layer["extents"] is not None
    assert "time" in layer["extents"]

    time_extent = layer["extents"]["time"]
    assert "interval" in time_extent
    assert len(time_extent["interval"]) == 2
    assert time_extent["interval"][0] == "2020-01-01T00:00:00Z"
    assert time_extent["interval"][1] == "2020-12-01T00:00:00Z"

    # Test the tileset metadata endpoint - extents should no longer be at tileset level
    metadata_response = client.get("/datasets/climate/tiles/WebMercatorQuad")
    assert metadata_response.status_code == 200

    metadata = metadata_response.json()
    assert "extents" not in metadata


def test_multi_dimensional_dataset():
    """Test dataset with multiple dimension types (time, elevation, custom)"""
    import pandas as pd

    # Create a dataset with multiple dimensions
    time_coords = pd.date_range("2020-01-01", periods=6, freq="MS")
    elevation_coords = [0, 100, 500, 1000, 2000]
    scenario_coords = ["RCP45", "RCP85", "Historical"]

    data = xr.Dataset(
        {
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
            )
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
    assert time_extent["interval"][0] == "2020-01-01T00:00:00Z"
    assert time_extent["interval"][1] == "2020-06-01T00:00:00Z"

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
    assert scenario_extent["interval"] == ["RCP45", "RCP85", "Historical"]

    from xpublish_tiles.xpublish.tiles.metadata import extract_dataset_extents

    # Create a dataset with multiple dimensions
    time_coords = pd.date_range("2023-01-01", periods=3, freq="H")
    elevation_coords = [0, 100, 500]
    scenario_coords = ["A", "B"]

    dataset = xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.random.randn(3, 3, 2, 5, 10),
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
                            "long_name": "Height above ground",
                            "axis": "Z",
                        },
                    ),
                    "scenario": (
                        ["scenario"],
                        scenario_coords,
                        {"long_name": "Test scenario"},
                    ),
                    "lat": (
                        ["lat"],
                        np.linspace(-2, 2, 5),
                        {"axis": "Y", "standard_name": "latitude"},
                    ),
                    "lon": (
                        ["lon"],
                        np.linspace(-5, 5, 10),
                        {"axis": "X", "standard_name": "longitude"},
                    ),
                },
            )
        }
    )

    extents = extract_dataset_extents(dataset, "temperature")

    # Should have 3 non-spatial dimensions
    assert len(extents) == 3
    assert "time" in extents
    assert "elevation" in extents
    assert "scenario" in extents

    # Check time extent
    time_extent = extents["time"]
    assert "interval" in time_extent
    assert "resolution" in time_extent
    assert time_extent["interval"][0] == "2023-01-01T00:00:00Z"
    assert time_extent["interval"][1] == "2023-01-01T02:00:00Z"
    assert time_extent["resolution"] == "PT1H"  # Hourly

    # Check elevation extent
    elevation_extent = extents["elevation"]
    assert "interval" in elevation_extent
    assert "units" in elevation_extent
    assert "description" in elevation_extent
    assert "resolution" in elevation_extent
    assert elevation_extent["interval"] == [0.0, 500.0]
    assert elevation_extent["units"] == "meters"
    assert elevation_extent["description"] == "Height above ground"
    assert elevation_extent["resolution"] == 100.0  # Min step size

    # Check scenario extent (categorical)
    scenario_extent = extents["scenario"]
    assert "interval" in scenario_extent
    assert "description" in scenario_extent
    assert scenario_extent["interval"] == ["A", "B"]
    assert scenario_extent["description"] == "Test scenario"


def test_calculate_temporal_resolution():
    """Test the _calculate_temporal_resolution function directly"""
    from xpublish_tiles.xpublish.tiles.metadata import _calculate_temporal_resolution

    # Test hourly resolution
    hourly_values = [
        "2023-01-01T00:00:00Z",
        "2023-01-01T01:00:00Z",
        "2023-01-01T02:00:00Z",
        "2023-01-01T03:00:00Z",
    ]
    assert _calculate_temporal_resolution(hourly_values) == "PT1H"

    # Test daily resolution
    daily_values = [
        "2023-01-01T00:00:00Z",
        "2023-01-02T00:00:00Z",
        "2023-01-03T00:00:00Z",
    ]
    assert _calculate_temporal_resolution(daily_values) == "P1D"

    # Test monthly resolution (approximately)
    monthly_values = [
        "2023-01-01T00:00:00Z",
        "2023-02-01T00:00:00Z",
        "2023-03-01T00:00:00Z",
    ]
    result = _calculate_temporal_resolution(monthly_values)
    assert result.startswith("P") and result.endswith("D")  # Should be in days

    # Test 15-minute resolution
    minute_values = [
        "2023-01-01T00:00:00Z",
        "2023-01-01T00:15:00Z",
        "2023-01-01T00:30:00Z",
    ]
    assert _calculate_temporal_resolution(minute_values) == "PT15M"

    # Test edge cases
    assert _calculate_temporal_resolution([]) == "PT1H"  # Empty list
    assert (
        _calculate_temporal_resolution(["2023-01-01T00:00:00Z"]) == "PT1H"
    )  # Single value
    assert _calculate_temporal_resolution([1, 2, 3]) == "PT1H"  # Non-string values


def test_dimension_extraction_utilities():
    """Test the dimension extraction utility functions directly"""
    import pandas as pd

    from xpublish_tiles.xpublish.tiles.tile_matrix import extract_dimension_extents

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

    dimensions = extract_dimension_extents(data_array)

    # Should extract time and depth, but not lat/lon (spatial)
    assert len(dimensions) == 2

    # Check time dimension
    time_dim = next(d for d in dimensions if d.name == "time")
    assert time_dim.type.value == "temporal"
    assert len(time_dim.values) == 4
    assert time_dim.extent[0] == "2021-01-01T00:00:00Z"
    assert time_dim.extent[1] == "2021-01-04T00:00:00Z"

    # Check depth dimension
    depth_dim = next(d for d in dimensions if d.name == "depth")
    assert depth_dim.type.value == "vertical"
    assert depth_dim.units == "m"
    assert depth_dim.description == "Ocean depth"
    assert depth_dim.extent == [0.0, 50.0]


def test_no_dimensions_dataset():
    """Test dataset with only spatial dimensions"""
    data = xr.Dataset(
        {
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
            )
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


def test_cf_axis_detection():
    """Test that CF axis detection works correctly"""
    import pandas as pd

    from xpublish_tiles.xpublish.tiles.tile_matrix import extract_dimension_extents

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

    dimensions = extract_dimension_extents(data_array)

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


def test_helper_functions():
    """Test the helper functions for extracting bounds and generating limits"""
    from xpublish_tiles.xpublish.tiles.tile_matrix import (
        extract_dataset_bounds,
        get_all_tile_matrix_set_ids,
        get_tile_matrix_limits,
    )

    # Test dataset bounds extraction
    data = xr.Dataset(
        {
            "temp": xr.DataArray(
                np.random.randn(10, 20),
                dims=["lat", "lon"],
                coords={
                    "lat": np.linspace(-45, 45, 10),
                    "lon": np.linspace(-90, 90, 20),
                },
            )
        }
    )

    bounds = extract_dataset_bounds(data)
    assert bounds is not None
    assert bounds.lowerLeft == [-90.0, -45.0]
    assert bounds.upperRight == [90.0, 45.0]
    assert bounds.crs == "http://www.opengis.net/def/crs/EPSG/0/4326"

    # Test getting all TMS IDs
    tms_ids = get_all_tile_matrix_set_ids()
    assert isinstance(tms_ids, list)
    assert "WebMercatorQuad" in tms_ids
    assert len(tms_ids) >= 1

    # Test tile matrix limits generation
    limits = get_tile_matrix_limits("WebMercatorQuad", range(3))  # Just 0-2
    assert len(limits) == 3
    assert limits[0].tileMatrix == "0"
    assert limits[0].maxTileRow == 0  # 2^0 - 1 = 0
    assert limits[1].maxTileRow == 1  # 2^1 - 1 = 1
    assert limits[2].maxTileRow == 3  # 2^2 - 1 = 3


def test_bbox_overlap_error():
    """Test that requesting a tile with no overlap with dataset bounds returns proper error"""
    # EU3035 covers Europe, so we'll request a tile that's far away (e.g., in Pacific Ocean)
    rest = xpublish.Rest({"europe": EU3035.create()}, plugins={"tiles": TilesPlugin()})
    client = TestClient(rest.app)

    # Request a tile that's completely outside the dataset bounds
    # This tile at zoom 9 should be in the Pacific Ocean, far from Europe
    response = client.get(
        "/datasets/europe/tiles/WebMercatorQuad/9/10/200"
        "?variables=foo&style=raster/viridis&width=256&height=256"
    )

    # Should return 400 Bad Request with appropriate error message
    assert response.status_code == 400
    error_detail = response.json()["detail"]
    assert "no overlap with dataset bounds" in error_detail
    assert "WebMercatorQuad/9/10/200" in error_detail
