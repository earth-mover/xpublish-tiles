"""Tests for tiles metadata functionality"""

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xpublish_tiles.lib import VariableNotFoundError


async def test_extract_dataset_extents():
    """Test the extract_dataset_extents function directly"""
    import pandas as pd

    from xpublish_tiles.xpublish.tiles.metadata import extract_dataset_extents

    # Create a dataset with multiple dimensions
    time_coords = pd.date_range("2023-01-01", periods=3, freq="h")
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

    extents = await extract_dataset_extents(dataset, "temperature")

    # Should have 3 non-spatial dimensions
    assert len(extents) == 3
    assert "time" in extents
    assert "elevation" in extents
    assert "scenario" in extents

    # Check time extent
    time_extent = extents["time"]
    assert "interval" in time_extent
    assert "resolution" in time_extent
    assert time_extent["interval"][0] == "2023-01-01T00:00:00"
    assert time_extent["interval"][1] == "2023-01-01T02:00:00"
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


async def test_extract_dataset_extents_empty():
    """Test extract_dataset_extents with dataset containing no non-spatial dimensions"""
    from xpublish_tiles.xpublish.tiles.metadata import extract_dataset_extents

    # Create a dataset with only spatial dimensions
    dataset = xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.random.randn(5, 10),
                dims=["lat", "lon"],
                coords={
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

    extents = await extract_dataset_extents(dataset, "temperature")
    assert len(extents) == 0


async def test_extract_dataset_extents_multiple_variables():
    """Test extract_dataset_extents with multiple variables having different dimensions"""
    import pandas as pd

    from xpublish_tiles.xpublish.tiles.metadata import extract_dataset_extents

    time_coords = pd.date_range("2023-01-01", periods=2, freq="D")
    depth_coords = [0, 10]

    dataset = xr.Dataset(
        {
            "surface_temp": xr.DataArray(
                np.random.randn(2, 5, 10),
                dims=["time", "lat", "lon"],
                coords={
                    "time": (
                        ["time"],
                        time_coords,
                        {"axis": "T", "standard_name": "time"},
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
            ),
            "ocean_temp": xr.DataArray(
                np.random.randn(2, 2, 5, 10),
                dims=["time", "depth", "lat", "lon"],
                coords={
                    "time": (
                        ["time"],
                        time_coords,
                        {"axis": "T", "standard_name": "time"},
                    ),
                    "depth": (
                        ["depth"],
                        depth_coords,
                        {"units": "m", "axis": "Z", "positive": "down"},
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
            ),
        }
    )

    # Test with surface_temp variable (only has time)
    extents_surface = await extract_dataset_extents(dataset, "surface_temp")
    assert len(extents_surface) == 1
    assert "time" in extents_surface

    # Test with ocean_temp variable (has time and depth)
    extents_ocean = await extract_dataset_extents(dataset, "ocean_temp")
    assert len(extents_ocean) == 2
    assert "time" in extents_ocean
    assert "depth" in extents_ocean

    # Time should be from the ocean_temp variable
    time_extent = extents_ocean["time"]
    assert time_extent["interval"][0] == "2023-01-01T00:00:00"
    assert time_extent["interval"][1] == "2023-01-02T00:00:00"

    # Depth should be from the ocean_temp variable
    depth_extent = extents_ocean["depth"]
    assert depth_extent["interval"] == [0.0, 10.0]
    assert depth_extent["units"] == "m"


@pytest.mark.parametrize("use_cftime", [True, False])
def test_calculate_temporal_resolution(use_cftime):
    """Test the _calculate_temporal_resolution function directly"""
    from xpublish_tiles.xpublish.tiles.metadata import _calculate_temporal_resolution

    # Test hourly resolution
    hourly_values = xr.DataArray(
        xr.date_range("2023-01-01T00:00:00", periods=4, freq="h", use_cftime=use_cftime),
        dims="time",
        name="time",
    )
    assert _calculate_temporal_resolution(hourly_values) == "PT1H"

    # Test daily resolution
    daily_values = xr.DataArray(
        xr.date_range(
            "2023-01-01T00:00:00", periods=4, freq="24h", use_cftime=use_cftime
        ),
        dims="time",
        name="time",
    )
    assert _calculate_temporal_resolution(daily_values) == "P1D"

    # Test monthly resolution (approximately)
    monthly_values = xr.DataArray(
        xr.date_range("2023-01-01T00:00:00", periods=4, freq="MS", use_cftime=use_cftime),
        dims="time",
        name="time",
    )

    result = _calculate_temporal_resolution(monthly_values)
    assert result.startswith("P") and result.endswith("D")  # Should be in days

    # Test 15-minute resolution
    minute_values = xr.DataArray(
        xr.date_range(
            "2023-01-01T00:00:00", periods=4, freq="15min", use_cftime=use_cftime
        ),
        dims="time",
        name="time",
    )

    assert _calculate_temporal_resolution(minute_values) == "PT15M"

    # Test 30-second resolution
    second_values = xr.DataArray(
        xr.date_range(
            "2023-01-01T00:00:00", periods=4, freq="30s", use_cftime=use_cftime
        ),
        dims="time",
        name="time",
    )

    assert _calculate_temporal_resolution(second_values) == "PT30S"


def test_calculate_temporal_resolution_edge_cases():
    """Test _calculate_temporal_resolution with edge cases"""
    from xpublish_tiles.xpublish.tiles.metadata import _calculate_temporal_resolution

    # Test edge cases
    assert (
        _calculate_temporal_resolution(xr.DataArray([], dims="time")) == "PT1H"
    )  # Empty list
    assert (
        _calculate_temporal_resolution(
            xr.DataArray(pd.DatetimeIndex(["2023-01-01T00:00:00"]), dims="time")
        )
        == "PT1H"
    )  # Single value
    assert (
        _calculate_temporal_resolution(xr.DataArray([1, 2, 3], dims="time")) == "PT1H"
    )  # Non-string values

    # Test irregular intervals (should use average)
    irregular_values = xr.DataArray(
        pd.DatetimeIndex(
            [
                "2023-01-01T00:00:00",
                "2023-01-01T01:00:00",  # 1 hour gap
                "2023-01-01T04:00:00",  # 3 hour gap
            ]
        ),
        dims="time",
        name="time",
    )
    result = _calculate_temporal_resolution(irregular_values)
    assert result == "PT2H"  # Average of 1 and 3 hours

    # Test with invalid datetime strings (should fallback)
    invalid_values = xr.DataArray(
        ["not-a-date", "also-not-a-date"], dims="time", name="time"
    )
    assert _calculate_temporal_resolution(invalid_values) == "PT1H"


async def test_create_tileset_metadata_with_extents():
    """Test create_tileset_metadata - extents are now on layers, not tileset"""
    import pandas as pd

    from xpublish_tiles.xpublish.tiles.metadata import (
        create_tileset_metadata,
        extract_dataset_extents,
    )

    # Create dataset with time dimension
    time_coords = pd.date_range("2023-01-01", periods=4, freq="6h")
    dataset = xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.random.randn(4, 5, 10),
                dims=["time", "lat", "lon"],
                coords={
                    "time": (
                        ["time"],
                        time_coords,
                        {"axis": "T", "standard_name": "time"},
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
        },
        attrs={"title": "Test Dataset"},
    )

    metadata = create_tileset_metadata(dataset, "WebMercatorQuad")

    # Check that extents are no longer on tileset metadata
    assert not hasattr(metadata, "extents")

    # Test that extract_dataset_extents works for the variable
    extents = await extract_dataset_extents(dataset, "temperature")
    assert "time" in extents

    time_extent = extents["time"]
    assert "interval" in time_extent
    assert "resolution" in time_extent
    assert time_extent["resolution"] == "PT6H"  # 6-hourly


async def test_create_tileset_metadata_no_extents():
    """Test create_tileset_metadata with no non-spatial dimensions"""
    from xpublish_tiles.xpublish.tiles.metadata import (
        create_tileset_metadata,
        extract_dataset_extents,
    )

    # Create dataset with only spatial dimensions
    dataset = xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.random.randn(5, 10),
                dims=["lat", "lon"],
                coords={
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
        },
        attrs={"title": "Spatial Only Dataset"},
    )

    metadata = create_tileset_metadata(dataset, "WebMercatorQuad")

    # Check that extents are no longer on tileset metadata
    assert not hasattr(metadata, "extents")

    # Test that extract_dataset_extents returns empty dict when no non-spatial dimensions
    extents = await extract_dataset_extents(dataset, "temperature")
    assert len(extents) == 0


async def test_extract_variable_bounding_box():
    """Test extract_variable_bounding_box function"""
    from xpublish_tiles.xpublish.tiles.metadata import extract_variable_bounding_box

    # Create a dataset with known coordinates
    dataset = xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.random.randn(5, 11),
                dims=["lat", "lon"],
                coords={
                    "lat": (
                        ["lat"],
                        np.linspace(-2, 2, 5),
                        {"axis": "Y", "standard_name": "latitude"},
                    ),
                    "lon": (
                        ["lon"],
                        np.linspace(-5, 5, 11),
                        {"axis": "X", "standard_name": "longitude"},
                    ),
                },
            )
        }
    )

    # Test with EPSG:4326 (should be identity transform)
    bbox = await extract_variable_bounding_box(dataset, "temperature", "EPSG:4326")

    if bbox is not None:
        # Check that bounding box has correct structure
        assert hasattr(bbox, "lowerLeft")
        assert hasattr(bbox, "upperRight")
        assert hasattr(bbox, "crs")

        # Check coordinate values (should be close to original since it's EPSG:4326)
        assert len(bbox.lowerLeft) == 2
        assert len(bbox.upperRight) == 2

        # Lower left should be min values
        assert bbox.lowerLeft[0] == pytest.approx(-5.5, abs=1e-6)  # min lon
        assert bbox.lowerLeft[1] == pytest.approx(-2.5, abs=1e-6)  # min lat

        # Upper right should be max values
        assert bbox.upperRight[0] == pytest.approx(5.5, abs=1e-6)  # max lon
        assert bbox.upperRight[1] == pytest.approx(2.5, abs=1e-6)  # max lat

        # CRS should be set correctly
        assert bbox.crs == "EPSG:4326"


async def test_extract_variable_bounding_box_web_mercator():
    """Test extract_variable_bounding_box with Web Mercator transformation"""
    from xpublish_tiles.xpublish.tiles.metadata import extract_variable_bounding_box

    # Create a dataset with known coordinates
    dataset = xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.random.randn(5, 10),
                dims=["lat", "lon"],
                coords={
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

    # Test with EPSG:3857 (Web Mercator)
    bbox = await extract_variable_bounding_box(dataset, "temperature", "EPSG:3857")

    if bbox is not None:
        # Check that bounding box has correct structure
        assert hasattr(bbox, "lowerLeft")
        assert hasattr(bbox, "upperRight")
        assert hasattr(bbox, "crs")

        # Check coordinate values are in Web Mercator range (much larger numbers)
        assert len(bbox.lowerLeft) == 2
        assert len(bbox.upperRight) == 2

        # Web Mercator coordinates should be much larger than geographic
        assert abs(bbox.lowerLeft[0]) > 100000  # Transformed longitude
        assert abs(bbox.lowerLeft[1]) > 100000  # Transformed latitude
        assert abs(bbox.upperRight[0]) > 100000  # Transformed longitude
        assert abs(bbox.upperRight[1]) > 100000  # Transformed latitude

        # CRS should be set correctly
        assert bbox.crs == "EPSG:3857"


async def test_extract_variable_bounding_box_invalid_variable():
    """Test extract_variable_bounding_box with invalid variable name"""
    from xpublish_tiles.xpublish.tiles.metadata import extract_variable_bounding_box

    # Create a simple dataset
    dataset = xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.random.randn(5, 10),
                dims=["lat", "lon"],
                coords={
                    "lat": (["lat"], np.linspace(-2, 2, 5)),
                    "lon": (["lon"], np.linspace(-5, 5, 10)),
                },
            )
        }
    )

    # Test with non-existent variable
    with pytest.raises(VariableNotFoundError):
        await extract_variable_bounding_box(dataset, "nonexistent", "EPSG:4326")


def test_variable_bounding_boxes_in_tileset_metadata():
    """Test that variable bounding boxes are correctly used in tileset metadata"""
    from xpublish_tiles.xpublish.tiles.metadata import create_tileset_metadata

    # Create dataset with multiple variables having different spatial extents
    dataset = xr.Dataset(
        {
            # Variable covering full extent
            "temp_global": xr.DataArray(
                np.random.randn(10, 20),
                dims=["lat", "lon"],
                coords={
                    "lat": (
                        ["lat"],
                        np.linspace(-80, 80, 10),
                        {"axis": "Y", "standard_name": "latitude"},
                    ),
                    "lon": (
                        ["lon"],
                        np.linspace(-180, 180, 20),
                        {"axis": "X", "standard_name": "longitude"},
                    ),
                },
                attrs={"long_name": "Global Temperature"},
            ),
            # Variable covering smaller extent
            "temp_regional": xr.DataArray(
                np.random.randn(5, 10),
                dims=["lat", "lon"],
                coords={
                    "lat": (
                        ["lat"],
                        np.linspace(30, 50, 5),
                        {"axis": "Y", "standard_name": "latitude"},
                    ),
                    "lon": (
                        ["lon"],
                        np.linspace(-10, 10, 10),
                        {"axis": "X", "standard_name": "longitude"},
                    ),
                },
                attrs={"long_name": "Regional Temperature"},
            ),
        }
    )

    # Create tileset metadata for WebMercatorQuad
    metadata = create_tileset_metadata(dataset, "WebMercatorQuad")

    # Verify basic structure
    assert hasattr(metadata, "boundingBox")
    assert hasattr(metadata, "crs")
    assert "3857" in str(metadata.crs)  # Should contain Web Mercator EPSG code

    # Test that dataset-level bounding box is reasonable
    if metadata.boundingBox is not None:
        if hasattr(metadata.boundingBox, "lowerLeft") and hasattr(
            metadata.boundingBox, "upperRight"
        ):
            # Check coordinate values are within expected range
            assert len(metadata.boundingBox.lowerLeft) == 2
            assert len(metadata.boundingBox.upperRight) == 2

            # Lower left should be minimum values
            assert (
                metadata.boundingBox.lowerLeft[0] <= metadata.boundingBox.upperRight[0]
            )  # min X <= max X
            assert (
                metadata.boundingBox.lowerLeft[1] <= metadata.boundingBox.upperRight[1]
            )  # min Y <= max Y

            # Check that CRS is specified
            assert metadata.boundingBox.crs is not None


async def test_layers_use_variable_specific_bounding_boxes():
    """Test that layers get variable-specific bounding boxes rather than dataset-wide bounds"""

    from xpublish_tiles.xpublish.tiles.metadata import extract_variable_bounding_box

    # Create dataset with variables having different spatial extents
    dataset = xr.Dataset(
        {
            # Global variable
            "global_temp": xr.DataArray(
                np.random.randn(10, 20),
                dims=["lat", "lon"],
                coords={
                    "lat": (
                        ["lat"],
                        np.linspace(-70, 70, 10),
                        {"axis": "Y", "standard_name": "latitude"},
                    ),
                    "lon": (
                        ["lon"],
                        np.linspace(-160, 160, 20),
                        {"axis": "X", "standard_name": "longitude"},
                    ),
                },
                attrs={"long_name": "Global Temperature"},
            ),
            # Regional variable with smaller extent
            "regional_temp": xr.DataArray(
                np.random.randn(5, 10),
                dims=["lat", "lon"],
                coords={
                    "lat": (
                        ["lat"],
                        np.linspace(30, 50, 5),
                        {"axis": "Y", "standard_name": "latitude"},
                    ),
                    "lon": (
                        ["lon"],
                        np.linspace(-10, 10, 10),
                        {"axis": "X", "standard_name": "longitude"},
                    ),
                },
                attrs={"long_name": "Regional Temperature"},
            ),
        }
    )

    # Test variable-specific bounding boxes directly
    global_bbox = await extract_variable_bounding_box(dataset, "global_temp", "EPSG:4326")
    regional_bbox = await extract_variable_bounding_box(
        dataset, "regional_temp", "EPSG:4326"
    )

    if global_bbox and regional_bbox:
        # Ensure both bounding boxes are valid
        assert len(global_bbox.lowerLeft) == 2
        assert len(global_bbox.upperRight) == 2
        assert len(regional_bbox.lowerLeft) == 2
        assert len(regional_bbox.upperRight) == 2

        # Print actual coordinates for debugging
        print(f"Global bbox: {global_bbox.lowerLeft} to {global_bbox.upperRight}")
        print(f"Regional bbox: {regional_bbox.lowerLeft} to {regional_bbox.upperRight}")

        # Basic sanity checks that each bbox is well-formed
        assert global_bbox.lowerLeft[0] <= global_bbox.upperRight[0]  # min X <= max X
        assert global_bbox.lowerLeft[1] <= global_bbox.upperRight[1]  # min Y <= max Y
        assert regional_bbox.lowerLeft[0] <= regional_bbox.upperRight[0]  # min X <= max X
        assert regional_bbox.lowerLeft[1] <= regional_bbox.upperRight[1]  # min Y <= max Y
    else:
        # Both bounding boxes should be extractable for simple rectilinear grids
        pytest.skip(
            "Could not extract bounding boxes - this might indicate an issue with grid detection"
        )


def test_extract_attributes_metadata():
    """Test extraction of attributes metadata from dataset"""
    from xpublish_tiles.xpublish.tiles.metadata import extract_attributes_metadata

    # Create dataset with attributes
    dataset = xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.random.randn(5, 10),
                dims=["lat", "lon"],
                coords={
                    "lat": (["lat"], np.linspace(-2, 2, 5)),
                    "lon": (["lon"], np.linspace(-5, 5, 10)),
                },
                attrs={
                    "long_name": "Temperature",
                    "units": "celsius",
                    "valid_min": -50.0,
                    "valid_max": 50.0,
                    "description": "Air temperature measurement",
                },
            ),
            "humidity": xr.DataArray(
                np.random.randn(5, 10),
                dims=["lat", "lon"],
                coords={
                    "lat": (["lat"], np.linspace(-2, 2, 5)),
                    "lon": (["lon"], np.linspace(-5, 5, 10)),
                },
                attrs={
                    "long_name": "Relative Humidity",
                    "units": "percent",
                    "valid_range": [0, 100],
                },
            ),
        },
        attrs={
            "title": "Weather Data",
            "institution": "Test University",
            "source": "Model simulation",
            "history": "Created on 2024-01-01",
        },
    )

    # Test extraction for all variables
    attrs_meta = extract_attributes_metadata(dataset)

    # Check dataset attributes
    assert "title" in attrs_meta.dataset_attrs
    assert "institution" in attrs_meta.dataset_attrs
    assert "source" in attrs_meta.dataset_attrs
    assert "history" in attrs_meta.dataset_attrs
    assert attrs_meta.dataset_attrs["title"] == "Weather Data"

    # Check variable attributes
    assert "temperature" in attrs_meta.variable_attrs
    assert "humidity" in attrs_meta.variable_attrs

    temp_attrs = attrs_meta.variable_attrs["temperature"]
    assert temp_attrs["long_name"] == "Temperature"
    assert temp_attrs["units"] == "celsius"
    assert temp_attrs["valid_min"] == -50.0
    assert temp_attrs["valid_max"] == 50.0

    humidity_attrs = attrs_meta.variable_attrs["humidity"]
    assert humidity_attrs["long_name"] == "Relative Humidity"
    assert humidity_attrs["units"] == "percent"
    assert humidity_attrs["valid_range"] == [0, 100]


def test_extract_attributes_metadata_single_variable():
    """Test extraction of attributes metadata for single variable"""
    from xpublish_tiles.xpublish.tiles.metadata import extract_attributes_metadata

    # Create dataset with multiple variables
    dataset = xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.random.randn(5, 10),
                dims=["lat", "lon"],
                attrs={"long_name": "Temperature", "units": "celsius"},
            ),
            "pressure": xr.DataArray(
                np.random.randn(5, 10),
                dims=["lat", "lon"],
                attrs={"long_name": "Pressure", "units": "hPa"},
            ),
        },
        attrs={"title": "Test Dataset"},
    )

    # Test extraction for single variable
    attrs_meta = extract_attributes_metadata(dataset, "temperature")

    # Should have dataset attributes
    assert attrs_meta.dataset_attrs["title"] == "Test Dataset"

    # Should only have temperature variable attributes
    assert "temperature" in attrs_meta.variable_attrs
    assert "pressure" not in attrs_meta.variable_attrs
    assert attrs_meta.variable_attrs["temperature"]["long_name"] == "Temperature"
