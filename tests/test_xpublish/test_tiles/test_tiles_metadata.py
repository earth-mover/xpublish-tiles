"""Tests for tiles metadata functionality"""

import numpy as np

import xarray as xr


def test_extract_dataset_extents():
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


def test_extract_dataset_extents_empty():
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

    extents = extract_dataset_extents(dataset, "temperature")
    assert len(extents) == 0


def test_extract_dataset_extents_multiple_variables():
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
    extents_surface = extract_dataset_extents(dataset, "surface_temp")
    assert len(extents_surface) == 1
    assert "time" in extents_surface

    # Test with ocean_temp variable (has time and depth)
    extents_ocean = extract_dataset_extents(dataset, "ocean_temp")
    assert len(extents_ocean) == 2
    assert "time" in extents_ocean
    assert "depth" in extents_ocean

    # Time should be from the ocean_temp variable
    time_extent = extents_ocean["time"]
    assert time_extent["interval"][0] == "2023-01-01T00:00:00Z"
    assert time_extent["interval"][1] == "2023-01-02T00:00:00Z"

    # Depth should be from the ocean_temp variable
    depth_extent = extents_ocean["depth"]
    assert depth_extent["interval"] == [0.0, 10.0]
    assert depth_extent["units"] == "m"


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

    # Test 30-second resolution
    second_values = [
        "2023-01-01T00:00:00Z",
        "2023-01-01T00:00:30Z",
        "2023-01-01T00:01:00Z",
    ]
    assert _calculate_temporal_resolution(second_values) == "PT30S"


def test_calculate_temporal_resolution_edge_cases():
    """Test _calculate_temporal_resolution with edge cases"""
    from xpublish_tiles.xpublish.tiles.metadata import _calculate_temporal_resolution

    # Test edge cases
    assert _calculate_temporal_resolution([]) == "PT1H"  # Empty list
    assert (
        _calculate_temporal_resolution(["2023-01-01T00:00:00Z"]) == "PT1H"
    )  # Single value
    assert _calculate_temporal_resolution([1, 2, 3]) == "PT1H"  # Non-string values

    # Test irregular intervals (should use average)
    irregular_values = [
        "2023-01-01T00:00:00Z",
        "2023-01-01T01:00:00Z",  # 1 hour gap
        "2023-01-01T04:00:00Z",  # 3 hour gap
    ]
    result = _calculate_temporal_resolution(irregular_values)
    assert result == "PT2H"  # Average of 1 and 3 hours

    # Test with invalid datetime strings (should fallback)
    invalid_values = ["not-a-date", "also-not-a-date"]
    assert _calculate_temporal_resolution(invalid_values) == "PT1H"


def test_create_tileset_metadata_with_extents():
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
    extents = extract_dataset_extents(dataset, "temperature")
    assert "time" in extents

    time_extent = extents["time"]
    assert "interval" in time_extent
    assert "resolution" in time_extent
    assert time_extent["resolution"] == "PT6H"  # 6-hourly


def test_create_tileset_metadata_no_extents():
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
    extents = extract_dataset_extents(dataset, "temperature")
    assert len(extents) == 0
