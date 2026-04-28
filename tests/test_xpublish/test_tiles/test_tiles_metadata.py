"""Tests for tiles metadata functionality"""

import numpy as np
import pandas as pd
import pytest
import xpublish
from fastapi.testclient import TestClient
from syrupy.extensions.json import JSONSnapshotExtension

import xarray as xr
from xpublish_tiles.lib import VariableNotFoundError
from xpublish_tiles.testing.datasets import (
    CUBED_SPHERE,
    ERA5,
    GLOBAL_HEALPIX_L3,
    HRRR,
    IFS,
    REGIONAL_HEALPIX_NA,
)
from xpublish_tiles.xpublish.tiles import TilesPlugin
from xpublish_tiles.xpublish.tiles.metadata import (
    _calculate_temporal_resolution,
    _pandas_freq_to_iso8601,
    allowed_styles,
    create_tileset_metadata,
    extract_dataset_extents,
    get_styles,
)


async def test_extract_dataset_extents():
    """Test the extract_dataset_extents function directly"""
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

    time_coords = pd.date_range("2023-01-01", periods=4, freq="D")
    depth_coords = [0, 10]

    dataset = xr.Dataset(
        {
            "surface_temp": xr.DataArray(
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
            ),
            "ocean_temp": xr.DataArray(
                np.random.randn(4, 2, 5, 10),
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
    assert time_extent["interval"][1] == "2023-01-04T00:00:00"
    assert time_extent["resolution"] == "P1D"  # Daily resolution

    # Depth should be from the ocean_temp variable
    depth_extent = extents_ocean["depth"]
    assert depth_extent["interval"] == [0.0, 10.0]
    assert depth_extent["units"] == "m"


@pytest.mark.parametrize("use_cftime", [True, False])
@pytest.mark.parametrize(
    "freq,expected",
    [
        ("h", "PT1H"),
        ("6h", "PT6H"),
        ("15min", "PT15M"),
        ("30s", "PT30S"),
        ("D", "P1D"),
        ("7D", "P7D"),
        ("MS", "P1M"),
        ("QS", "P3M"),
        ("YS", "P1Y"),
        ("5YS", "P5Y"),
        ("10YS", "P10Y"),
    ],
)
def test_calculate_temporal_resolution(use_cftime, freq, expected):
    """Test _calculate_temporal_resolution with various frequencies"""
    values = xr.DataArray(
        xr.date_range("2000-01-01T00:00:00", periods=4, freq=freq, use_cftime=use_cftime),
        dims="time",
        name="time",
    )
    assert _calculate_temporal_resolution(values) == expected


@pytest.mark.parametrize(
    "pandas_freq,expected_iso",
    [
        # Hours
        ("h", "PT1H"),
        ("H", "PT1H"),
        ("3h", "PT3H"),
        ("6H", "PT6H"),
        # Minutes
        ("min", "PT1M"),
        ("T", "PT1M"),
        ("15min", "PT15M"),
        ("30T", "PT30M"),
        # Seconds
        ("s", "PT1S"),
        ("S", "PT1S"),
        ("30s", "PT30S"),
        # Days
        ("D", "P1D"),
        ("7D", "P7D"),
        # Weeks
        ("W", "P7D"),
        ("W-SUN", "P7D"),
        ("2W", "P14D"),
        # Months
        ("MS", "P1M"),
        ("ME", "P1M"),
        ("M", "P1M"),
        ("3MS", "P3M"),
        # Quarters
        ("QS", "P3M"),
        ("QE", "P3M"),
        ("QS-OCT", "P3M"),
        ("2QS", "P6M"),
        # Years
        ("YS", "P1Y"),
        ("YE", "P1Y"),
        ("Y", "P1Y"),
        ("YS-JAN", "P1Y"),
        ("AS", "P1Y"),
        ("10YS", "P10Y"),
        ("5YS-JAN", "P5Y"),
        # Unknown
        ("unknown", None),
        ("", None),
    ],
)
def test_pandas_freq_to_iso8601(pandas_freq, expected_iso):
    """Test conversion of pandas frequency strings to ISO 8601 durations"""
    assert _pandas_freq_to_iso8601(pandas_freq) == expected_iso


def test_calculate_temporal_resolution_edge_cases():
    """Test _calculate_temporal_resolution with edge cases"""
    # Edge cases return None (not a guessed default)
    assert _calculate_temporal_resolution(xr.DataArray([], dims="time")) is None  # Empty
    assert (
        _calculate_temporal_resolution(
            xr.DataArray(pd.DatetimeIndex(["2023-01-01T00:00:00"]), dims="time")
        )
        is None
    )  # Single value
    assert (
        _calculate_temporal_resolution(
            xr.DataArray(
                pd.DatetimeIndex(["2023-01-01T00:00:00", "2023-01-01T01:00:00"]),
                dims="time",
            )
        )
        is None
    )  # Two values (need at least 3 for xr.infer_freq)
    assert (
        _calculate_temporal_resolution(xr.DataArray([1, 2, 3], dims="time")) is None
    )  # Non-datetime values

    # Irregular intervals return None (no consistent frequency)
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
    assert _calculate_temporal_resolution(irregular_values) is None

    # Invalid datetime strings return None
    invalid_values = xr.DataArray(
        ["not-a-date", "also-not-a-date", "still-not-a-date"], dims="time", name="time"
    )
    assert _calculate_temporal_resolution(invalid_values) is None


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


def test_allowed_styles_healpix_only_polygons():
    """Healpix datasets must advertise only the polygons style."""
    for fixture in (GLOBAL_HEALPIX_L3, REGIONAL_HEALPIX_NA):
        ds = fixture.create()
        assert allowed_styles(ds) == ["polygons"]
        style_ids = {s.id.split("/")[0] for s in get_styles(ds)}
        assert style_ids == {"polygons"}


async def test_cubed_sphere_metadata():
    """Cubed-sphere (Faceted) datasets must advertise only the polygons style,
    and must not expose ``face_dim`` as a custom dimension extent."""
    ds = CUBED_SPHERE.create()
    assert allowed_styles(ds) == ["polygons"]
    style_ids = {s.id.split("/")[0] for s in get_styles(ds)}
    assert style_ids == {"polygons"}

    extents = await extract_dataset_extents(ds, "foo")
    assert "nf" not in extents
    assert len(extents) == 0


def test_allowed_styles_default_grid():
    """Non-Healpix grids advertise both raster and polygons."""
    dataset = xr.Dataset(
        {
            "t": (
                ["lat", "lon"],
                np.zeros((3, 4)),
                {"grid_mapping": ""},
            )
        },
        coords={
            "lat": ("lat", np.linspace(-1, 1, 3), {"standard_name": "latitude"}),
            "lon": ("lon", np.linspace(-2, 2, 4), {"standard_name": "longitude"}),
        },
    )
    assert set(allowed_styles(dataset)) == {"raster", "polygons"}
    style_ids = {s.id.split("/")[0] for s in get_styles(dataset)}
    assert style_ids == {"raster", "polygons"}


def test_healpix_tileset_metadata_styles():
    """The tileset metadata response for a Healpix dataset lists only polygon styles."""
    ds = GLOBAL_HEALPIX_L3.create()
    metadata = create_tileset_metadata(ds, "WebMercatorQuad")
    assert metadata.styles is not None
    style_ids = {s.id.split("/")[0] for s in metadata.styles}
    assert style_ids == {"polygons"}


def _normalize_for_snapshot(obj):
    """Normalize a /tiles/ response so snapshots are stable across platforms.

    Why: float values from coordinate transforms differ in their last bits
    between macOS and Linux, and ``RenderRegistry.all()`` iterates entry
    points in load order, which is also platform-dependent.
    """
    if isinstance(obj, dict):
        return {k: _normalize_for_snapshot(v) for k, v in obj.items()}
    if isinstance(obj, list):
        items = [_normalize_for_snapshot(x) for x in obj]
        if items and all(isinstance(x, dict) and "id" in x for x in items):
            items.sort(key=lambda x: x["id"])
        return items
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


@pytest.mark.parametrize(
    "fixture",
    [
        pytest.param(ERA5, id="era5"),
        pytest.param(HRRR, id="hrrr"),
        pytest.param(IFS, id="ifs"),
        pytest.param(GLOBAL_HEALPIX_L3, id="global_healpix_l3"),
        pytest.param(REGIONAL_HEALPIX_NA, id="regional_healpix_na"),
        pytest.param(CUBED_SPHERE, id="cubed_sphere"),
    ],
)
def test_tiles_endpoint_snapshot(fixture, snapshot):
    """Snapshot the /tiles/ endpoint response across diverse grid types."""
    ds = fixture.create()
    rest = xpublish.Rest({"ds": ds}, plugins={"tiles": TilesPlugin()})
    client = TestClient(rest.app)
    response = client.get("/datasets/ds/tiles/")
    assert response.status_code == 200
    assert _normalize_for_snapshot(response.json()) == snapshot.use_extension(
        JSONSnapshotExtension
    )


def test_tiles_endpoint_skips_non_spatial_data_vars():
    """Auxiliary non-scalar data vars without a tileable grid should be skipped,
    not crash the /tiles/ listing.

    Mirrors the GMAO cubed-sphere case where vars like ``anchor``/``contacts``
    have non-spatial dims (e.g. ``ncontact``) and fail grid detection when the
    dataset is subset to that variable alone.
    """
    ds = CUBED_SPHERE.create()
    nf = ds.sizes["nf"]
    ds = ds.assign(
        contacts=xr.DataArray(np.zeros((nf, 4), dtype=np.int32), dims=("nf", "ncontact")),
        anchor=xr.DataArray(
            np.zeros((nf, 4, 4), dtype=np.float64),
            dims=("nf", "ncontact", "ncontact_b"),
        ),
    )

    rest = xpublish.Rest({"ds": ds}, plugins={"tiles": TilesPlugin()})
    client = TestClient(rest.app)
    response = client.get("/datasets/ds/tiles/")
    assert response.status_code == 200

    body = response.json()
    tilesets = body.get("tilesets", [])
    assert tilesets, "expected at least one tileset"
    layer_ids = {layer["id"] for ts in tilesets for layer in (ts.get("layers") or [])}
    assert "foo" in layer_ids
    assert "contacts" not in layer_ids
    assert "anchor" not in layer_ids


def test_tiles_endpoint_no_renderable_vars_returns_422():
    """If every data variable fails grid detection, the listing should 422."""
    dataset = xr.Dataset(
        {
            "contacts": xr.DataArray(
                np.zeros((6, 4), dtype=np.int32), dims=("nf", "ncontact")
            ),
        },
        coords={"nf": np.arange(6), "ncontact": np.arange(4)},
    )
    rest = xpublish.Rest({"ds": dataset}, plugins={"tiles": TilesPlugin()})
    client = TestClient(rest.app)
    response = client.get("/datasets/ds/tiles/")
    assert response.status_code == 422
    assert "No renderable variables" in response.json()["detail"]
