"""Tests for attributes metadata functionality"""

import numpy as np

import xarray as xr


def test_filter_sensitive_attributes():
    """Test that sensitive attributes are filtered out"""
    from xpublish_tiles.xpublish.tiles.metadata import filter_sensitive_attributes

    attrs = {
        "title": "Test Dataset",
        "author": "Test Author",
        "password": "secret123",
        "api_token": "abc123",
        "private_key": "xyz789",
        "_private_info": "hidden",
        "description": "A test dataset",
    }

    filtered = filter_sensitive_attributes(attrs)

    # Should keep normal attributes
    assert "title" in filtered
    assert "author" in filtered
    assert "description" in filtered

    # Should remove sensitive attributes
    assert "password" not in filtered
    assert "api_token" not in filtered
    assert "private_key" not in filtered
    assert "_private_info" not in filtered


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


def test_create_tileset_metadata_with_attributes():
    """Test that tileset metadata includes attributes"""
    from xpublish_tiles.xpublish.tiles.metadata import create_tileset_metadata

    # Create dataset with attributes
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
                attrs={
                    "long_name": "Air Temperature",
                    "units": "K",
                    "standard_name": "air_temperature",
                },
            )
        },
        attrs={
            "title": "Test Weather Dataset",
            "institution": "Test Lab",
            "Conventions": "CF-1.8",
        },
    )

    metadata = create_tileset_metadata(dataset, "WebMercatorQuad")

    # Check that attributes are present
    assert metadata.attributes is not None

    # Check dataset attributes
    assert "title" in metadata.attributes.dataset_attrs
    assert "institution" in metadata.attributes.dataset_attrs
    assert "Conventions" in metadata.attributes.dataset_attrs
    assert metadata.attributes.dataset_attrs["title"] == "Test Weather Dataset"

    # Check variable attributes
    assert "temperature" in metadata.attributes.variable_attrs
    temp_attrs = metadata.attributes.variable_attrs["temperature"]
    assert temp_attrs["long_name"] == "Air Temperature"
    assert temp_attrs["units"] == "K"
    assert temp_attrs["standard_name"] == "air_temperature"


def test_wms_convert_attributes_to_wms():
    """Test conversion of attributes to WMS format"""
    from xpublish_tiles.xpublish.wms.utils import convert_attributes_to_wms

    attrs = {
        "title": "Test Dataset",
        "version": 1.2,
        "active": True,
        "keywords": ["weather", "temperature"],
        "complex_obj": {"nested": "value"},
    }

    wms_attrs = convert_attributes_to_wms(attrs)

    # Should have one WMS attribute for each input attribute
    assert len(wms_attrs) == 5

    # Find attributes by name
    attr_dict = {attr.name: attr.value for attr in wms_attrs}

    assert attr_dict["title"] == "Test Dataset"
    assert attr_dict["version"] == "1.2"
    assert attr_dict["active"] == "true"  # Boolean converted to lowercase string
    assert (
        attr_dict["keywords"] == "weather, temperature"
    )  # List converted to comma-separated
    assert "nested" in attr_dict["complex_obj"]  # Complex object converted to string


def test_wms_layers_include_attributes():
    """Test that WMS layers include variable attributes"""
    from xpublish_tiles.xpublish.wms.utils import extract_layers

    # Create dataset with attributes
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
                attrs={
                    "long_name": "Air Temperature",
                    "units": "celsius",
                    "valid_min": -50,
                    "valid_max": 50,
                },
            )
        }
    )

    layers = extract_layers(dataset, "http://example.com")

    # Should have one layer
    assert len(layers) == 1

    layer = layers[0]
    assert layer.name == "temperature"

    # Check that attributes are present
    assert len(layer.attributes) > 0

    # Convert to dict for easier checking
    attr_dict = {attr.name: attr.value for attr in layer.attributes}

    assert attr_dict["long_name"] == "Air Temperature"
    assert attr_dict["units"] == "celsius"
    assert attr_dict["valid_min"] == "-50"
    assert attr_dict["valid_max"] == "50"


def test_wms_capabilities_include_dataset_attributes():
    """Test that WMS capabilities include dataset attributes in root layer"""
    from xpublish_tiles.xpublish.wms.utils import create_capabilities_response

    # Create dataset with attributes
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
                attrs={"long_name": "Air Temperature"},
            )
        },
        attrs={
            "title": "Weather Dataset",
            "institution": "Test Institution",
            "source": "Model run",
        },
    )

    capabilities = create_capabilities_response(dataset, "http://example.com")

    # Check that root layer has dataset attributes
    root_layer = capabilities.capability.layer
    assert len(root_layer.attributes) > 0

    # Convert to dict for easier checking
    attr_dict = {attr.name: attr.value for attr in root_layer.attributes}

    assert attr_dict["title"] == "Weather Dataset"
    assert attr_dict["institution"] == "Test Institution"
    assert attr_dict["source"] == "Model run"
