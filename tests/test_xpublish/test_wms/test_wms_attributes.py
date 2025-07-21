"""Tests for custom attributes from datasets in WMS metadata"""

import numpy as np

import xarray as xr


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
