"""Utilities for WMS dataset introspection and metadata extraction"""

from typing import Any

import numpy as np

import xarray as xr
from xpublish_tiles.grids import guess_grid_system
from xpublish_tiles.pipeline import transformer_from_crs
from xpublish_tiles.xpublish.wms.types import (
    WMSAttributeResponse,
    WMSBoundingBoxResponse,
    WMSCapabilitiesResponse,
    WMSCapabilityResponse,
    WMSDCPTypeResponse,
    WMSDimensionResponse,
    WMSFormatResponse,
    WMSGetCapabilitiesOperationResponse,
    WMSGetMapOperationResponse,
    WMSHTTPResponse,
    WMSLayerResponse,
    WMSOnlineResourceResponse,
    WMSRequestResponse,
    WMSServiceResponse,
    WMSStyleResponse,
)


def convert_attributes_to_wms(attrs: dict[str, Any]) -> list[WMSAttributeResponse]:
    """Convert xarray attributes to WMS attribute elements

    Args:
        attrs: Dictionary of attributes to convert

    Returns:
        List of WMSAttributeResponse objects
    """
    wms_attrs = []
    for name, value in attrs.items():
        # Convert value to string representation
        if isinstance(value, str):
            str_value = value
        elif isinstance(value, bool):
            str_value = str(value).lower()
        elif isinstance(value, int | float):
            str_value = str(value)
        elif isinstance(value, list | tuple):
            str_value = ", ".join(str(v) for v in value)
        else:
            str_value = str(value)

        wms_attrs.append(WMSAttributeResponse(name=name, value=str_value))

    return wms_attrs


def extract_dimensions(dataset: xr.Dataset) -> list[WMSDimensionResponse]:
    """Extract all dimensions from dataset coordinates.

    Returns:
        List of WMSDimensionResponse objects for all non-spatial dimensions
    """
    dimensions = []

    # Skip spatial coordinates (x, y, lon, lat)
    spatial_coords = {"x", "y", "lon", "lat", "longitude", "latitude"}

    for coord_name, coord in dataset.coords.items():
        coord_name_str = str(coord_name)
        if coord_name_str.lower() in spatial_coords:
            continue

        # Extract dimension metadata
        units = getattr(coord, "units", "")

        # Handle different dimension types
        if coord_name_str.lower() in ["time", "t"]:
            # Time dimension
            if hasattr(coord, "values"):
                if np.issubdtype(coord.dtype, np.timedelta64):
                    # Convert timedelta64 to strings
                    values = ",".join(str(v) for v in coord.values)
                    default = str(coord.values[-1]) if len(coord.values) > 0 else None
                elif np.issubdtype(coord.dtype, np.datetime64):
                    # Convert datetime64 to ISO strings
                    times = [np.datetime_as_string(t, unit="s") for t in coord.values]
                    values = ",".join(times)
                    default = times[-1] if times else None
                else:
                    values = ",".join(str(v) for v in coord.values)
                    default = str(coord.values[-1]) if len(coord.values) > 0 else None
            else:
                values = ""
                default = None

            dimensions.append(
                WMSDimensionResponse(
                    name="time",
                    units=units or "ISO8601",
                    default=default,
                    values=values,
                    multiple_values=True,
                    nearest_value=True,
                )
            )

        elif coord_name_str.lower() in ["elevation", "z", "depth", "height", "level"]:
            # Elevation/vertical dimension
            if hasattr(coord, "values"):
                values = ",".join(str(float(v)) for v in coord.values)
                default = str(float(coord.values[0])) if len(coord.values) > 0 else None
            else:
                values = ""
                default = None

            dimensions.append(
                WMSDimensionResponse(
                    name=coord_name_str.lower(),
                    units=units or "m",
                    default=default,
                    values=values,
                    multiple_values=True,
                    nearest_value=True,
                )
            )

        else:
            # Arbitrary dimension
            if hasattr(coord, "values"):
                # Handle different data types
                if np.issubdtype(coord.dtype, np.timedelta64):
                    # convert timedelta64 to strings
                    values = ",".join(str(t) for t in coord.values)
                    default = str(coord.values[-1]) if len(coord.values) > 0 else None
                elif np.issubdtype(coord.dtype, np.datetime64):
                    values = ",".join(
                        np.datetime_as_string(t, unit="s") for t in coord.values
                    )
                    default = (
                        np.datetime_as_string(coord.values[-1], unit="s")
                        if len(coord.values) > 0
                        else None
                    )
                elif np.issubdtype(coord.dtype, np.number):
                    values = ",".join(str(float(v)) for v in coord.values)
                    default = (
                        str(float(coord.values[-1])) if len(coord.values) > 0 else None
                    )
                else:
                    values = ",".join(str(v) for v in coord.values)
                    default = str(coord.values[-1]) if len(coord.values) > 0 else None
            else:
                values = ""
                default = None

            dimensions.append(
                WMSDimensionResponse(
                    name=coord_name_str,
                    units=units,
                    default=default,
                    values=values,
                    multiple_values=True,
                    nearest_value=True,
                )
            )

    return dimensions


def get_available_wms_styles() -> list[WMSStyleResponse]:
    """Get all available styles from registered renderers."""
    from xpublish_tiles.render import RenderRegistry

    styles = []

    for renderer_cls in RenderRegistry.all().values():
        # Add default variant alias
        default_variant = renderer_cls.default_variant()
        default_style_info = renderer_cls.describe_style("default")
        default_style_info["title"] = (
            f"{renderer_cls.style_id().title()} - Default ({default_variant.title()})"
        )
        default_style_info["description"] = (
            f"Default {renderer_cls.style_id()} rendering (alias for {default_variant})"
        )
        styles.append(
            WMSStyleResponse(
                name=default_style_info["id"],
                title=default_style_info["title"],
                abstract=default_style_info["description"],
            )
        )

        # Add all actual variants
        for variant in renderer_cls.supported_variants():
            style_info = renderer_cls.describe_style(variant)
            styles.append(
                WMSStyleResponse(
                    name=style_info["id"],
                    title=style_info["title"],
                    abstract=style_info["description"],
                )
            )

    return styles


def extract_layers(dataset: xr.Dataset, base_url: str) -> list[WMSLayerResponse]:
    """Extract layer information from dataset data variables.

    Args:
        dataset: xarray Dataset
        base_url: Base URL for the service

    Returns:
        List of WMSLayerResponse objects for each data variable
    """
    layers = []

    # Extract dimensions
    dimensions = extract_dimensions(dataset)

    for var_name, var in dataset.data_vars.items():
        # Extract variable metadata
        title = getattr(var, "long_name", var_name)
        abstract = getattr(var, "description", getattr(var, "comment", None))

        # Extract variable attributes
        wms_attributes = convert_attributes_to_wms(var.attrs)

        # Extract geographic bounds
        grid = guess_grid_system(dataset, var_name)
        supported_crs = ["EPSG:4326", "EPSG:3857"]
        supported_bounds = []
        bounding_boxes = []

        for crs in supported_crs:
            transformer = transformer_from_crs(crs_from=grid.crs, crs_to=crs)
            bounds = transformer.transform_bounds(
                grid.bbox.west, grid.bbox.south, grid.bbox.east, grid.bbox.north
            )
            supported_bounds.append(bounds)

        supported_crs.append(grid.crs.to_string())
        bounding_boxes = [
            WMSBoundingBoxResponse(
                crs=grid.crs.to_string(),
                minx=grid.bbox.west,
                miny=grid.bbox.south,
                maxx=grid.bbox.east,
                maxy=grid.bbox.north,
            )
        ]

        layer = WMSLayerResponse(
            name=var_name,
            title=title,
            abstract=abstract,
            crs=supported_crs,
            bounding_box=bounding_boxes,
            dimensions=dimensions,
            attributes=wms_attributes,
            styles=[],  # Styles inherited from root layer
            queryable=True,
            opaque=False,
        )
        layers.append(layer)

    return layers


def create_capabilities_response(
    dataset: xr.Dataset,
    base_url: str,
    version: str = "1.3.0",
    service_title: str = "XPublish WMS Service",
    service_abstract: str | None = None,
) -> WMSCapabilitiesResponse:
    """Create a complete WMS GetCapabilities response from a dataset.

    Args:
        dataset: xarray Dataset
        base_url: Base URL for the service
        version: WMS version (default: "1.3.0")
        service_title: Title for the service
        service_abstract: Abstract description of the service

    Returns:
        WMSCapabilitiesResponse object
    """
    # Create service information
    online_resource = WMSOnlineResourceResponse(href=base_url)

    service = WMSServiceResponse(
        name="WMS",
        title=service_title,
        abstract=service_abstract,
        online_resource=online_resource,
        fees="none",
        access_constraints="none",
    )

    # Create DCP Type for all operations
    dcp_type = WMSDCPTypeResponse(
        http=WMSHTTPResponse(get=WMSOnlineResourceResponse(href=base_url))
    )

    # Create request information
    request = WMSRequestResponse(
        get_capabilities=WMSGetCapabilitiesOperationResponse(
            formats=[
                WMSFormatResponse(format="text/xml"),
                WMSFormatResponse(format="application/json"),
            ],
            dcp_type=dcp_type,
        ),
        get_map=WMSGetMapOperationResponse(
            formats=[
                WMSFormatResponse(format="image/png"),
                WMSFormatResponse(format="image/jpeg"),
            ],
            dcp_type=dcp_type,
        ),
    )

    # Extract layers from dataset
    layers = extract_layers(dataset, base_url)

    # Create root layer containing all data layers and styles
    available_styles = get_available_wms_styles()

    # Extract dataset attributes for root layer
    dataset_wms_attributes = convert_attributes_to_wms(dataset.attrs)

    root_layer = WMSLayerResponse(
        title="Dataset Layers",
        abstract="All available data layers with raster visualization styles",
        layers=layers,
        attributes=dataset_wms_attributes,
        styles=available_styles,  # All styles defined at root level
        queryable=False,
    )

    # Create capability information
    capability = WMSCapabilityResponse(
        request=request, exception=["XML", "INIMAGE", "BLANK"], layer=root_layer
    )

    # Create complete capabilities response
    capabilities = WMSCapabilitiesResponse(
        version=version, service=service, capability=capability
    )

    return capabilities
