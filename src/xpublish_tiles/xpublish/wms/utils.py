"""Utilities for WMS dataset introspection and metadata extraction"""

import numpy as np

import xarray as xr
from xpublish_tiles.utils import get_available_raster_styles
from xpublish_tiles.xpublish.wms.types import (
    WMSBoundingBoxResponse,
    WMSCapabilitiesResponse,
    WMSCapabilityResponse,
    WMSDCPTypeResponse,
    WMSDimensionResponse,
    WMSFormatResponse,
    WMSGeographicBoundingBoxResponse,
    WMSGetCapabilitiesOperationResponse,
    WMSGetMapOperationResponse,
    WMSHTTPResponse,
    WMSLayerResponse,
    WMSOnlineResourceResponse,
    WMSRequestResponse,
    WMSServiceResponse,
    WMSStyleResponse,
)


def extract_geographic_bounds(dataset: xr.Dataset) -> tuple[float, float, float, float]:
    """Extract geographic bounding box from dataset coordinates.

    Returns:
        Tuple of (west, east, south, north) in geographic coordinates
    """
    # Try to find longitude/latitude coordinates
    lon_coord = None
    lat_coord = None

    for coord_name, coord in dataset.coords.items():
        if hasattr(coord, "standard_name"):
            if coord.standard_name in ["longitude", "lon"]:
                lon_coord = coord
            elif coord.standard_name in ["latitude", "lat"]:
                lat_coord = coord
        elif str(coord_name).lower() in ["lon", "longitude", "x"]:
            lon_coord = coord
        elif str(coord_name).lower() in ["lat", "latitude", "y"]:
            lat_coord = coord

    if lon_coord is None or lat_coord is None:
        # Fallback to first two coordinates if no standard names found
        coords = list(dataset.coords.values())
        if len(coords) >= 2:
            lon_coord = coords[0]
            lat_coord = coords[1]
        else:
            # Default bounds if no coordinates found
            return -180.0, 180.0, -90.0, 90.0

    west = float(lon_coord.min().values)
    east = float(lon_coord.max().values)
    south = float(lat_coord.min().values)
    north = float(lat_coord.max().values)

    return west, east, south, north


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
                if np.issubdtype(coord.dtype, np.datetime64):
                    # Convert datetime64 to ISO strings
                    times = [np.datetime_as_string(t, unit="s") for t in coord.values]
                    values = ",".join(times)
                    default = times[0] if times else None
                else:
                    values = ",".join(str(v) for v in coord.values)
                    default = str(coord.values[0]) if len(coord.values) > 0 else None
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
                if np.issubdtype(coord.dtype, np.datetime64):
                    values = ",".join(
                        np.datetime_as_string(t, unit="s") for t in coord.values
                    )
                    default = (
                        np.datetime_as_string(coord.values[0], unit="s")
                        if len(coord.values) > 0
                        else None
                    )
                elif np.issubdtype(coord.dtype, np.number):
                    values = ",".join(str(float(v)) for v in coord.values)
                    default = (
                        str(float(coord.values[0])) if len(coord.values) > 0 else None
                    )
                else:
                    values = ",".join(str(v) for v in coord.values)
                    default = str(coord.values[0]) if len(coord.values) > 0 else None
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
    """Get all available raster styles based on matplotlib colormaps."""
    style_dicts = get_available_raster_styles()

    return [
        WMSStyleResponse(
            name=style["id"],
            title=style["title"],
            abstract=style["description"],
        )
        for style in style_dicts
    ]


def extract_layers(dataset: xr.Dataset, base_url: str) -> list[WMSLayerResponse]:
    """Extract layer information from dataset data variables.

    Args:
        dataset: xarray Dataset
        base_url: Base URL for the service

    Returns:
        List of WMSLayerResponse objects for each data variable
    """
    layers = []

    # Get geographic bounds
    west, east, south, north = extract_geographic_bounds(dataset)

    # Create geographic bounding box
    geo_bbox = WMSGeographicBoundingBoxResponse(
        west_bound_longitude=west,
        east_bound_longitude=east,
        south_bound_latitude=south,
        north_bound_latitude=north,
    )

    # Create bounding boxes for different CRS
    bounding_boxes = [
        WMSBoundingBoxResponse(
            crs="EPSG:4326", minx=west, miny=south, maxx=east, maxy=north
        ),
        WMSBoundingBoxResponse(
            crs="EPSG:3857",
            minx=west * 111319.49,  # Rough conversion to Web Mercator
            miny=south * 111319.49,
            maxx=east * 111319.49,
            maxy=north * 111319.49,
        ),
    ]

    # Extract dimensions
    dimensions = extract_dimensions(dataset)

    for var_name, var in dataset.data_vars.items():
        # Extract variable metadata
        title = getattr(var, "long_name", var_name)
        abstract = getattr(var, "description", getattr(var, "comment", None))

        layer = WMSLayerResponse(
            name=var_name,
            title=title,
            abstract=abstract,
            crs=["EPSG:4326", "EPSG:3857"],
            ex_geographic_bounding_box=geo_bbox,
            bounding_box=bounding_boxes,
            dimensions=dimensions,
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
    west, east, south, north = extract_geographic_bounds(dataset)
    available_styles = get_available_wms_styles()

    root_layer = WMSLayerResponse(
        title="Dataset Layers",
        abstract="All available data layers with raster visualization styles",
        crs=["EPSG:4326", "EPSG:3857"],
        ex_geographic_bounding_box=WMSGeographicBoundingBoxResponse(
            west_bound_longitude=west,
            east_bound_longitude=east,
            south_bound_latitude=south,
            north_bound_latitude=north,
        ),
        layers=layers,
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
