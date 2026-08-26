"""Utilities for WMS dataset introspection and metadata extraction"""

import math
from typing import Any
from urllib.parse import quote

import numpy as np
from pyproj import CRS

import xarray as xr
from xpublish_tiles.grids import GridSystem, detect_grids
from xpublish_tiles.utils import normalize_tilejson_bounds
from xpublish_tiles.xpublish.wms.types import (
    WMSAttributeResponse,
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
    WMSLegendURLResponse,
    WMSOnlineResourceResponse,
    WMSRequestResponse,
    WMSServiceResponse,
    WMSStyleResponse,
    crs_is_north_first,
)

LEGEND_WIDTH = 100
LEGEND_HEIGHT = 100

CRS84 = CRS.from_user_input("OGC:CRS84")
EPSG4326 = CRS.from_epsg(4326)
EPSG3857 = CRS.from_epsg(3857)


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


def extract_dimensions(
    dataset: xr.Dataset, var_dims: set[str] | None = None
) -> list[WMSDimensionResponse]:
    """Extract all dimensions from dataset coordinates.

    Args:
        dataset: xarray Dataset
        var_dims: When given, only include dimensions in this set so layers
            don't advertise dimensions their variable doesn't have

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
        if var_dims is not None and coord_name_str not in var_dims:
            continue
        # Skip scalar coordinates (e.g., radar site latitude/longitude/altitude)
        if coord.ndim == 0:
            continue
        # Skip auxiliary coordinates that aren't indexable dimensions
        # (e.g., elevation(azimuth) in radar data — varies per ray, not a selectable WMS dimension)
        if coord_name_str not in dataset.dims:
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


def _style_response(
    style_info: dict[str, str], base_url: str | None, layer_name: str | None
) -> WMSStyleResponse:
    legend_url = None
    if base_url is not None and layer_name is not None:
        href = (
            f"{base_url}?service=WMS&version=1.3.0&request=GetLegendGraphic"
            f"&layer={quote(layer_name)}&styles={quote(style_info['id'])}"
            f"&width={LEGEND_WIDTH}&height={LEGEND_HEIGHT}&format=image/png"
        )
        legend_url = WMSLegendURLResponse(
            width=LEGEND_WIDTH,
            height=LEGEND_HEIGHT,
            format="image/png",
            online_resource=WMSOnlineResourceResponse(href=href),
        )
    return WMSStyleResponse(
        name=style_info["id"],
        title=style_info["title"],
        abstract=style_info["description"],
        legend_url=legend_url,
    )


def get_available_wms_styles(
    dataset: xr.Dataset | None = None,
    *,
    base_url: str | None = None,
    layer_name: str | None = None,
) -> list[WMSStyleResponse]:
    """Get all available styles from registered renderers, filtered for ``dataset``'s grid.

    When ``base_url`` and ``layer_name`` are given, each style carries a
    LegendURL pointing at GetLegendGraphic for that layer.
    """
    from xpublish_tiles.render import RenderRegistry
    from xpublish_tiles.xpublish.tiles.metadata import allowed_styles

    allowed = set(allowed_styles(dataset))
    styles = []

    for renderer_cls in RenderRegistry.all().values():
        if renderer_cls.style_id() not in allowed:
            continue
        # Add default variant alias
        default_variant = renderer_cls.default_variant()
        default_style_info = renderer_cls.describe_style("default")
        default_style_info["title"] = (
            f"{renderer_cls.style_id().title()} - Default ({default_variant.title()})"
        )
        default_style_info["description"] = (
            f"Default {renderer_cls.style_id()} rendering (alias for {default_variant})"
        )
        styles.append(_style_response(default_style_info, base_url, layer_name))

        # Add all actual variants
        for variant in renderer_cls.supported_variants():
            style_info = renderer_cls.describe_style(variant)
            styles.append(_style_response(style_info, base_url, layer_name))

    return styles


def _geographic_bounds(
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    west, south, east, north = bounds
    west = west if math.isfinite(west) else -180.0
    south = south if math.isfinite(south) else -90.0
    east = east if math.isfinite(east) else 180.0
    north = north if math.isfinite(north) else 90.0
    # 0..360-convention grids keep their native longitudes through the
    # identity transform to EPSG:4326; WMS wants [-180, 180]
    west, south, east, north = normalize_tilejson_bounds((west, south, east, north))
    return west, max(south, -90.0), east, min(north, 90.0)


def _bounding_box_response(
    crs_id: str, crs: CRS, bounds: tuple[float, float, float, float]
) -> WMSBoundingBoxResponse:
    west, south, east, north = bounds
    if crs_is_north_first(crs):
        return WMSBoundingBoxResponse(
            crs=crs_id, minx=south, miny=west, maxx=north, maxy=east
        )
    return WMSBoundingBoxResponse(
        crs=crs_id, minx=west, miny=south, maxx=east, maxy=north
    )


def _layer_bounds(
    grid: GridSystem,
) -> tuple[list[str], WMSGeographicBoundingBoxResponse, list[WMSBoundingBoxResponse]]:
    geo_bounds = _geographic_bounds(grid.transform_bbox("EPSG:4326"))
    west, south, east, north = geo_bounds

    ex_geographic = WMSGeographicBoundingBoxResponse(
        west_bound_longitude=west,
        east_bound_longitude=east,
        south_bound_latitude=south,
        north_bound_latitude=north,
    )

    supported_crs = ["CRS:84", "EPSG:4326", "EPSG:3857"]
    bounding_boxes = [
        _bounding_box_response("CRS:84", CRS84, geo_bounds),
        _bounding_box_response("EPSG:4326", EPSG4326, geo_bounds),
    ]

    mercator_bounds = grid.transform_bbox("EPSG:3857")
    if all(map(math.isfinite, mercator_bounds)):
        bounding_boxes.append(
            _bounding_box_response("EPSG:3857", EPSG3857, mercator_bounds)
        )

    authority = grid.crs.to_authority()
    if authority is not None:
        native_crs = ":".join(authority)
        if native_crs not in supported_crs:
            supported_crs.append(native_crs)
            bounding_boxes.append(
                _bounding_box_response(
                    native_crs,
                    grid.crs,
                    (grid.bbox.west, grid.bbox.south, grid.bbox.east, grid.bbox.north),
                )
            )

    return supported_crs, ex_geographic, bounding_boxes


def extract_layers(dataset: xr.Dataset, base_url: str) -> list[WMSLayerResponse]:
    """Extract layer information from dataset data variables.

    Args:
        dataset: xarray Dataset
        base_url: Base URL for the service

    Returns:
        List of WMSLayerResponse objects for each renderable data variable
    """
    layers = []

    for var_name_, grid in detect_grids(dataset).items():
        var_name = str(var_name_)
        var = dataset[var_name]
        title = str(getattr(var, "long_name", var_name))
        abstract = getattr(var, "description", getattr(var, "comment", None))
        wms_attributes = convert_attributes_to_wms(var.attrs)
        dimensions = extract_dimensions(dataset, set(map(str, var.dims)) - grid.dims)
        supported_crs, ex_geographic, bounding_boxes = _layer_bounds(grid)
        styles = get_available_wms_styles(dataset, base_url=base_url, layer_name=var_name)

        layer = WMSLayerResponse(
            name=var_name,
            title=title,
            abstract=abstract,
            crs=supported_crs,
            ex_geographic_bounding_box=ex_geographic,
            bounding_box=bounding_boxes,
            dimensions=dimensions,
            attributes=wms_attributes,
            styles=styles,
            queryable=False,  # GetFeatureInfo is not implemented yet
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

    # Extract layers from dataset; styles (with legend URLs) live on each layer
    layers = extract_layers(dataset, base_url)

    # Extract dataset attributes for root layer
    dataset_wms_attributes = convert_attributes_to_wms(dataset.attrs)

    root_layer = WMSLayerResponse(
        title="Dataset Layers",
        abstract="All available data layers with raster visualization styles",
        layers=layers,
        attributes=dataset_wms_attributes,
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
