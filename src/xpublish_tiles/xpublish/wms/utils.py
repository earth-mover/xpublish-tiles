"""Utilities for WMS dataset introspection and metadata extraction"""

import numpy as np
from pyproj import CRS

import xarray as xr
from xpublish_tiles.grids import (
    bbox_in_crs,
    canonical_geographic_bbox,
    compute_bbox_union_native,
)
from xpublish_tiles.xpublish.tiles.metadata import extract_dataset_extents
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
    """Extract a canonical geographic bbox using pipeline grid inference.

    Returns a tuple (west, east, south, north) in EPSG:4326 with longitudes
    normalized to [-180, 180] and latitudes clamped to [-90, 90].
    """
    native_crs, native_bbox = compute_bbox_union_native(dataset)
    geo_bbox = canonical_geographic_bbox(native_crs, native_bbox)
    return (
        float(geo_bbox.west),
        float(geo_bbox.east),
        float(geo_bbox.south),
        float(geo_bbox.north),
    )


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

    # Derive accurate bounds using grid inference
    native_crs, native_bbox = compute_bbox_union_native(dataset)

    # Geographic (EPSG:4326)
    west, east, south, north = extract_geographic_bounds(dataset)
    geo_bbox = WMSGeographicBoundingBoxResponse(
        west_bound_longitude=west,
        east_bound_longitude=east,
        south_bound_latitude=south,
        north_bound_latitude=north,
    )

    # Accurate projected bbox in Web Mercator
    bbox_3857 = bbox_in_crs(native_crs, native_bbox, CRS.from_epsg(3857))

    # Create bounding boxes for different CRS
    bounding_boxes = [
        WMSBoundingBoxResponse(
            crs="EPSG:4326", minx=west, miny=south, maxx=east, maxy=north
        ),
        WMSBoundingBoxResponse(
            crs="EPSG:3857",
            minx=float(bbox_3857.west),
            miny=float(bbox_3857.south),
            maxx=float(bbox_3857.east),
            maxy=float(bbox_3857.north),
        ),
    ]

    # Extract dimensions using unified extents calculator
    extents = extract_dataset_extents(dataset, None)

    def _to_wms_dimensions(
        ext: dict[str, dict[str, object]],
    ) -> list[WMSDimensionResponse]:
        dims: list[WMSDimensionResponse] = []
        for name, meta in ext.items():
            units = str(meta.get("units")) if meta.get("units") is not None else ""
            default = meta.get("default")
            default_str = str(default) if default is not None else None
            values_list = meta.get("values")
            interval = meta.get("interval")
            if isinstance(values_list, list) and values_list:
                values = ",".join(str(v) for v in values_list)
            elif isinstance(interval, list) and interval:
                values = ",".join(str(v) for v in interval)
            else:
                values = ""
            if name.lower() in ("time", "t"):
                dim_name = "time"
                units = units or "ISO8601"
            else:
                dim_name = name
            dims.append(
                WMSDimensionResponse(
                    name=dim_name,
                    units=units,
                    default=default_str,
                    values=values,
                    multiple_values=True,
                    nearest_value=True,
                )
            )
        return dims

    dimensions = _to_wms_dimensions(extents)

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
