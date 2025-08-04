from typing import Any, Union
from urllib.parse import urlencode

import pyproj
from fastapi import Request

from xarray import Dataset
from xpublish_tiles.utils import get_available_raster_styles
from xpublish_tiles.xpublish.tiles.tile_matrix import (
    TILE_MATRIX_SET_SUMMARIES,
    extract_dataset_bounds,
    extract_dimension_extents,
)
from xpublish_tiles.xpublish.tiles.types import (
    DataType,
    DimensionType,
    Link,
    Style,
    TileJSON,
    TileSetMetadata,
)


def create_tileset_metadata(dataset: Dataset, tile_matrix_set_id: str) -> TileSetMetadata:
    """Create tileset metadata for a dataset and tile matrix set"""
    # Get tile matrix set summary
    if tile_matrix_set_id not in TILE_MATRIX_SET_SUMMARIES:
        raise ValueError(f"Tile matrix set '{tile_matrix_set_id}' not found")

    tms_summary = TILE_MATRIX_SET_SUMMARIES[tile_matrix_set_id]()

    # Extract dataset metadata
    dataset_attrs = dataset.attrs
    title = dataset_attrs.get("title", "Dataset")

    # Extract dataset bounds
    dataset_bounds = extract_dataset_bounds(dataset)

    # Get available styles
    style_dicts = get_available_raster_styles()
    styles = [
        Style(
            id=style["id"],
            title=style["title"],
            description=style["description"],
        )
        for style in style_dicts
    ]

    # Create main tileset metadata
    return TileSetMetadata(
        title=f"{title} - {tile_matrix_set_id}",
        tileMatrixSetURI=tms_summary.uri,
        crs=tms_summary.crs,
        dataType=DataType.MAP,
        links=[
            Link(
                href=f"./{tile_matrix_set_id}/{{tileMatrix}}/{{tileRow}}/{{tileCol}}",
                rel="item",
                type="image/png",
                title="Tile",
                templated=True,
            ),
            Link(
                href=f"/tileMatrixSets/{tile_matrix_set_id}",
                rel="http://www.opengis.net/def/rel/ogc/1.0/tiling-scheme",
                type="application/json",
                title=f"Definition of {tile_matrix_set_id}",
            ),
        ],
        boundingBox=dataset_bounds,
        styles=styles,
    )


def extract_dataset_extents(
    dataset: Dataset, variable_name: str | None
) -> dict[str, dict[str, Any]]:
    """Extract dimension extents from dataset and convert to OGC format"""
    extents = {}

    # Collect all dimensions from all data variables
    all_dimensions = {}

    # When a variable name is provided, extract dimensions from that variable only
    if variable_name:
        ds = dataset[[variable_name]]
    else:
        ds = dataset

    for var_data in ds.data_vars.values():
        dimensions = extract_dimension_extents(var_data)
        for dim in dimensions:
            # Use the first occurrence of each dimension name
            if dim.name not in all_dimensions:
                all_dimensions[dim.name] = dim

    # Convert DimensionExtent objects to OGC extents format
    for dim_name, dim_extent in all_dimensions.items():
        extent_dict = {"interval": dim_extent.extent}

        # Calculate resolution if possible
        if dim_extent.values and len(dim_extent.values) > 1:
            values = dim_extent.values
            if dim_extent.type == DimensionType.TEMPORAL:
                # For temporal dimensions, try to calculate time resolution
                extent_dict["resolution"] = _calculate_temporal_resolution(values)
            elif isinstance(values[0], int | float):
                # For numeric dimensions, calculate step size
                diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
                if diffs:
                    extent_dict["resolution"] = min(diffs)

        # Add units if available
        if dim_extent.units:
            extent_dict["units"] = dim_extent.units

        # Add description if available
        if dim_extent.description:
            extent_dict["description"] = dim_extent.description

        # Add default value if available
        if dim_extent.default is not None:
            extent_dict["default"] = dim_extent.default

        extents[dim_name] = extent_dict

    return extents


def _calculate_temporal_resolution(values: list[Union[str, float, int]]) -> str:
    """Calculate temporal resolution from datetime values"""
    if len(values) < 2:
        return "PT1H"  # Default to hourly

    try:
        import pandas as pd

        # Convert to datetime if they're strings
        if isinstance(values[0], str):
            dt_values = [pd.to_datetime(v) for v in values[:10]]  # Sample first 10
        else:
            return "PT1H"  # Default for non-string values

        # Calculate differences
        diffs = [
            (dt_values[i + 1] - dt_values[i]).total_seconds()
            for i in range(len(dt_values) - 1)
        ]

        if not diffs:
            return "PT1H"

        # Get the most common difference
        avg_diff = sum(diffs) / len(diffs)

        # Convert to ISO 8601 duration format
        if avg_diff >= 86400:  # >= 1 day
            days = int(avg_diff / 86400)
            return f"P{days}D"
        elif avg_diff >= 3600:  # >= 1 hour
            hours = int(avg_diff / 3600)
            return f"PT{hours}H"
        elif avg_diff >= 60:  # >= 1 minute
            minutes = int(avg_diff / 60)
            return f"PT{minutes}M"
        else:
            seconds = int(avg_diff)
            return f"PT{seconds}S"

    except Exception:
        return "PT1H"  # Default fallback


def _transform_bounds_to_crs(bounds: list[float], target_crs: str) -> list[float]:
    """Transform bounds from WGS84 to target CRS"""
    try:
        # Source bounds are in WGS84 [west, south, east, north]
        source_crs = pyproj.CRS.from_epsg(4326)
        target_crs_obj = pyproj.CRS.from_user_input(target_crs)

        transformer = pyproj.Transformer.from_crs(
            source_crs, target_crs_obj, always_xy=True
        )

        # Transform corner points
        west, south, east, north = bounds
        x1, y1 = transformer.transform(west, south)
        x2, y2 = transformer.transform(east, north)

        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    except Exception:
        # If transformation fails, return original bounds
        return bounds


def _calculate_appropriate_zoom_levels(dataset: Dataset, tms_id: str) -> tuple[int, int]:
    """Calculate appropriate min/max zoom levels based on dataset resolution"""
    try:
        # Default zoom levels
        min_zoom = 0
        max_zoom = 18

        # Try to estimate based on spatial resolution if available
        if "lat" in dataset.coords and "lon" in dataset.coords:
            lat_res = (
                abs(float(dataset.lat[1] - dataset.lat[0]))
                if len(dataset.lat) > 1
                else 1.0
            )
            lon_res = (
                abs(float(dataset.lon[1] - dataset.lon[0]))
                if len(dataset.lon) > 1
                else 1.0
            )

            # Estimate zoom level based on resolution
            # This is a rough approximation
            avg_res = (lat_res + lon_res) / 2

            if avg_res >= 1.0:  # Very coarse data
                max_zoom = 6
            elif avg_res >= 0.1:  # Coarse data
                max_zoom = 10
            elif avg_res >= 0.01:  # Medium resolution
                max_zoom = 14
            else:  # High resolution
                max_zoom = 18

        return min_zoom, max_zoom
    except Exception:
        return 0, 18


def _calculate_center_point(bounds: list[float], target_crs: str) -> list[float]:
    """Calculate center point from bounds"""
    try:
        if len(bounds) != 4:
            return [0, 0, 2]

        west, south, east, north = bounds
        center_x = (west + east) / 2
        center_y = (south + north) / 2

        # If target CRS is geographic (4326), return as lon/lat
        if "4326" in target_crs or "WGS84" in target_crs:
            return [center_x, center_y, 2]

        # For projected coordinates, transform back to WGS84 for center
        try:
            source_crs = pyproj.CRS.from_user_input(target_crs)
            target_crs_obj = pyproj.CRS.from_epsg(4326)
            transformer = pyproj.Transformer.from_crs(
                source_crs, target_crs_obj, always_xy=True
            )
            lon, lat = transformer.transform(center_x, center_y)
            return [lon, lat, 2]
        except Exception:
            return [0, 0, 2]

    except Exception:
        return [0, 0, 2]


def create_tilejson(
    dataset: Dataset, tile_matrix_set_id: str, request: Request
) -> TileJSON:
    """Create TileJSON specification for a dataset and tile matrix set

    Args:
        dataset: xarray Dataset
        tile_matrix_set_id: Tile matrix set identifier
        request: FastAPI request object to extract query parameters and build URLs

    Returns:
        TileJSON object

    Raises:
        ValueError: If tile matrix set not found
    """
    # Validate tile matrix set
    if tile_matrix_set_id not in TILE_MATRIX_SET_SUMMARIES:
        raise ValueError(f"Tile matrix set '{tile_matrix_set_id}' not found")

    tms_summary = TILE_MATRIX_SET_SUMMARIES[tile_matrix_set_id]()

    # Extract dataset metadata
    dataset_attrs = dataset.attrs
    title = dataset_attrs.get("title", "Dataset")
    description = dataset_attrs.get("description", "")
    attribution = dataset_attrs.get("attribution")
    version = dataset_attrs.get("version")

    # Get dataset bounds
    dataset_bounds = extract_dataset_bounds(dataset)
    bounds = None
    center = None

    if dataset_bounds:
        # Convert bounds to list format [west, south, east, north]
        bounds = [
            dataset_bounds.lowerLeft[0],
            dataset_bounds.lowerLeft[1],
            dataset_bounds.upperRight[0],
            dataset_bounds.upperRight[1],
        ]

        # Transform bounds to target CRS if needed
        target_crs = str(tms_summary.crs)
        if not ("4326" in target_crs or "WGS84" in target_crs):
            bounds = _transform_bounds_to_crs(bounds, target_crs)

        # Calculate center point
        center = _calculate_center_point(bounds, target_crs)

    # Calculate appropriate zoom levels
    min_zoom, max_zoom = _calculate_appropriate_zoom_levels(dataset, tile_matrix_set_id)

    # Build tile URL template with current query parameters
    base_url = str(
        request.url_for(
            "get_dataset_tile",
            dataset_id=request.path_params.get("dataset_id", ""),
            tileMatrixSetId=tile_matrix_set_id,
            tileMatrix="{z}",
            tileRow="{y}",
            tileCol="{x}",
        )
    )

    # Extract query parameters, excluding TileJSON-specific ones
    query_params = dict(request.query_params)
    tilejson_params = {"tilejson", "format"}  # Parameters specific to TileJSON endpoint
    filtered_params = {k: v for k, v in query_params.items() if k not in tilejson_params}

    # Add query string if there are parameters
    if filtered_params:
        query_string = urlencode(filtered_params)
        tile_url = f"{base_url}?{query_string}"
    else:
        tile_url = base_url

    # Create TileJSON object
    tilejson = TileJSON(
        tilejson="3.0.0",
        name=f"{title} - {tile_matrix_set_id}",
        description=description
        or f"Tiles for {title} in {tile_matrix_set_id} projection",
        tiles=[tile_url],
        bounds=bounds,
        center=center,
        minzoom=min_zoom,
        maxzoom=max_zoom,
        attribution=attribution,
        version=version,
        scheme="xyz",  # Standard XYZ tile scheme
    )

    return tilejson
