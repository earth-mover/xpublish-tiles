"""Tile matrix set definitions for OGC Tiles API"""

from typing import Optional, Union

import pyproj

from xarray import Dataset
from xpublish_tiles.xpublish.tiles.models import (
    BoundingBox,
    CRSType,
    Link,
    TileMatrix,
    TileMatrixSet,
    TileMatrixSetLimit,
    TileMatrixSetSummary,
)


def get_web_mercator_quad() -> TileMatrixSet:
    """Get the complete WebMercatorQuad tile matrix set definition with all zoom levels 0-21"""

    # Base values for WebMercator
    origin_x = -20037508.3428
    origin_y = 20037508.3428
    tile_size = 256
    base_scale_denominator = 559082264.029

    tile_matrices = []

    # Generate all zoom levels from 0 to 21
    for zoom in range(22):  # 0 to 21 inclusive
        scale_denominator = base_scale_denominator / (2**zoom)
        matrix_size = 2**zoom

        tile_matrices.append(
            TileMatrix(
                id=str(zoom),
                scaleDenominator=scale_denominator,
                topLeftCorner=[origin_x, origin_y],
                tileWidth=tile_size,
                tileHeight=tile_size,
                matrixWidth=matrix_size,
                matrixHeight=matrix_size,
            )
        )

    return TileMatrixSet(
        id="WebMercatorQuad",
        title="Web Mercator Quad",
        uri="http://www.opengis.net/def/tilematrixset/OGC/1.0/WebMercatorQuad",
        crs="http://www.opengis.net/def/crs/EPSG/0/3857",
        tileMatrices=tile_matrices,
    )


def get_web_mercator_quad_summary() -> TileMatrixSetSummary:
    """Get summary information for WebMercatorQuad tile matrix set"""
    return TileMatrixSetSummary(
        id="WebMercatorQuad",
        title="Web Mercator Quad",
        uri="http://www.opengis.net/def/tilematrixset/OGC/1.0/WebMercatorQuad",
        crs="http://www.opengis.net/def/crs/EPSG/0/3857",
        links=[
            Link(
                href="/tiles/tileMatrixSets/WebMercatorQuad",
                rel="self",
                type="application/json",
                title="Web Mercator Quad tile matrix set",
            )
        ],
    )


# Registry of available tile matrix sets
TILE_MATRIX_SETS = {
    "WebMercatorQuad": get_web_mercator_quad,
}

TILE_MATRIX_SET_SUMMARIES = {
    "WebMercatorQuad": get_web_mercator_quad_summary,
}


def extract_tile_bbox_and_crs(
    tileMatrixSetId: str, tileMatrix: int, tileRow: int, tileCol: int
) -> tuple[list[float], pyproj.CRS]:
    """Extract bounding box and CRS from tile coordinates.

    Args:
        tileMatrixSetId: ID of the tile matrix set
        tileMatrix: Zoom level/tile matrix ID
        tileRow: Row index of the tile
        tileCol: Column index of the tile

    Returns:
        tuple: (bbox as [minX, minY, maxX, maxY], pyproj.CRS object)

    Raises:
        ValueError: If tile matrix set or tile matrix not found, or CRS conversion fails
    """
    if tileMatrixSetId not in TILE_MATRIX_SETS:
        raise ValueError(f"Tile matrix set '{tileMatrixSetId}' not found")

    tile_matrix_set = TILE_MATRIX_SETS[tileMatrixSetId]()

    tile_matrix_def = None
    for tm in tile_matrix_set.tileMatrices:
        if tm.id == str(tileMatrix):
            tile_matrix_def = tm
            break

    if not tile_matrix_def:
        raise ValueError(f"Tile matrix '{tileMatrix}' not found")

    origin_x, origin_y = tile_matrix_def.topLeftCorner
    tile_width = tile_matrix_def.tileWidth
    tile_height = tile_matrix_def.tileHeight

    pixel_size = tile_matrix_def.scaleDenominator * 0.00028

    min_x = origin_x + (tileCol * tile_width * pixel_size)
    max_x = origin_x + ((tileCol + 1) * tile_width * pixel_size)
    max_y = origin_y - (tileRow * tile_height * pixel_size)
    min_y = origin_y - ((tileRow + 1) * tile_height * pixel_size)

    bbox = [min_x, min_y, max_x, max_y]

    # Convert CRS to pyproj.CRS object
    if isinstance(tile_matrix_set.crs, str):
        # Handle string CRS (URI format)
        crs_type = CRSType(uri=tile_matrix_set.crs)
        pyproj_crs = crs_type.to_pyproj_crs()
    else:
        # Handle CRSType object
        pyproj_crs = tile_matrix_set.crs.to_pyproj_crs()

    if pyproj_crs is None:
        raise ValueError(f"Could not convert CRS '{tile_matrix_set.crs}' to pyproj.CRS")

    return bbox, pyproj_crs


def extract_dataset_bounds(dataset: Dataset) -> Optional[BoundingBox]:
    """Extract geographic bounds from a dataset.

    TODO: This functionality may be handled by the tile rendering pipeline in the future.

    Args:
        dataset: xarray Dataset to extract bounds from

    Returns:
        BoundingBox object if bounds can be extracted, None otherwise
    """
    try:
        # Try to get bounds from dataset bounds attribute
        if hasattr(dataset, "bounds"):
            bounds = dataset.bounds
            return BoundingBox(
                lowerLeft=[float(bounds[0]), float(bounds[1])],
                upperRight=[float(bounds[2]), float(bounds[3])],
                crs="http://www.opengis.net/def/crs/EPSG/0/4326",
            )
        # Try to extract from lat/lon coordinates
        elif "lat" in dataset.coords and "lon" in dataset.coords:
            lat_min, lat_max = float(dataset.lat.min()), float(dataset.lat.max())
            lon_min, lon_max = float(dataset.lon.min()), float(dataset.lon.max())
            return BoundingBox(
                lowerLeft=[lon_min, lat_min],
                upperRight=[lon_max, lat_max],
                crs="http://www.opengis.net/def/crs/EPSG/0/4326",
                orderedAxes=["X", "Y"],
            )
    except Exception:
        # If we can't extract bounds, return None
        pass
    return None


def get_tile_matrix_limits(
    tms_id: str, zoom_levels: Optional[range] = None
) -> list[TileMatrixSetLimit]:
    """Generate tile matrix limits for the specified zoom levels.

    TODO: Calculate actual limits based on dataset bounds instead of full world coverage.

    Args:
        tms_id: Tile matrix set identifier
        zoom_levels: Range of zoom levels to generate limits for (default: 0-18)

    Returns:
        List of TileMatrixSetLimit objects
    """
    if zoom_levels is None:
        zoom_levels = range(19)  # 0-18

    limits = []
    for z in zoom_levels:
        max_tiles = 2**z - 1
        limits.append(
            TileMatrixSetLimit(
                tileMatrix=str(z),
                minTileRow=0,
                maxTileRow=max_tiles,
                minTileCol=0,
                maxTileCol=max_tiles,
            )
        )
    return limits


def get_all_tile_matrix_set_ids() -> list[str]:
    """Get list of all available tile matrix set IDs."""
    return list(TILE_MATRIX_SETS.keys())


def extract_dimension_extents(data_array) -> list:
    """Extract dimension extent information from an xarray DataArray.

    Uses cf_xarray to detect CF-compliant axes for robust dimension classification.

    Args:
        data_array: xarray DataArray to extract dimensions from

    Returns:
        List of DimensionExtent objects for non-spatial dimensions
    """
    import cf_xarray as cfxr  # noqa: F401 - needed to enable .cf accessor
    import numpy as np
    import pandas as pd

    from xpublish_tiles.xpublish.tiles.models import DimensionExtent, DimensionType

    dimensions = []

    # Get CF axes information
    try:
        cf_axes = data_array.cf.axes
    except Exception:
        # Fallback if cf_xarray fails
        cf_axes = {}

    # Identify spatial and temporal dimensions using CF conventions
    spatial_dims = set()
    temporal_dims = set()
    vertical_dims = set()

    # Add CF-detected spatial dimensions (X, Y axes)
    spatial_dims.update(cf_axes.get("X", []))
    spatial_dims.update(cf_axes.get("Y", []))

    # Add CF-detected temporal dimensions (T axis)
    temporal_dims.update(cf_axes.get("T", []))

    # Add CF-detected vertical dimensions (Z axis)
    vertical_dims.update(cf_axes.get("Z", []))

    for dim_name in data_array.dims:
        # Skip spatial dimensions (X, Y axes)
        if dim_name in spatial_dims:
            continue

        coord = data_array.coords.get(dim_name)
        if coord is None:
            continue

        # Determine dimension type using CF axes
        dim_type = DimensionType.CUSTOM
        if dim_name in temporal_dims:
            dim_type = DimensionType.TEMPORAL
        elif dim_name in vertical_dims:
            dim_type = DimensionType.VERTICAL

        # Extract coordinate values
        values = coord.values

        # Handle different coordinate types
        values_list: list[Union[str, float, int]]
        extent: list[Union[str, float, int]]

        if np.issubdtype(values.dtype, np.datetime64):
            # Convert datetime to ISO strings
            if hasattr(values, "astype"):
                datetime_series = pd.to_datetime(values)
                formatted_series = datetime_series.strftime("%Y-%m-%dT%H:%M:%SZ")
                str_values = list(formatted_series)
            else:
                str_values = [
                    pd.to_datetime(val).strftime("%Y-%m-%dT%H:%M:%SZ") for val in values
                ]
            extent = [str_values[0], str_values[-1]]
            values_list = list(str_values)
        elif np.issubdtype(values.dtype, np.number):
            # Numeric coordinates
            extent = [float(values.min()), float(values.max())]
            values_list = [float(val) for val in values]
        else:
            # String/categorical coordinates
            values_list = [str(val) for val in values]
            extent = values_list  # For categorical, extent is all values

        # Get units and description from attributes
        units = coord.attrs.get("units")
        description = coord.attrs.get("long_name") or coord.attrs.get("description")

        # Determine default value (first value)
        default = values_list[0] if values_list else None

        # Limit values list size for performance
        limited_values = values_list if len(values_list) <= 100 else None

        dimension = DimensionExtent(
            name=dim_name,
            type=dim_type,
            extent=extent,
            values=limited_values,
            units=units,
            description=description,
            default=default,
        )
        dimensions.append(dimension)

    return dimensions
