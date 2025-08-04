"""Tile matrix set definitions for OGC Tiles API"""

from typing import Optional, Union

import morecantile
import morecantile.errors
import numpy as np
import pyproj
import pyproj.aoi

from xarray import Dataset
from xpublish_tiles.types import OutputBBox, OutputCRS
from xpublish_tiles.xpublish.tiles.types import (
    BoundingBox,
    Link,
    TileMatrix,
    TileMatrixSet,
    TileMatrixSetLimit,
    TileMatrixSetSummary,
)


def get_tile_matrix_set(tms_id: str) -> TileMatrixSet:
    """Get a complete tile matrix set definition for any morecantile TMS.

    Args:
        tms_id: The tile matrix set identifier (e.g., 'WebMercatorQuad')

    Returns:
        TileMatrixSet object with all tile matrices

    Raises:
        ValueError: If the TMS ID is not found in morecantile
    """
    try:
        tms = morecantile.tms.get(tms_id)
    except morecantile.errors.InvalidIdentifier as e:
        raise ValueError(f"Tile matrix set '{tms_id}' not found") from e

    tile_matrices = [
        TileMatrix(
            id=matrix.id,
            scaleDenominator=matrix.scaleDenominator,
            topLeftCorner=list(matrix.pointOfOrigin),
            tileWidth=matrix.tileWidth,
            tileHeight=matrix.tileHeight,
            matrixWidth=matrix.matrixWidth,
            matrixHeight=matrix.matrixHeight,
        )
        for matrix in tms.tileMatrices
    ]

    return TileMatrixSet(
        id=str(tms.id),
        title=str(tms.title) if tms.title else tms_id,
        uri=str(tms.uri) if tms.uri else None,
        crs=str(tms.crs),
        tileMatrices=tile_matrices,
    )


def get_tile_matrix_set_summary(tms_id: str) -> TileMatrixSetSummary:
    """Get summary information for any morecantile tile matrix set.

    Args:
        tms_id: The tile matrix set identifier (e.g., 'WebMercatorQuad')

    Returns:
        TileMatrixSetSummary object

    Raises:
        ValueError: If the TMS ID is not found in morecantile
    """
    try:
        tms = morecantile.tms.get(tms_id)
    except morecantile.errors.InvalidIdentifier as e:
        raise ValueError(f"Tile matrix set '{tms_id}' not found") from e

    tms_id_str = str(tms.id)
    tms_title = str(tms.title) if tms.title else tms_id
    return TileMatrixSetSummary(
        id=tms_id_str,
        title=tms_title,
        uri=str(tms.uri) if tms.uri else None,
        crs=str(tms.crs),
        links=[
            Link(
                href=f"/tiles/tileMatrixSets/{tms_id_str}",
                rel="self",
                type="application/json",
                title=f"{tms_title} tile matrix set",
            )
        ],
    )


# Legacy functions for backward compatibility
def get_web_mercator_quad() -> TileMatrixSet:
    """Get the complete WebMercatorQuad tile matrix set definition using morecantile"""
    return get_tile_matrix_set("WebMercatorQuad")


def get_web_mercator_quad_summary() -> TileMatrixSetSummary:
    """Get summary information for WebMercatorQuad tile matrix set using morecantile"""
    return get_tile_matrix_set_summary("WebMercatorQuad")


# Generate registry of all available tile matrix sets from morecantile
def _create_tms_registries() -> tuple[dict, dict]:
    """Create registries for all available morecantile TMS."""
    tms_sets = {}
    tms_summaries = {}

    for tms_id in morecantile.tms.list():
        # Create lambda functions that capture the tms_id
        tms_sets[tms_id] = lambda tid=tms_id: get_tile_matrix_set(tid)
        tms_summaries[tms_id] = lambda tid=tms_id: get_tile_matrix_set_summary(tid)

    return tms_sets, tms_summaries


# Registry of available tile matrix sets
TILE_MATRIX_SETS, TILE_MATRIX_SET_SUMMARIES = _create_tms_registries()


def extract_tile_bbox_and_crs(
    tileMatrixSetId: str, tileMatrix: int, tileRow: int, tileCol: int
) -> tuple[OutputBBox, OutputCRS]:
    """Extract bounding box and CRS from tile coordinates using morecantile.

    Args:
        tileMatrixSetId: ID of the tile matrix set
        tileMatrix: Zoom level/tile matrix ID
        tileRow: Row index of the tile
        tileCol: Column index of the tile

    Returns:
        tuple: (bbox as OutputBBox, OutputCRS object)

    Raises:
        ValueError: If tile matrix set not found
    """
    try:
        tms = morecantile.tms.get(tileMatrixSetId)
    except morecantile.errors.InvalidIdentifier as e:
        raise ValueError(f"Tile matrix set '{tileMatrixSetId}' not found") from e
    tile = morecantile.Tile(x=tileCol, y=tileRow, z=tileMatrix)

    # Get the bounding box in the TMS's CRS (projected coordinates)
    bbox = tms.xy_bounds(tile)
    output_bbox = OutputBBox(
        pyproj.aoi.BBox(
            west=bbox.left, south=bbox.bottom, east=bbox.right, north=bbox.top
        )
    )
    crs = pyproj.CRS.from_wkt(tms.crs.to_wkt())
    return output_bbox, OutputCRS(crs)


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
        # Try to extract from lat/lon coordinates with various naming conventions
        else:
            lat_coord = None
            lon_coord = None

            # Look for latitude coordinate with common names
            for lat_name in ["lat", "latitude", "y", "Y"]:
                if lat_name in dataset.coords:
                    lat_coord = dataset.coords[lat_name]
                    break

            # Look for longitude coordinate with common names
            for lon_name in ["lon", "longitude", "x", "X"]:
                if lon_name in dataset.coords:
                    lon_coord = dataset.coords[lon_name]
                    break

            # Also check using CF conventions
            if lat_coord is None or lon_coord is None:
                try:
                    import cf_xarray as cfxr  # noqa: F401

                    # Try to find coordinates using CF standard names
                    for coord_name, coord in dataset.coords.items():
                        standard_name = coord.attrs.get("standard_name", "")
                        axis = coord.attrs.get("axis", "")

                        if standard_name == "latitude" or axis == "Y":
                            lat_coord = coord
                        elif standard_name == "longitude" or axis == "X":
                            lon_coord = coord
                except ImportError:
                    pass

            if lat_coord is not None and lon_coord is not None:
                lat_min, lat_max = float(lat_coord.min()), float(lat_coord.max())
                lon_min, lon_max = float(lon_coord.min()), float(lon_coord.max())

                # Handle 0-360 longitude range by converting to -180-180
                if lon_min >= 0 and lon_max > 180:
                    # Convert 0-360 to -180-180
                    lon_values = lon_coord.values
                    lon_values = np.where(lon_values > 180, lon_values - 360, lon_values)
                    lon_min, lon_max = float(lon_values.min()), float(lon_values.max())

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

    from xpublish_tiles.xpublish.tiles.types import DimensionExtent, DimensionType

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
