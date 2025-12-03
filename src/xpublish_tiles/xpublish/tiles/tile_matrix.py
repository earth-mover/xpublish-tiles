"""Tile matrix set definitions for OGC Tiles API"""

from collections.abc import Hashable
from typing import Union, cast

import cf_xarray as cfxr  # noqa: F401 - needed to enable .cf accessor
import morecantile
import morecantile.errors
import numpy as np
import pandas as pd
import pyproj
import pyproj.aoi

import xarray as xr
from xpublish_tiles.grids import guess_grid_system
from xpublish_tiles.lib import async_run
from xpublish_tiles.tiles_lib import get_min_zoom
from xpublish_tiles.types import OutputBBox, OutputCRS
from xpublish_tiles.xpublish.tiles.types import (
    DimensionExtent,
    DimensionType,
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
        crs=tms.crs,
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
        crs=tms.crs,
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


async def get_tile_matrix_limits(
    tms_id: str, dataset: xr.Dataset, zoom_levels: range | None = None
) -> list[TileMatrixSetLimit]:
    """Generate tile matrix limits for the specified zoom levels based on dataset bounds.

    Args:
        tms_id: Tile matrix set identifier
        dataset: xarray Dataset to extract bounds from
        zoom_levels: Range of zoom levels to generate limits for. If None, will be calculated
                    from the Grid's min/max zoom levels.

    Returns:
        List of TileMatrixSetLimit objects
    """
    for name, var in dataset.data_vars.items():
        if var.ndim >= 1:
            first_data_var = name
            break
    else:
        raise ValueError("Could not find a DataArray with at least one dimension.")

    grid = await async_run(guess_grid_system, dataset, first_data_var)
    tms = morecantile.tms.get(tms_id)

    if zoom_levels is None:
        min_zoom = await async_run(get_min_zoom, grid, tms, dataset[first_data_var])
        zoom_levels = range(min_zoom, tms.maxzoom)

    limits = []
    for z in zoom_levels:
        minmax = tms.minmax(z)
        limits.append(
            TileMatrixSetLimit(
                tileMatrix=str(z),
                minTileRow=minmax["x"]["min"],
                maxTileRow=minmax["x"]["max"],
                minTileCol=minmax["y"]["min"],
                maxTileCol=minmax["y"]["max"],
            )
        )

    return limits


def get_all_tile_matrix_set_ids() -> list[str]:
    """Get list of all available tile matrix set IDs."""
    return list(TILE_MATRIX_SETS.keys())


async def extract_dimension_extents(
    ds: xr.Dataset, name: Hashable, max_actual_values: int = 50
) -> list[DimensionExtent]:
    """Extract dimension extent information from an xarray DataArray.

    Uses cf_xarray to detect CF-compliant axes for robust dimension classification.

    Args:
        data_array: xarray DataArray to extract dimensions from
        name: Name of the data array
        max_actual_values: Maximum number of actual values to extract,
            otherwise only extents and resolution will be extracted.

    Returns:
        List of DimensionExtent objects for non-spatial dimensions
    """
    dimensions = []

    grid = await async_run(guess_grid_system, ds, name)
    data_array = ds[name]

    # Get CF axes information
    try:
        cf_axes = data_array.cf.axes
    except Exception:
        # Fallback if cf_xarray fails
        cf_axes = {}

    # Identify spatial and temporal dimensions using CF conventions
    spatial_dims = {grid.Xdim, grid.Ydim}
    temporal_dims = set()
    vertical_dims = {grid.Z} if grid.Z else set()

    # Add CF-detected temporal dimensions (T axis)
    temporal_dims.update(cf_axes.get("T", []))

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
        extent: list[Union[str, float, int]]
        actual_values: list[str | float | int] | None = None

        if len(values) == 0:
            extent = []
        elif np.issubdtype(values.dtype, np.timedelta64):
            extent = [str(values[0]), str(values[-1])]
            if len(values) <= max_actual_values:
                actual_values = [str(value) for value in values]
        elif np.issubdtype(values.dtype, np.datetime64):
            dt_values = cast(list[pd.Timestamp], pd.to_datetime(values))
            # Convert datetime to ISO strings - only get first and last values for extent
            first_datetime = dt_values[0].isoformat()
            last_datetime = dt_values[-1].isoformat()
            extent = [first_datetime, last_datetime]
            if len(values) <= max_actual_values:
                actual_values = [value.isoformat() for value in dt_values]
        elif np.issubdtype(values.dtype, np.number):
            # Numeric coordinates
            extent = [values.min(), values.max()]
            if len(values) <= max_actual_values:
                actual_values = values
        else:
            extent = [str(values[0]), str(values[-1])]
            if len(values) <= max_actual_values:
                actual_values = [str(value) for value in values]

        # Get units and description from attributes
        units = coord.attrs.get("units")
        description = coord.attrs.get("long_name") or coord.attrs.get("description")

        # Determine default value (first value)
        default = None
        if extent:
            if dim_type == DimensionType.VERTICAL:
                default = extent[0]
            else:
                default = extent[-1]

        dimension = DimensionExtent(
            name=str(dim_name),
            type=dim_type,
            extent=extent,
            units=units,
            description=description,
            default=default,
            values=actual_values,
        )
        dimensions.append(dimension)

    return dimensions
