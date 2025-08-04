from typing import Any, Union

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
