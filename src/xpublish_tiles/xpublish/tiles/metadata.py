import functools
from typing import Any, Optional, cast

import morecantile.models

import xarray as xr
from xarray import Dataset
from xpublish_tiles.grids import Curvilinear, RasterAffine, Rectilinear, guess_grid_system
from xpublish_tiles.logger import logger
from xpublish_tiles.pipeline import transformer_from_crs
from xpublish_tiles.render import RenderRegistry
from xpublish_tiles.xpublish.tiles.tile_matrix import (
    TILE_MATRIX_SET_SUMMARIES,
    extract_dimension_extents,
)
from xpublish_tiles.xpublish.tiles.types import (
    AttributesMetadata,
    BoundingBox,
    DataType,
    DimensionType,
    Link,
    Style,
    TileSetMetadata,
)


@functools.cache
def get_styles():
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
            Style(
                id=default_style_info["id"],
                title=default_style_info["title"],
                description=default_style_info["description"],
            )
        )

        # Add all actual variants
        for variant in renderer_cls.supported_variants():
            style_info = renderer_cls.describe_style(variant)
            styles.append(
                Style(
                    id=style_info["id"],
                    title=style_info["title"],
                    description=style_info["description"],
                )
            )
    return styles


def extract_attributes_metadata(
    dataset: Dataset, variable_name: str | None = None
) -> AttributesMetadata:
    """Extract and filter attributes from dataset and variables

    Args:
        dataset: xarray Dataset
        variable_name: Optional variable name to extract attributes for specific variable only

    Returns:
        AttributesMetadata object with filtered dataset and variable attributes
    """
    # Extract variable attributes
    variable_attrs = {}
    if variable_name:
        # Extract attributes for specific variable only
        if variable_name in dataset.data_vars:
            variable_attrs[variable_name] = dataset[variable_name].attrs
    else:
        # Extract attributes for all data variables
        for var_name, var_data in dataset.data_vars.items():
            variable_attrs[var_name] = var_data.attrs

    return AttributesMetadata(dataset_attrs=dataset.attrs, variable_attrs=variable_attrs)


def create_tileset_metadata(dataset: Dataset, tile_matrix_set_id: str) -> TileSetMetadata:
    """Create tileset metadata for a dataset and tile matrix set"""
    # Get tile matrix set summary
    if tile_matrix_set_id not in TILE_MATRIX_SET_SUMMARIES:
        raise ValueError(f"Tile matrix set '{tile_matrix_set_id}' not found")

    tms_summary = TILE_MATRIX_SET_SUMMARIES[tile_matrix_set_id]()

    # Extract dataset metadata
    dataset_attrs = dataset.attrs
    title = dataset_attrs.get("title", "Dataset")

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
        styles=get_styles(),
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
        ds = cast(xr.Dataset, dataset[[variable_name]])
    else:
        ds = dataset

    for var in ds.data_vars:
        dimensions = extract_dimension_extents(ds, var)
        for dim in dimensions:
            # Use the first occurrence of each dimension name
            if dim.name not in all_dimensions:
                all_dimensions[dim.name] = dim

    # Convert DimensionExtent objects to OGC extents format
    for dim_name, dim_extent in all_dimensions.items():
        extent_dict = {"interval": dim_extent.extent}
        values = dataset[dim_name]

        # Calculate resolution if possible
        if len(values) > 1:
            if dim_extent.type == DimensionType.TEMPORAL:
                # For temporal dimensions, try to calculate time resolution
                extent_dict["resolution"] = _calculate_temporal_resolution(values)
            elif isinstance(values.data[0].item(), int | float):
                # For numeric dimensions, calculate step size
                data = values.data
                diffs = [abs(data[i + 1] - data[i]).item() for i in range(len(data) - 1)]
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


def _calculate_temporal_resolution(values: xr.DataArray) -> str:
    """Calculate temporal resolution from datetime values"""
    if hasattr(values, "size"):
        if values.size < 2:
            return "PT1H"  # Default to hourly
    elif not bool(values):
        return "PT1H"  # Default to hourly

    try:
        # Calculate differences
        diffs = values[:10].diff(values.name).dt.total_seconds().data

        # Get the most common difference
        avg_diff = diffs.mean()

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


def extract_variable_bounding_box(
    dataset: Dataset, variable_name: str, target_crs: str | morecantile.models.CRS
) -> Optional[BoundingBox]:
    """Extract variable-specific bounding box and transform to target CRS

    Args:
        dataset: xarray Dataset
        variable_name: Name of the variable to extract bounds for
        target_crs: Target coordinate reference system

    Returns:
        BoundingBox object if bounds can be extracted, None otherwise
    """
    try:
        # Get the grid system for this variable
        guessed_grid = guess_grid_system(dataset, variable_name)

        if not isinstance(guessed_grid, RasterAffine | Rectilinear | Curvilinear):
            logger.error(f"Failed to guess grid system for {variable_name}")
            return None

        grid = cast(RasterAffine | Rectilinear | Curvilinear, guessed_grid)

        # After isinstance check, grid is guaranteed to have crs and bbox attributes

        # Convert target CRS to string format for transformer
        if isinstance(target_crs, morecantile.models.CRS):
            target_crs_str = target_crs.to_epsg() or target_crs.to_wkt() or ""
        else:
            target_crs_str = target_crs

        # Transform bounds to target CRS
        transformer = transformer_from_crs(crs_from=grid.crs, crs_to=target_crs_str)
        transformed_bounds = transformer.transform_bounds(
            grid.bbox.west,
            grid.bbox.south,
            grid.bbox.east,
            grid.bbox.north,
        )

        return BoundingBox(
            lowerLeft=[transformed_bounds[0], transformed_bounds[1]],
            upperRight=[transformed_bounds[2], transformed_bounds[3]],
            crs=target_crs,
        )

    except Exception as e:
        logger.error(f"Failed to transform bounds: {e}")
        return None
