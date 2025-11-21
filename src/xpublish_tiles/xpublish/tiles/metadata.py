import functools
from typing import Any, cast

import morecantile.models
import numpy as np

import xarray as xr
from xarray import Dataset
from xpublish_tiles.grids import guess_grid_system
from xpublish_tiles.lib import VariableNotFoundError, async_run
from xpublish_tiles.logger import logger
from xpublish_tiles.pipeline import transformer_from_crs
from xpublish_tiles.render import RenderRegistry
from xpublish_tiles.xpublish.tiles.tile_matrix import (
    TILE_MATRIX_SET_SUMMARIES,
    TILE_MATRIX_SETS,
    extract_dimension_extents,
    get_tile_matrix_limits,
)
from xpublish_tiles.xpublish.tiles.types import (
    AttributesMetadata,
    BoundingBox,
    DataType,
    DimensionType,
    Layer,
    Link,
    Style,
    TileSetMetadata,
    TilesetSummary,
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


async def extract_dataset_extents(
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

    for var, array in ds.data_vars.items():
        if array.ndim == 0:
            continue
        dimensions = await extract_dimension_extents(ds, var)
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
            elif np.issubdtype(values.data.dtype, np.integer) or np.issubdtype(
                values.data.dtype, np.floating
            ):
                # If the type is an unsigned integer, we need to cast to an int to avoid overflow
                if np.issubdtype(values.data.dtype, np.unsignedinteger):
                    values = values.astype(np.int64)

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


async def extract_variable_bounding_box(
    dataset: Dataset, variable_name: str, target_crs: str | morecantile.models.CRS
) -> BoundingBox | None:
    """Extract variable-specific bounding box and transform to target CRS

    Args:
        dataset: xarray Dataset
        variable_name: Name of the variable to extract bounds for
        target_crs: Target coordinate reference system

    Returns:
        BoundingBox object if bounds can be extracted, None otherwise
    """
    try:
        # Get the grid system for this variable (run in thread to avoid blocking)
        grid = await async_run(guess_grid_system, dataset, variable_name)

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
    except VariableNotFoundError as e:
        raise e

    except Exception as e:
        logger.error(f"Failed to transform bounds: {e}")
        return None


async def create_tileset_for_tms(
    dataset: Dataset,
    tms_id: str,
    layer_extents: dict[str, dict[str, Any]],
    title: str,
    description: str,
    keywords: list[str],
    dataset_attrs: dict[str, Any],
    styles: list[Style],
) -> TilesetSummary | None:
    """Create a tileset summary for a specific tile matrix set

    Args:
        dataset: xarray Dataset
        tms_id: Tile matrix set identifier
        layer_extents: Pre-computed layer extents for all variables
        title: Dataset title
        description: Dataset description
        keywords: Dataset keywords
        dataset_attrs: Dataset attributes
        styles: Available styles

    Returns:
        TilesetSummary object if tile matrix set exists, None otherwise
    """
    if tms_id not in TILE_MATRIX_SETS:
        return None

    tms_summary = TILE_MATRIX_SET_SUMMARIES[tms_id]()

    # Create layers for each data variable
    layers = []
    for var_name, var_data in dataset.data_vars.items():
        # Skip scalar variables
        if var_data.ndim == 0:
            continue
        extents = layer_extents[var_name]

        # Extract variable-specific bounding box, fallback to dataset bounds
        var_bounding_box = await extract_variable_bounding_box(
            dataset, var_name, tms_summary.crs
        )

        layer = Layer(
            id=var_name,
            title=str(var_data.attrs.get("long_name", var_name)),
            description=var_data.attrs.get("description", ""),
            dataType=DataType.COVERAGE,
            boundingBox=var_bounding_box,
            crs=tms_summary.crs,
            links=[
                Link(
                    href=f"./{tms_id}/{{tileMatrix}}/{{tileRow}}/{{tileCol}}?variables={var_name}",
                    rel="item",
                    type="image/png",
                    title=f"Tiles for {var_name}",
                    templated=True,
                )
            ],
            extents=extents,
        )
        layers.append(layer)

    # Define tile matrix limits
    tileMatrixSetLimits = await get_tile_matrix_limits(tms_id, dataset)

    tileset = TilesetSummary(
        title=f"{title} - {tms_id}",
        description=description or f"Tiles for {title} in {tms_id} projection",
        tileMatrixSetURI=tms_summary.uri,
        crs=tms_summary.crs,
        dataType=DataType.MAP,
        links=[
            Link(
                href=f"./{tms_id}",
                rel="self",
                type="application/json",
                title=f"Tileset metadata for {tms_id}",
            ),
            Link(
                href=f"/tileMatrixSets/{tms_id}",
                rel="http://www.opengis.net/def/rel/ogc/1.0/tiling-scheme",
                type="application/json",
                title=f"Definition of {tms_id}",
            ),
        ],
        tileMatrixSetLimits=tileMatrixSetLimits,
        layers=layers if layers else None,
        keywords=keywords if keywords else None,
        attribution=dataset_attrs.get("attribution"),
        license=dataset_attrs.get("license"),
        version=dataset_attrs.get("version"),
        pointOfContact=dataset_attrs.get("contact"),
        mediaTypes=["image/png", "image/jpeg"],
        styles=styles,
    )
    return tileset
