import functools
import re
from typing import Any

import morecantile.models
import pyproj.exceptions

import xarray as xr
from xarray import Dataset
from xpublish_tiles.grids import (
    CubedSphere,
    GridSystem,
    Healpix,
    detect_grids,
)
from xpublish_tiles.logger import logger
from xpublish_tiles.render import RenderRegistry
from xpublish_tiles.xpublish.tiles.tile_matrix import (
    TILE_MATRIX_SET_SUMMARIES,
    TILE_MATRIX_SETS,
    extract_dimension_extents,
    get_tile_matrix_limits,
)
from xpublish_tiles.xpublish.tiles.types import (
    BoundingBox,
    DataType,
    Layer,
    Link,
    Style,
    TileSetMetadata,
    TilesetSummary,
)

# Result of ``allowed_styles`` is dataset-wide and stable. Memoize per dataset
# so repeat /tiles/ requests skip grid detection entirely.
_ALLOWED_STYLES_CACHE: dict[str, list[str]] = {}


def allowed_styles(
    dataset: Dataset | None = None,
    *,
    var_grids: dict[str, GridSystem] | None = None,
) -> list[str]:
    """Return the style IDs supported by ``dataset``'s grid.

    Healpix and Faceted grids (e.g. cubed sphere) only support ``polygons``;
    every other grid supports both ``raster`` and ``polygons``.

    ``var_grids`` may be a precomputed ``detect_grids`` mapping; passing it
    avoids re-running grid detection on the hot path.
    """
    if dataset is None:
        return ["raster", "polygons"]

    xpublish_id = dataset.attrs.get("_xpublish_id")
    if xpublish_id is not None and xpublish_id in _ALLOWED_STYLES_CACHE:
        return _ALLOWED_STYLES_CACHE[xpublish_id]

    if var_grids is None:
        var_grids = detect_grids(dataset)

    result = ["raster", "polygons"]
    if any(isinstance(grid, (Healpix, CubedSphere)) for grid in var_grids.values()):
        result = ["polygons"]

    if xpublish_id is not None:
        _ALLOWED_STYLES_CACHE[xpublish_id] = result
    return result


def get_styles(
    dataset: Dataset | None = None,
    *,
    var_grids: dict[str, GridSystem] | None = None,
) -> list[Style]:
    """Return supported styles for ``dataset``'s grid."""
    allowed = set(allowed_styles(dataset, var_grids=var_grids))
    styles: list[Style] = []
    for style_id in RenderRegistry.all():
        if style_id not in allowed:
            continue
        styles.extend(_styles_for_renderer(style_id))
    return styles


@functools.cache
def _styles_for_renderer(style_id: str) -> tuple[Style, ...]:
    renderer_cls = RenderRegistry.all()[style_id]
    default_variant = renderer_cls.default_variant()
    default_style_info = renderer_cls.describe_style("default")
    default_style_info["title"] = (
        f"{renderer_cls.style_id().title()} - Default ({default_variant.title()})"
    )
    default_style_info["description"] = (
        f"Default {renderer_cls.style_id()} rendering (alias for {default_variant})"
    )
    styles = [
        Style(
            id=default_style_info["id"],
            title=default_style_info["title"],
            description=default_style_info["description"],
        )
    ]
    for variant in renderer_cls.supported_variants():
        style_info = renderer_cls.describe_style(variant)
        styles.append(
            Style(
                id=style_info["id"],
                title=style_info["title"],
                description=style_info["description"],
            )
        )
    return tuple(styles)


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
        styles=get_styles(dataset),
    )


async def extract_dataset_extents(
    dataset: Dataset,
    variable_name: str,
    *,
    cf_coords: dict | None = None,
) -> dict[str, dict[str, Any]]:
    """Format ``variable_name``'s non-spatial dimension extents as OGC extents."""
    dimensions = await extract_dimension_extents(
        dataset, variable_name, cf_coords=cf_coords
    )

    extents: dict[str, dict[str, Any]] = {}
    for dim in dimensions:
        extent_dict: dict[str, Any] = {"interval": dim.extent}
        if dim.resolution is not None:
            extent_dict["resolution"] = dim.resolution
        if dim.units:
            extent_dict["units"] = dim.units
        if dim.description:
            extent_dict["description"] = dim.description
        if dim.default is not None:
            extent_dict["default"] = dim.default
        extents[dim.name] = extent_dict

    return extents


def _pandas_freq_to_iso8601(freq: str) -> str | None:
    """Convert pandas frequency string to ISO 8601 duration format.

    Args:
        freq: Pandas frequency string (e.g., 'h', 'D', 'MS', 'YS-JAN')

    Returns:
        ISO 8601 duration string (e.g., 'PT1H', 'P1D', 'P1M', 'P1Y') or None if unknown
    """
    # Extract numeric prefix (e.g., '3' from '3h', '10' from '10YS')
    match = re.match(r"^(\d*)(.+)$", freq)
    if not match:
        return None

    count_str, base = match.groups()
    count = int(count_str) if count_str else 1

    # Normalize base by removing anchors (e.g., 'YS-JAN' -> 'YS', 'W-SUN' -> 'W')
    base_normalized = base.split("-")[0]

    # Map pandas aliases to ISO 8601
    # Time-based (use PT prefix)
    if base_normalized in ("h", "H"):
        return f"PT{count}H"
    if base_normalized in ("min", "T"):
        return f"PT{count}M"
    if base_normalized in ("s", "S"):
        return f"PT{count}S"

    # Date-based (use P prefix)
    if base_normalized == "D":
        return f"P{count}D"
    if base_normalized == "W":
        return f"P{count * 7}D"
    if base_normalized in ("MS", "ME", "M"):
        return f"P{count}M"
    if base_normalized in ("QS", "QE", "Q"):
        return f"P{count * 3}M"
    if base_normalized in ("YS", "YE", "Y", "A", "AS", "AE"):
        return f"P{count}Y"

    return None


def _calculate_temporal_resolution(values: xr.DataArray) -> str | None:
    """Calculate temporal resolution from datetime values.

    Uses xr.infer_freq() which supports both numpy datetime64 and cftime types.
    Returns ISO 8601 duration format for regular frequencies, None for irregular
    or undeterminable frequencies.

    Args:
        values: xarray DataArray with datetime-like values

    Returns:
        ISO 8601 duration string (e.g., 'PT1H', 'P1D', 'P1M', 'P1Y') or None
    """
    # Need at least 3 values for xr.infer_freq
    if not hasattr(values, "size") or values.size < 3:
        return None

    try:
        freq = xr.infer_freq(values)
        if freq is None:
            return None
        return _pandas_freq_to_iso8601(freq)
    except Exception:
        # TypeError: not datetime-like, ValueError: not enough values or not 1D
        return None


def extract_variable_bounding_box(
    target_crs: str | morecantile.models.CRS,
    *,
    grid: GridSystem,
) -> BoundingBox | None:
    """Transform ``grid``'s bounding box into ``target_crs``.

    Returns ``None`` if the projection transform fails.
    """
    # ``morecantile.CRS.srs`` returns the cached pyproj ``_srs`` string;
    # ``to_epsg`` round-trips through the EPSG database and is unexpectedly
    # expensive when called per variable.
    if isinstance(target_crs, morecantile.models.CRS):
        target_crs_str = target_crs.srs
    else:
        target_crs_str = target_crs

    try:
        transformed_bounds = grid.transform_bbox(target_crs_str)
    except pyproj.exceptions.ProjError as e:
        logger.error("Failed to transform bounds", exc_info=e)
        return None

    return BoundingBox(
        lowerLeft=[transformed_bounds[0], transformed_bounds[1]],
        upperRight=[transformed_bounds[2], transformed_bounds[3]],
        crs=target_crs,
    )


async def create_tileset_for_tms(
    dataset: Dataset,
    tms_id: str,
    layer_extents: dict[str, dict[str, Any]],
    title: str,
    description: str,
    keywords: list[str],
    dataset_attrs: dict[str, Any],
    styles: list[Style],
    var_grids: dict[str, GridSystem],
    *,
    cf_coords: dict | None = None,
) -> TilesetSummary | None:
    """Create a tileset summary for a specific tile matrix set

    Args:
        dataset: xarray Dataset (coarsest level for multiscale datasets)
        tms_id: Tile matrix set identifier
        layer_extents: Pre-computed layer extents for all variables
        title: Dataset title
        description: Dataset description
        keywords: Dataset keywords
        dataset_attrs: Dataset attributes
        styles: Available styles
        var_grids: Renderable variable -> grid mapping (from ``detect_grids``)

    Returns:
        TilesetSummary object if tile matrix set exists, None otherwise
    """
    if tms_id not in TILE_MATRIX_SETS:
        return None

    tms_summary = TILE_MATRIX_SET_SUMMARIES[tms_id]()

    # Create layers for each renderable data variable
    layers = []
    for var_name in var_grids:
        if var_name not in layer_extents:
            continue
        var_data = dataset[var_name]
        extents = layer_extents[var_name]

        # The transform is memoized per crs on the grid (transform_bbox), so
        # variables sharing a grid don't re-transform.
        var_bounding_box = extract_variable_bounding_box(
            tms_summary.crs, grid=var_grids[var_name]
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
                ),
                Link(
                    href=f"./legend?variables={var_name}",
                    rel="ogc-rel:legend",
                    type="image/png",
                    title=f"Legend for {var_name}",
                ),
            ],
            extents=extents,
        )
        layers.append(layer)

    # Pass a known-renderable variable so limits aren't computed from an
    # ancillary variable whose grid can't be detected.
    tileMatrixSetLimits = await get_tile_matrix_limits(
        tms_id,
        dataset,
        representative_var=next(iter(var_grids)),
        cf_coords=cf_coords,
    )

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
