import math
from collections.abc import Hashable
from dataclasses import dataclass

import morecantile
from pyproj import CRS

import xarray as xr
from xarray import DataTree
from xpublish_tiles.config import config
from xpublish_tiles.logger import logger
from xpublish_tiles.types import OverviewSelectionStrategy


@dataclass
class ResolutionLevel:
    """A dataset at a specific resolution level."""

    path: str | None  # None for root, child name for children
    dataset: xr.Dataset
    pixel_size: float


def get_pixel_size(ds: xr.Dataset) -> float | None:
    """Get pixel size from spatial:transform attribute.

    Per GeoZarr spec, array-level attrs override group-level attrs.
    Checks data variable attrs first, then falls back to dataset attrs.
    """
    transform = None
    # Check array-level attrs first (GeoZarr: arrays override group)
    # Use _variables to avoid expensive DataArray construction
    for var in ds.data_vars:
        transform = ds._variables[var].attrs.get("spatial:transform")
        if transform is not None:
            break
    # Fall back to dataset/group-level attrs
    if transform is None:
        transform = ds.attrs.get("spatial:transform")
    if transform and len(transform) >= 1:
        return abs(transform[0])  # X resolution
    return None


def get_crs(ds: xr.Dataset) -> CRS | None:
    """Get CRS from proj: attributes.

    Per GeoZarr spec, proj: attributes are on the group level.
    """
    if "proj:code" in ds.attrs:
        try:
            return CRS.from_user_input(ds.attrs["proj:code"])
        except Exception as e:
            logger.error(f"Failed to parse proj:code {ds.attrs['proj:code']!r}: {e}")
    if "proj:wkt2" in ds.attrs:
        try:
            return CRS.from_wkt(ds.attrs["proj:wkt2"])
        except Exception as e:
            logger.error(f"Failed to parse proj:wkt2: {e}")
    return None


def _pixel_size_in_tms_units(
    pixel_size: float, data_crs: CRS | None, tms: morecantile.TileMatrixSet
) -> float:
    """Convert pixel size to TMS units for comparison.

    If data CRS is geographic (degrees) and TMS is projected (meters),
    convert using approximate meters per degree at the equator.
    """
    if data_crs is None:
        return pixel_size

    tms_crs = CRS.from_user_input(tms.crs)

    # If both are in same units, no conversion needed
    if data_crs.is_geographic == tms_crs.is_geographic:
        return pixel_size

    # Data is geographic (degrees), TMS is projected (meters)
    if data_crs.is_geographic and not tms_crs.is_geographic:
        # Approximate conversion: 1 degree ≈ 111,000 meters at equator
        return pixel_size * 111_000

    # Data is projected (meters), TMS is geographic (degrees)
    if not data_crs.is_geographic and tms_crs.is_geographic:
        return pixel_size / 111_000

    return pixel_size


def scan_resolution_levels(tree: DataTree) -> list[ResolutionLevel]:
    """Scan all datasets in tree and return sorted resolution levels.

    Iterates over all nodes in the tree (root and all descendants at any depth)
    and collects those with data variables and spatial:transform attributes.

    Returns levels sorted from finest (smallest pixel size) to coarsest.
    """
    levels: list[ResolutionLevel] = []

    for path, node in tree.subtree_with_keys:
        ds = node.dataset
        if ds is None or not ds.data_vars:
            continue

        pixel_size = get_pixel_size(ds)
        if pixel_size is None:
            continue

        # path is "." for root, otherwise the relative path
        level_path = None if path == "." else path
        levels.append(ResolutionLevel(path=level_path, pixel_size=pixel_size, dataset=ds))

    # Sort by pixel size (finest/smallest first)
    levels.sort(key=lambda x: x.pixel_size)

    return levels


def get_coarsest_level(tree: DataTree) -> ResolutionLevel | None:
    """Get the coarsest (lowest resolution) level from a DataTree.

    Used for calculating minzoom in tilejson - the coarsest overview
    determines the lowest zoom level that can be rendered.
    """
    levels = scan_resolution_levels(tree)
    if not levels:
        return None
    # levels are sorted finest to coarsest, so last is coarsest
    return levels[-1]


def get_coarsest_level_for(tree: DataTree, name: Hashable) -> ResolutionLevel | None:
    """Coarsest level that holds ``name``. Overview levels may carry fewer variables than the finest."""
    for level in reversed(scan_resolution_levels(tree)):
        if name in level.dataset.data_vars:
            return level
    return None


def get_resolution_level(
    tree: DataTree,
    *,
    zoom: int | None = None,
    tms: morecantile.TileMatrixSet | None = None,
    strategy: OverviewSelectionStrategy | None = None,
) -> ResolutionLevel | None:
    """Get the appropriate resolution level from a DataTree.

    Behavior:
    - If zoom + tms provided: Select a level according to ``strategy``, which
      defaults to the ``overview_selection_strategy`` config value.
      NEAREST picks the level whose pixel size is nearest the tile's pixel
      size in log space, so the crossover between adjacent levels sits at the
      geometric mean of their pixel sizes. Ties break toward the coarser
      level: oversampling costs far more (bytes read + decode) than the
      slight blur of upsampling.
      COARSER picks the finest level that is still no finer than the tile,
      FINER the coarsest level that is still no coarser than the tile. Both
      fall back to the closest level available when the tile's pixel size
      falls outside the range of levels.
    - If no zoom: Return finest (highest resolution) level available
    - If no valid levels found: Return None

    Returns ResolutionLevel with dataset and path info, or None if no levels found.
    """
    levels = scan_resolution_levels(tree)

    if not levels:
        return None

    # No zoom specified - return finest level (first in sorted list)
    if zoom is None or tms is None:
        return levels[0]

    if strategy is None:
        strategy = OverviewSelectionStrategy(config.get("overview_selection_strategy"))

    # Select best level for the requested zoom
    data_crs = get_crs(levels[0].dataset)
    tile_pixel_size = tms.matrix(zoom).cellSize

    def _log_ratio(level: ResolutionLevel) -> float:
        pixel_size_tms = _pixel_size_in_tms_units(level.pixel_size, data_crs, tms)
        # quantize so float noise can't flip an exact match to the wrong side
        return round(math.log(pixel_size_tms / tile_pixel_size), 9)

    match strategy:
        case OverviewSelectionStrategy.NEAREST:
            # min keeps the first of equal keys; coarsest-first makes ties resolve coarse
            return min(reversed(levels), key=lambda level: abs(_log_ratio(level)))
        case OverviewSelectionStrategy.COARSER:
            # levels are finest-first, so the first non-finer level is the
            # finest one that is still at least as coarse as the tile
            return next(
                (level for level in levels if _log_ratio(level) >= 0),
                levels[-1],
            )
        case OverviewSelectionStrategy.FINER:
            return next(
                (level for level in reversed(levels) if _log_ratio(level) <= 0),
                levels[0],
            )


def assign_leaf_xpublish_ids(tree: DataTree) -> None:
    """Give each child node a distinct, path-qualified ``_xpublish_id``.

    Multiscale levels share dimension names, so the grid cache key
    (``(_xpublish_id, dim names)``) collides across levels unless each leaf
    carries a distinct id. Without this, the grid is rebuilt on every tile.
    No-op when the root has no ``_xpublish_id``.
    """
    root_id = tree.attrs.get("_xpublish_id")
    if root_id is None:
        return
    for path, node in tree.subtree_with_keys:
        if path == ".":
            continue
        node.attrs["_xpublish_id"] = f"{root_id}/{path}"


def get_dataset(
    tree: DataTree,
    *,
    zoom: int | None = None,
    tms: morecantile.TileMatrixSet | None = None,
    strategy: OverviewSelectionStrategy | None = None,
) -> xr.Dataset:
    """Extract the appropriate Dataset from a DataTree.

    Behavior:
    - If zoom + tms provided: Select a level according to ``strategy``; see
      ``get_resolution_level``.
    - If no zoom: Return finest (highest resolution) level available
    - If no valid levels found: Try root dataset, else raise ValueError
    """
    level = get_resolution_level(tree, zoom=zoom, tms=tms, strategy=strategy)

    if level is not None:
        return level.dataset

    # No levels with spatial:transform - fall back to root dataset
    root_ds = tree.to_dataset()
    if root_ds.data_vars:
        return root_ds

    # Empty root and no valid levels
    msg = "DataTree has no extractable dataset (no data at root or children with spatial:transform)"
    raise ValueError(msg)
