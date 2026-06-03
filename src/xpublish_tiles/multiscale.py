from dataclasses import dataclass
from typing import TYPE_CHECKING

import morecantile
from pyproj import CRS

import xarray as xr
from xarray import DataTree
from xpublish_tiles.logger import logger

if TYPE_CHECKING:
    from xpublish_tiles.grids import GridSystem
    from xpublish_tiles.tiles_lib import get_min_zoom as _get_min_zoom


@dataclass
class ResolutionLevel:
    """A dataset at a specific resolution level."""

    path: str | None  # None for root, child name for children
    dataset: xr.Dataset
    pixel_size: float
    min_zoom: int | None = None

    def get_min_zoom(
        self,
        *,
        grid: "GridSystem",
        tms: morecantile.TileMatrixSet,
        variable: str,
        style: str,
        xpublish_id: str | None = None,
    ) -> int:
        """Get or compute the minimum renderable zoom for this level.

        Uses the cached min_zoom if available, otherwise computes it using
        tiles_lib.get_min_zoom and caches the result.
        """
        if self.min_zoom is not None:
            return self.min_zoom

        da = self.dataset[variable]
        self.min_zoom = _get_min_zoom(grid, tms, da, style, xpublish_id)
        return self.min_zoom


def get_pixel_size(ds: xr.Dataset) -> float | None:
    """Get pixel size from spatial:transform attribute.

    Per GeoZarr spec, array-level attrs override group-level attrs.
    Checks data variable attrs first, then falls back to dataset attrs.
    """
    transform = None
    # Check array-level attrs first (GeoZarr: arrays override group)
    for var in ds.data_vars:
        transform = ds[var].attrs.get("spatial:transform")
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
        ds = node.to_dataset()
        if not ds.data_vars:
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


def get_resolution_level(
    tree: DataTree,
    *,
    zoom: int | None = None,
    tms: morecantile.TileMatrixSet | None = None,
) -> ResolutionLevel | None:
    """Get the appropriate resolution level from a DataTree.

    Behavior:
    - If zoom + tms provided: Select best resolution level for that zoom
      by comparing each level's pixel size to the tile's pixel size.
      Selects the coarsest level that is still finer than the tile.
    - If no zoom: Return finest (highest resolution) level available
    - If no valid levels found: Return None

    Returns ResolutionLevel with dataset and path info, or None if no levels found.
    """
    levels = scan_resolution_levels(tree)

    if not levels:
        return None

    if zoom is not None and tms is not None:
        return _select_level_for_zoom_from_levels(levels, tms, zoom)

    # Return finest level (first in sorted list)
    return levels[0]


def _select_level_for_zoom_from_levels(
    levels: list[ResolutionLevel],
    tms: morecantile.TileMatrixSet,
    zoom: int,
) -> ResolutionLevel:
    """Select the best resolution level from pre-scanned levels.

    Internal helper that operates on already-scanned levels to avoid rescanning.
    """
    # Get CRS from the first level's dataset for unit conversion
    data_crs = get_crs(levels[0].dataset)

    tile_matrix = tms.matrix(zoom)
    tile_pixel_size = tile_matrix.cellSize

    # Levels are sorted finest (smallest pixel) to coarsest (largest pixel)
    # Default to finest level (for when all are coarser than tile - need upscaling)
    selected = levels[0]

    # Iterate from coarsest to finest, find coarsest level with pixel spacing
    # still finer than tile
    for level in reversed(levels):
        # Convert pixel size to TMS units for proper comparison
        pixel_size_tms = _pixel_size_in_tms_units(level.pixel_size, data_crs, tms)
        if pixel_size_tms <= tile_pixel_size:
            selected = level
            break

    return selected


def get_dataset(
    tree: DataTree,
    *,
    zoom: int | None = None,
    tms: morecantile.TileMatrixSet | None = None,
) -> xr.Dataset:
    """Extract the appropriate Dataset from a DataTree.

    Behavior:
    - If zoom + tms provided: Select best resolution level for that zoom
      by comparing each level's pixel size to the tile's pixel size.
      Selects the coarsest level that is still finer than the tile.
    - If no zoom: Return finest (highest resolution) level available
    - If no valid levels found: Try root dataset, else raise ValueError
    """
    level = get_resolution_level(tree, zoom=zoom, tms=tms)

    if level is not None:
        return level.dataset

    # No levels with spatial:transform - fall back to root dataset
    root_ds = tree.to_dataset()
    if root_ds.data_vars:
        return root_ds

    # Empty root and no valid levels
    msg = "DataTree has no extractable dataset (no data at root or children with spatial:transform)"
    raise ValueError(msg)
