from dataclasses import dataclass

import morecantile

import xarray as xr
from xarray import DataTree


@dataclass
class ResolutionLevel:
    """A dataset at a specific resolution level."""

    path: str | None  # None for root, child name for children
    pixel_size: float
    dataset: xr.Dataset


def get_spatial_transform(ds: xr.Dataset) -> list[float] | None:
    """Extract spatial:transform from dataset attributes."""
    return ds.attrs.get("spatial:transform")


def get_pixel_size(ds: xr.Dataset) -> float | None:
    """Get pixel size from dataset's spatial:transform attribute."""
    transform = get_spatial_transform(ds)
    if transform and len(transform) >= 1:
        return abs(transform[0])  # X resolution
    return None


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


def is_multiscale(tree: DataTree) -> bool:
    """Check if DataTree has multiple resolution levels.

    Returns True if there are 2+ datasets with spatial:transform attributes.
    This handles both:
    - GeoZarr convention (all levels in children)
    - Native-at-root (root has data, children have overviews)
    """
    levels = scan_resolution_levels(tree)
    return len(levels) >= 2


def _pixel_size_in_meters(pixel_size_degrees: float) -> float:
    """Convert pixel size from degrees to approximate meters at equator."""
    return pixel_size_degrees * 111_000


def select_level_for_zoom(
    tree: DataTree,
    tms: morecantile.TileMatrixSet,
    zoom: int,
) -> ResolutionLevel:
    """Select the best resolution level for a given tile zoom.

    Strategy: Choose the coarsest level whose pixel size is still finer than
    or equal to the tile pixel size.

    Falls back to finest level if all levels are coarser than tile pixels.
    """
    levels = scan_resolution_levels(tree)
    if not levels:
        msg = "No valid resolution levels found in tree"
        raise ValueError(msg)

    tile_matrix = tms.matrix(zoom)
    tile_pixel_size = tile_matrix.cellSize

    # Levels are sorted finest (smallest pixel) to coarsest (largest pixel)
    # Default to finest level (for when all are coarser than tile - need upscaling)
    selected = levels[0]

    # Iterate from coarsest to finest, find coarsest level still finer than tile
    for level in reversed(levels):
        level_meters = _pixel_size_in_meters(level.pixel_size)
        if level_meters <= tile_pixel_size:
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
    - If no zoom: Return finest (highest resolution) level available
    - If no valid levels found: Try root dataset, else raise ValueError
    """
    levels = scan_resolution_levels(tree)

    if levels:
        if zoom is not None and tms is not None:
            selected = select_level_for_zoom(tree, tms, zoom)
        else:
            # Return finest level (first in sorted list)
            selected = levels[0]
        return selected.dataset

    # No levels with spatial:transform - fall back to root dataset
    root_ds = tree.to_dataset()
    if root_ds.data_vars:
        return root_ds

    # Empty root and no valid levels
    msg = "DataTree has no extractable dataset (no data at root or children with spatial:transform)"
    raise ValueError(msg)
