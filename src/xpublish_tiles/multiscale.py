"""GeoZarr multiscale detection and level selection utilities.

This module provides functions for working with GeoZarr-conformant multiscale
DataTrees, including:
- Detection of multiscale pyramids
- Automatic level selection based on tile zoom
- Dataset extraction from DataTrees
"""

from __future__ import annotations

import morecantile

import xarray as xr
from xarray import DataTree


def is_multiscale(tree: DataTree) -> bool:
    """Check if DataTree is a GeoZarr multiscale pyramid.

    A DataTree is considered a GeoZarr multiscale if:
    - Root has a `multiscales` attribute with a `layout` list
    - Layout entries have `asset` keys pointing to existing children
    - Each layout entry has `spatial:transform` (resolution info)

    Parameters
    ----------
    tree : DataTree
        The DataTree to check.

    Returns
    -------
    bool
        True if the tree is a valid GeoZarr multiscale pyramid.
    """
    multiscales = tree.attrs.get("multiscales")
    if not multiscales:
        return False

    layout = multiscales.get("layout")
    if not isinstance(layout, list) or len(layout) == 0:
        return False

    # Verify at least one asset exists as a child and has spatial:transform
    for entry in layout:
        asset = entry.get("asset")
        if asset is not None and asset in tree.children:
            if "spatial:transform" in entry:
                return True

    return False


def get_layout_levels(tree: DataTree) -> list[dict]:
    """Get the layout levels from a multiscale DataTree, sorted by resolution.

    Levels are sorted from finest (highest resolution, smallest pixel size)
    to coarsest (lowest resolution, largest pixel size).

    Parameters
    ----------
    tree : DataTree
        A multiscale DataTree.

    Returns
    -------
    list[dict]
        Layout entries sorted by resolution (finest first).

    Raises
    ------
    ValueError
        If the tree is not a valid multiscale pyramid.
    """
    if not is_multiscale(tree):
        msg = "DataTree is not a valid GeoZarr multiscale pyramid"
        raise ValueError(msg)

    layout = tree.attrs["multiscales"]["layout"]

    # Filter to only valid entries with spatial:transform
    valid_entries = [
        entry
        for entry in layout
        if entry.get("asset") in tree.children and "spatial:transform" in entry
    ]

    # Sort by resolution (pixel size). The affine transform is [a, b, c, d, e, f]
    # where a is X resolution and e is Y resolution (usually negative for north-up).
    # We use abs(a) as the pixel size for sorting.
    def get_pixel_size(entry: dict) -> float:
        transform = entry["spatial:transform"]
        return abs(transform[0])  # X resolution

    return sorted(valid_entries, key=get_pixel_size)


def select_level_for_zoom(
    tree: DataTree,
    tms: morecantile.TileMatrixSet,
    zoom: int,
) -> str:
    """Select the best resolution level for a given tile zoom.

    Strategy: Choose the finest level whose resolution is >= the tile pixel size.
    This avoids upscaling data while minimizing oversampling.

    Parameters
    ----------
    tree : DataTree
        A multiscale DataTree.
    tms : morecantile.TileMatrixSet
        The tile matrix set being used.
    zoom : int
        The requested zoom level.

    Returns
    -------
    str
        The asset path (child name) for the selected level.

    Raises
    ------
    ValueError
        If the tree is not a valid multiscale pyramid or has no valid levels.
    """
    levels = get_layout_levels(tree)
    if not levels:
        msg = "No valid resolution levels found in multiscale pyramid"
        raise ValueError(msg)

    # Get tile pixel size at this zoom level
    tile_matrix = tms.matrix(zoom)
    tile_pixel_size = tile_matrix.cellSize

    # Find the finest level whose pixel size is >= tile pixel size
    # Levels are sorted finest (smallest pixel) to coarsest (largest pixel)
    selected = levels[0]  # Default to finest level

    for entry in levels:
        level_pixel_size = abs(entry["spatial:transform"][0])
        if level_pixel_size <= tile_pixel_size:
            selected = entry
            break

    return selected["asset"]


def get_dataset(
    tree: DataTree,
    *,
    zoom: int | None = None,
    tms: morecantile.TileMatrixSet | None = None,
) -> xr.Dataset:
    """Extract the appropriate Dataset from a DataTree.

    Behavior depends on the tree structure and parameters:
    - Multiscale + zoom + tms: Select level based on zoom, return that dataset
    - Multiscale + no zoom: Return the finest (highest resolution) level
    - Not multiscale: Return root dataset if present
    - No root dataset and not multiscale: Raise ValueError

    Parameters
    ----------
    tree : DataTree
        The DataTree to extract from.
    zoom : int, optional
        The tile zoom level for level selection.
    tms : morecantile.TileMatrixSet, optional
        The tile matrix set for resolution calculation.

    Returns
    -------
    xr.Dataset
        The extracted dataset.

    Raises
    ------
    ValueError
        If the tree has no root dataset and is not a valid multiscale pyramid.
    """
    if is_multiscale(tree):
        if zoom is not None and tms is not None:
            level_name = select_level_for_zoom(tree, tms, zoom)
        else:
            # Return finest level (first in sorted list)
            levels = get_layout_levels(tree)
            level_name = levels[0]["asset"]

        return tree[level_name].to_dataset()

    # Not multiscale - try to return root dataset
    root_ds = tree.to_dataset()
    if root_ds.data_vars:
        return root_ds

    # Empty root and not multiscale
    msg = (
        "DataTree has no data at root and is not a valid GeoZarr multiscale pyramid. "
        "Cannot extract dataset."
    )
    raise ValueError(msg)
