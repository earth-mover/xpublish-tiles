import morecantile

import xarray as xr
from xarray import DataTree


def is_multiscale(tree: DataTree) -> bool:
    """Check if DataTree is a multiscale pyramid.

    A DataTree is considered a [GeoZarr] multiscale if:
    - Root has a `multiscales` attribute with a `layout` list
    - Layout entries have `asset` keys pointing to existing children
    - Each layout entry has `spatial:transform` (resolution info)
    """
    multiscales = tree.attrs.get("multiscales")
    if not multiscales:
        return False

    layout = multiscales.get("layout")
    if not isinstance(layout, list) or len(layout) == 0:
        return False

    # Verify at least one asset exists as a child and has spatial:transform attr
    for entry in layout:
        asset = entry.get("asset")
        if asset is not None and asset in tree.children:
            if "spatial:transform" in entry:
                return True

    return False


def get_layout_levels(tree: DataTree) -> list[dict]:
    """Get the layout levels from a multiscale DataTree, sorted by resolution.

    Levels are sorted from finest (highest resolution, smallest pixel size)
    to coarsest (lowest resolution, largest pixel size) per Zarr multiscale spec.
    """
    if not is_multiscale(tree):
        raise ValueError("DataTree is not a valid GeoZarr multiscale pyramid")

    layout = tree.attrs["multiscales"]["layout"]

    # Filter to only valid entries with spatial:transform
    valid_entries = [
        entry
        for entry in layout
        if entry.get("asset") in tree.children and "spatial:transform" in entry
    ]

    # Sort by resolution - smallest pixel size (finest) first
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

    Choose the finest level whose resolution is >= the tile pixel size.
    Avoids upscaling data while minimizing oversampling.
    """
    levels = get_layout_levels(tree)
    if not levels:
        raise ValueError("No valid resolution levels found in multiscale pyramid")

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
    raise ValueError(
        "DataTree has no data at root and is not a valid GeoZarr multiscale pyramid. "
        "Cannot extract dataset."
    )
