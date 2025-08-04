from typing import Any, Optional

import matplotlib.pyplot as plt
import pyproj.aoi

import xarray as xr
from xpublish_tiles.grids import Curvilinear, Rectilinear, guess_grid_system


def lower_case_keys(d: Any) -> dict[str, Any]:
    """Convert keys to lowercase, handling both dict and QueryParams objects"""
    if hasattr(d, "items"):
        return {k.lower(): v for k, v in d.items()}
    else:
        # Handle other dict-like objects
        return {k.lower(): v for k, v in dict(d).items()}


def get_available_raster_styles() -> list[dict[str, str]]:
    """Get all available raster styles based on matplotlib colormaps.

    Returns:
        List of style dictionaries with id, title, and description
    """
    styles = []

    # Add default raster style
    styles.append(
        {
            "id": "raster",
            "title": "Default Raster Style",
            "description": "Default raster rendering style",
        }
    )

    # Get all available matplotlib colormaps
    colormaps = sorted(plt.colormaps())

    for cmap_name in colormaps:
        # Skip reversed colormaps to avoid duplication (they end with '_r')
        if cmap_name.endswith("_r"):
            continue

        styles.append(
            {
                "id": f"raster/{cmap_name}",
                "title": f"Raster - {cmap_name.title()}",
                "description": f"Raster rendering using {cmap_name} colormap",
            }
        )

    return styles


def extract_robust_bounds(
    dataset: xr.Dataset, variable_name: Optional[str] = None
) -> Optional[pyproj.aoi.BBox]:
    """Extract geographic bounds using the same robust logic as the pipeline.

    This function uses the grid system detection and bounds extraction logic
    from the pipeline to ensure consistent bounds across tiles, WMS, and rendering.

    Args:
        dataset: xarray Dataset to extract bounds from
        variable_name: Optional variable name to extract bounds for. If None, uses first data variable.

    Returns:
        BBox object with geographic bounds (EPSG:4326) or None if bounds cannot be extracted
    """
    try:
        # Import cf_xarray to enable .cf accessor
        import cf_xarray as cfxr  # noqa: F401
        import pyproj

        # Use first data variable if none specified
        if variable_name is None:
            if not dataset.data_vars:
                return None
            variable_name = next(iter(dataset.data_vars))

        # Use the same grid system detection as the pipeline
        grid_system = guess_grid_system(dataset, variable_name)

        # Check if grid system has the required attributes (only Rectilinear and Curvilinear have them)
        if not isinstance(grid_system, Rectilinear | Curvilinear):
            return None

        # Get the bbox from the grid system (this handles all the edge cases)
        grid_bbox = grid_system.bbox

        # If the grid is already in geographic coordinates, return as-is
        if grid_system.crs.is_geographic:
            return grid_bbox

        # Transform to geographic coordinates (EPSG:4326) if needed
        transformer = pyproj.Transformer.from_crs(
            grid_system.crs, pyproj.CRS.from_epsg(4326), always_xy=True
        )

        # Transform the bbox corners
        west, south = transformer.transform(grid_bbox.west, grid_bbox.south)
        east, north = transformer.transform(grid_bbox.east, grid_bbox.north)

        return pyproj.aoi.BBox(west=west, south=south, east=east, north=north)

    except Exception:
        # If extraction fails, return None
        return None
