from typing import Any

import matplotlib.pyplot as plt


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
