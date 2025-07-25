"""Tile matrix set definitions for OGC Tiles API"""

from typing import Optional

from xarray import Dataset
from xpublish_tiles.xpublish.tiles.models import (
    BoundingBox,
    Link,
    TileMatrix,
    TileMatrixSet,
    TileMatrixSetLimit,
    TileMatrixSetSummary,
)


def get_web_mercator_quad() -> TileMatrixSet:
    """Get the complete WebMercatorQuad tile matrix set definition with all zoom levels 0-21"""

    # Base values for WebMercator
    origin_x = -20037508.3428
    origin_y = 20037508.3428
    tile_size = 256
    base_scale_denominator = 559082264.029

    tile_matrices = []

    # Generate all zoom levels from 0 to 21
    for zoom in range(22):  # 0 to 21 inclusive
        scale_denominator = base_scale_denominator / (2**zoom)
        matrix_size = 2**zoom

        tile_matrices.append(
            TileMatrix(
                id=str(zoom),
                scaleDenominator=scale_denominator,
                topLeftCorner=[origin_x, origin_y],
                tileWidth=tile_size,
                tileHeight=tile_size,
                matrixWidth=matrix_size,
                matrixHeight=matrix_size,
            )
        )

    return TileMatrixSet(
        id="WebMercatorQuad",
        title="Web Mercator Quad",
        uri="http://www.opengis.net/def/tilematrixset/OGC/1.0/WebMercatorQuad",
        crs="http://www.opengis.net/def/crs/EPSG/0/3857",
        tileMatrices=tile_matrices,
    )


def get_web_mercator_quad_summary() -> TileMatrixSetSummary:
    """Get summary information for WebMercatorQuad tile matrix set"""
    return TileMatrixSetSummary(
        id="WebMercatorQuad",
        title="Web Mercator Quad",
        uri="http://www.opengis.net/def/tilematrixset/OGC/1.0/WebMercatorQuad",
        crs="http://www.opengis.net/def/crs/EPSG/0/3857",
        links=[
            Link(
                href="/tiles/tileMatrixSets/WebMercatorQuad",
                rel="self",
                type="application/json",
                title="Web Mercator Quad tile matrix set",
            )
        ],
    )


# Registry of available tile matrix sets
TILE_MATRIX_SETS = {
    "WebMercatorQuad": get_web_mercator_quad,
}

TILE_MATRIX_SET_SUMMARIES = {
    "WebMercatorQuad": get_web_mercator_quad_summary,
}


def extract_tile_bbox_and_crs(
    tileMatrixSetId: str, tileMatrix: int, tileRow: int, tileCol: int
) -> tuple[list[float], str]:
    """Extract bounding box and CRS from tile coordinates.

    Args:
        tileMatrixSetId: ID of the tile matrix set
        tileMatrix: Zoom level/tile matrix ID
        tileRow: Row index of the tile
        tileCol: Column index of the tile

    Returns:
        tuple: (bbox as [minX, minY, maxX, maxY], crs)

    Raises:
        ValueError: If tile matrix set or tile matrix not found
    """
    if tileMatrixSetId not in TILE_MATRIX_SETS:
        raise ValueError(f"Tile matrix set '{tileMatrixSetId}' not found")

    tile_matrix_set = TILE_MATRIX_SETS[tileMatrixSetId]()

    tile_matrix_def = None
    for tm in tile_matrix_set.tileMatrices:
        if tm.id == str(tileMatrix):
            tile_matrix_def = tm
            break

    if not tile_matrix_def:
        raise ValueError(f"Tile matrix '{tileMatrix}' not found")

    origin_x, origin_y = tile_matrix_def.topLeftCorner
    tile_width = tile_matrix_def.tileWidth
    tile_height = tile_matrix_def.tileHeight

    pixel_size = tile_matrix_def.scaleDenominator * 0.00028

    min_x = origin_x + (tileCol * tile_width * pixel_size)
    max_x = origin_x + ((tileCol + 1) * tile_width * pixel_size)
    max_y = origin_y - (tileRow * tile_height * pixel_size)
    min_y = origin_y - ((tileRow + 1) * tile_height * pixel_size)

    bbox = [min_x, min_y, max_x, max_y]
    crs = tile_matrix_set.crs

    return bbox, crs


def extract_dataset_bounds(dataset: Dataset) -> Optional[BoundingBox]:
    """Extract geographic bounds from a dataset.

    TODO: This functionality may be handled by the tile rendering pipeline in the future.

    Args:
        dataset: xarray Dataset to extract bounds from

    Returns:
        BoundingBox object if bounds can be extracted, None otherwise
    """
    try:
        # Try to get bounds from dataset bounds attribute
        if hasattr(dataset, "bounds"):
            bounds = dataset.bounds
            return BoundingBox(
                lowerLeft=[float(bounds[0]), float(bounds[1])],
                upperRight=[float(bounds[2]), float(bounds[3])],
                crs="http://www.opengis.net/def/crs/EPSG/0/4326",
            )
        # Try to extract from lat/lon coordinates
        elif "lat" in dataset.coords and "lon" in dataset.coords:
            lat_min, lat_max = float(dataset.lat.min()), float(dataset.lat.max())
            lon_min, lon_max = float(dataset.lon.min()), float(dataset.lon.max())
            return BoundingBox(
                lowerLeft=[lon_min, lat_min],
                upperRight=[lon_max, lat_max],
                crs="http://www.opengis.net/def/crs/EPSG/0/4326",
                orderedAxes=["X", "Y"],
            )
    except Exception:
        # If we can't extract bounds, return None
        pass
    return None


def get_tile_matrix_limits(
    tms_id: str, zoom_levels: Optional[range] = None
) -> list[TileMatrixSetLimit]:
    """Generate tile matrix limits for the specified zoom levels.

    TODO: Calculate actual limits based on dataset bounds instead of full world coverage.

    Args:
        tms_id: Tile matrix set identifier
        zoom_levels: Range of zoom levels to generate limits for (default: 0-18)

    Returns:
        List of TileMatrixSetLimit objects
    """
    if zoom_levels is None:
        zoom_levels = range(19)  # 0-18

    limits = []
    for z in zoom_levels:
        max_tiles = 2**z - 1
        limits.append(
            TileMatrixSetLimit(
                tileMatrix=str(z),
                minTileRow=0,
                maxTileRow=max_tiles,
                minTileCol=0,
                maxTileCol=max_tiles,
            )
        )
    return limits


def get_all_tile_matrix_set_ids() -> list[str]:
    """Get list of all available tile matrix set IDs."""
    return list(TILE_MATRIX_SETS.keys())
