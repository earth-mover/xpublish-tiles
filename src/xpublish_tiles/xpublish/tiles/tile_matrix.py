"""Tile matrix set definitions for OGC Tiles API"""

from xpublish_tiles.xpublish.tiles.models import (
    Link,
    TileMatrix,
    TileMatrixSet,
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
