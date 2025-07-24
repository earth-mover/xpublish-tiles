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
