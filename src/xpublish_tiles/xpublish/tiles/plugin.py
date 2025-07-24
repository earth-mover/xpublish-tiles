"""OGC Tiles API XPublish Plugin"""

from enum import Enum

from fastapi import APIRouter
from xpublish import Dependencies, Plugin, hookimpl

from tiles.models import (
    ConformanceDeclaration,
    Link,
    TileMatrixSet,
    TileMatrixSets,
    TileSetMetadata,
    TilesLandingPage,
)


class TilesPlugin(Plugin):
    name: str = "tiles"

    app_router_prefix: str = "/tiles"
    app_router_tags: list[str | Enum] = ["tiles"]

    dataset_router_prefix: str = "/tiles"
    dataset_router_tags: list[str | Enum] = ["tiles"]

    @hookimpl
    def app_router(self, deps: Dependencies):
        """Global tiles endpoints"""
        router = APIRouter(prefix=self.app_router_prefix, tags=self.app_router_tags)

        @router.get("/conformance", response_model=ConformanceDeclaration)
        async def get_conformance():
            """OGC API conformance declaration"""
            return ConformanceDeclaration(
                conformsTo=[
                    "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/core",
                    "http://www.opengis.net/spec/ogcapi-common-2/1.0/conf/collections",
                    "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/core",
                ]
            )

        @router.get("/tileMatrixSets", response_model=TileMatrixSets)
        async def get_tile_matrix_sets():
            """List available tile matrix sets"""
            return TileMatrixSets(
                tileMatrixSets=[
                    {
                        "id": "WebMercatorQuad",
                        "title": "Web Mercator Quad",
                        "uri": "http://www.opengis.net/def/tilematrixset/OGC/1.0/WebMercatorQuad",
                        "crs": "http://www.opengis.net/def/crs/EPSG/0/3857",
                        "links": [
                            Link(
                                href="/tiles/tileMatrixSets/WebMercatorQuad",
                                rel="self",
                                type="application/json",
                                title="Web Mercator Quad tile matrix set",
                            )
                        ],
                    }
                ]
            )

        @router.get("/tileMatrixSets/{tileMatrixSetId}", response_model=TileMatrixSet)
        async def get_tile_matrix_set(tileMatrixSetId: str):
            """Get specific tile matrix set definition"""
            if tileMatrixSetId == "WebMercatorQuad":
                return TileMatrixSet(
                    id="WebMercatorQuad",
                    title="Web Mercator Quad",
                    uri="http://www.opengis.net/def/tilematrixset/OGC/1.0/WebMercatorQuad",
                    crs="http://www.opengis.net/def/crs/EPSG/0/3857",
                    tileMatrices=[
                        {
                            "id": "0",
                            "scaleDenominator": 559082264.029,
                            "topLeftCorner": [-20037508.3428, 20037508.3428],
                            "tileWidth": 256,
                            "tileHeight": 256,
                            "matrixWidth": 1,
                            "matrixHeight": 1,
                        },
                        {
                            "id": "1",
                            "scaleDenominator": 279541132.015,
                            "topLeftCorner": [-20037508.3428, 20037508.3428],
                            "tileWidth": 256,
                            "tileHeight": 256,
                            "matrixWidth": 2,
                            "matrixHeight": 2,
                        },
                    ],
                )
            else:
                # Return 404 or empty for unknown tile matrix sets
                return TileMatrixSet(
                    id=tileMatrixSetId,
                    crs="unknown",
                    tileMatrices=[],
                )

        return router

    @hookimpl
    def dataset_router(self, deps: Dependencies):
        """Dataset-specific tiles endpoints"""
        router = APIRouter(
            prefix=self.dataset_router_prefix, tags=self.dataset_router_tags
        )

        @router.get("/", response_model=TilesLandingPage)
        async def get_dataset_tiles_landing():
            """Dataset tiles landing page"""
            return TilesLandingPage(
                title="Dataset Tiles",
                description="Tiles for this dataset",
                links=[
                    Link(
                        href="./WebMercatorQuad",
                        rel="item",
                        type="application/json",
                        title="WebMercatorQuad tileset",
                    )
                ],
            )

        @router.get("/{tileMatrixSetId}", response_model=TileSetMetadata)
        async def get_dataset_tileset_metadata(tileMatrixSetId: str):
            """Get tileset metadata for this dataset"""
            return TileSetMetadata(
                title=f"Dataset tiles in {tileMatrixSetId}",
                tileMatrixSetURI=f"http://www.opengis.net/def/tilematrixset/OGC/1.0/{tileMatrixSetId}",
                crs="http://www.opengis.net/def/crs/EPSG/0/3857",
                dataType="map",
                links=[
                    Link(
                        href=f"./{tileMatrixSetId}/{{tileMatrix}}/{{tileRow}}/{{tileCol}}",
                        rel="item",
                        type="image/png",
                        title="Tile",
                    )
                ],
            )

        @router.get("/{tileMatrixSetId}/{tileMatrix}/{tileRow}/{tileCol}")
        async def get_dataset_tile(
            tileMatrixSetId: str, tileMatrix: int, tileRow: int, tileCol: int
        ):
            """Get individual tile from this dataset"""
            # Return placeholder response for now
            return {
                "message": f"Tile {tileMatrixSetId}/{tileMatrix}/{tileRow}/{tileCol}",
                "tileMatrixSetId": tileMatrixSetId,
                "tileMatrix": tileMatrix,
                "tileRow": tileRow,
                "tileCol": tileCol,
            }

        return router
