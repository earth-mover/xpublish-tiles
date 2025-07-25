"""OGC Tiles API XPublish Plugin"""

from enum import Enum

from fastapi import APIRouter, HTTPException
from xpublish import Dependencies, Plugin, hookimpl

from xpublish_tiles.xpublish.tiles.models import (
    ConformanceDeclaration,
    Link,
    TileMatrixSet,
    TileMatrixSets,
    TileSetMetadata,
    TilesLandingPage,
)
from xpublish_tiles.xpublish.tiles.tile_matrix import (
    TILE_MATRIX_SET_SUMMARIES,
    TILE_MATRIX_SETS,
    extract_tile_bbox_and_crs,
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
            summaries = [
                summary_func() for summary_func in TILE_MATRIX_SET_SUMMARIES.values()
            ]
            return TileMatrixSets(tileMatrixSets=summaries)

        @router.get("/tileMatrixSets/{tileMatrixSetId}", response_model=TileMatrixSet)
        async def get_tile_matrix_set(tileMatrixSetId: str):
            """Get specific tile matrix set definition"""
            if tileMatrixSetId not in TILE_MATRIX_SETS:
                raise HTTPException(
                    status_code=404,
                    detail=f"Tile matrix set '{tileMatrixSetId}' not found",
                )

            return TILE_MATRIX_SETS[tileMatrixSetId]()

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
            try:
                bbox, crs = extract_tile_bbox_and_crs(
                    tileMatrixSetId, tileMatrix, tileRow, tileCol
                )
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

            # TODO: Pass bbox and crs to rendering pipeline
            return {
                "message": f"Tile {tileMatrixSetId}/{tileMatrix}/{tileRow}/{tileCol}",
                "tileMatrixSetId": tileMatrixSetId,
                "tileMatrix": tileMatrix,
                "tileRow": tileRow,
                "tileCol": tileCol,
                "bbox": bbox,
                "crs": crs,
            }

        return router
