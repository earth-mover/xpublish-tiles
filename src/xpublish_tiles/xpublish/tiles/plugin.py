"""OGC Tiles API XPublish Plugin"""

from enum import Enum

from fastapi import APIRouter, Depends, HTTPException
from xpublish import Dependencies, Plugin, hookimpl

from xarray import Dataset
from xpublish_tiles.xpublish.tiles.models import (
    BoundingBox,
    ConformanceDeclaration,
    Layer,
    Link,
    TileMatrixSet,
    TileMatrixSetLimit,
    TileMatrixSets,
    TileSetMetadata,
    TilesetsList,
    TilesetSummary,
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
                    "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/core",
                    "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/tileset",
                    "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/tilesets-list",
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

        @router.get("/", response_model=TilesetsList)
        async def get_dataset_tiles_list(dataset: Dataset = Depends(deps.dataset)):  # noqa: B008
            """List of available tilesets for this dataset"""
            # Get dataset variables that can be tiled
            tilesets = []

            # Extract dataset bounds if available
            dataset_bounds = None
            try:
                # Try to get bounds from dataset coordinates
                if hasattr(dataset, "bounds"):
                    bounds = dataset.bounds
                    dataset_bounds = BoundingBox(
                        lowerLeft=[float(bounds[0]), float(bounds[1])],
                        upperRight=[float(bounds[2]), float(bounds[3])],
                        crs="http://www.opengis.net/def/crs/EPSG/0/4326",
                    )
                elif "lat" in dataset.coords and "lon" in dataset.coords:
                    lat_min, lat_max = float(dataset.lat.min()), float(dataset.lat.max())
                    lon_min, lon_max = float(dataset.lon.min()), float(dataset.lon.max())
                    dataset_bounds = BoundingBox(
                        lowerLeft=[lon_min, lat_min],
                        upperRight=[lon_max, lat_max],
                        crs="http://www.opengis.net/def/crs/EPSG/0/4326",
                        orderedAxes=["X", "Y"],
                    )
            except Exception:
                # If we can't extract bounds, that's okay
                pass

            # Get dataset metadata
            dataset_attrs = dataset.attrs
            title = dataset_attrs.get("title", "Dataset")
            description = dataset_attrs.get("description", "")
            keywords = dataset_attrs.get("keywords", "")
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(",") if k.strip()]
            elif not isinstance(keywords, list):
                keywords = []

            # For now, create one tileset entry per supported tile matrix set
            supported_tms = ["WebMercatorQuad"]  # Can be expanded

            for tms_id in supported_tms:
                if tms_id in TILE_MATRIX_SETS:
                    tms_summary = TILE_MATRIX_SET_SUMMARIES[tms_id]()

                    # Create layers for each data variable
                    layers = []
                    for var_name, var_data in dataset.data_vars.items():
                        layer = Layer(
                            id=var_name,
                            title=var_data.attrs.get("long_name", var_name),
                            description=var_data.attrs.get("description", ""),
                            dataType="coverage",
                            boundingBox=dataset_bounds,
                            crs=tms_summary.crs,
                            links=[
                                Link(
                                    href=f"./{tms_id}/{var_name}/{{tileMatrix}}/{{tileRow}}/{{tileCol}}",
                                    rel="item",
                                    type="image/png",
                                    title=f"Tiles for {var_name}",
                                    templated=True,
                                )
                            ],
                        )
                        layers.append(layer)

                    # Define tile matrix limits (example for zoom levels 0-18)
                    tileMatrixSetLimits = []
                    if tms_id == "WebMercatorQuad":
                        # Add limits for a few zoom levels as example
                        for z in range(19):
                            max_tiles = 2**z - 1
                            tileMatrixSetLimits.append(
                                TileMatrixSetLimit(
                                    tileMatrix=str(z),
                                    minTileRow=0,
                                    maxTileRow=max_tiles,
                                    minTileCol=0,
                                    maxTileCol=max_tiles,
                                )
                            )

                    tileset = TilesetSummary(
                        title=f"{title} - {tms_id}",
                        description=description
                        or f"Tiles for {title} in {tms_id} projection",
                        tileMatrixSetURI=tms_summary.uri,
                        crs=tms_summary.crs,
                        dataType="map",  # Could be "coverage" for gridded data
                        links=[
                            Link(
                                href=f"./{tms_id}",
                                rel="self",
                                type="application/json",
                                title=f"Tileset metadata for {tms_id}",
                            ),
                            Link(
                                href=f"/tileMatrixSets/{tms_id}",
                                rel="http://www.opengis.net/def/rel/ogc/1.0/tiling-scheme",
                                type="application/json",
                                title=f"Definition of {tms_id}",
                            ),
                        ],
                        tileMatrixSetLimits=tileMatrixSetLimits,
                        layers=layers if layers else None,
                        boundingBox=dataset_bounds,
                        keywords=keywords if keywords else None,
                        attribution=dataset_attrs.get("attribution"),
                        license=dataset_attrs.get("license"),
                        version=dataset_attrs.get("version"),
                        pointOfContact=dataset_attrs.get("contact"),
                        mediaTypes=["image/png", "image/jpeg"],
                    )
                    tilesets.append(tileset)

            return TilesetsList(tilesets=tilesets)

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
                raise HTTPException(status_code=404, detail=str(e)) from e

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
