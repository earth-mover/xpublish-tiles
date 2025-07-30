"""OGC Tiles API XPublish Plugin"""

from enum import Enum
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from xpublish import Dependencies, Plugin, hookimpl

from xarray import Dataset
from xpublish_tiles.xpublish.tiles.metadata import create_tileset_metadata
from xpublish_tiles.xpublish.tiles.tile_matrix import (
    TILE_MATRIX_SET_SUMMARIES,
    TILE_MATRIX_SETS,
    extract_dataset_bounds,
    extract_dimension_extents,
    extract_tile_bbox_and_crs,
    get_all_tile_matrix_set_ids,
    get_tile_matrix_limits,
)
from xpublish_tiles.xpublish.tiles.types import (
    ConformanceDeclaration,
    DataType,
    Layer,
    Link,
    TileMatrixSet,
    TileMatrixSets,
    TileSetMetadata,
    TilesetsList,
    TilesetSummary,
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
            dataset_bounds = extract_dataset_bounds(dataset)

            # Get dataset metadata
            dataset_attrs = dataset.attrs
            title = dataset_attrs.get("title", "Dataset")
            description = dataset_attrs.get("description", "")
            keywords = dataset_attrs.get("keywords", "")
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(",") if k.strip()]
            elif not isinstance(keywords, list):
                keywords = []

            # Create one tileset entry per supported tile matrix set
            supported_tms = get_all_tile_matrix_set_ids()

            for tms_id in supported_tms:
                if tms_id in TILE_MATRIX_SETS:
                    tms_summary = TILE_MATRIX_SET_SUMMARIES[tms_id]()

                    # Create layers for each data variable
                    layers = []
                    for var_name, var_data in dataset.data_vars.items():
                        # Extract dimension information for this variable
                        dimensions = extract_dimension_extents(var_data)

                        layer = Layer(
                            id=var_name,
                            title=var_data.attrs.get("long_name", var_name),
                            description=var_data.attrs.get("description", ""),
                            dataType=DataType.COVERAGE,
                            boundingBox=dataset_bounds,
                            crs=tms_summary.crs,
                            dimensions=dimensions if dimensions else None,
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

                    # Define tile matrix limits
                    tileMatrixSetLimits = get_tile_matrix_limits(tms_id)

                    tileset = TilesetSummary(
                        title=f"{title} - {tms_id}",
                        description=description
                        or f"Tiles for {title} in {tms_id} projection",
                        tileMatrixSetURI=tms_summary.uri,
                        crs=tms_summary.crs,
                        dataType=DataType.MAP,
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
        async def get_dataset_tileset_metadata(
            tileMatrixSetId: str,
            dataset: Dataset = Depends(deps.dataset),  # noqa: B008
        ):
            """Get tileset metadata for this dataset"""
            try:
                return create_tileset_metadata(dataset, tileMatrixSetId)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e

        @router.get("/{tileMatrixSetId}/{tileMatrix}/{tileRow}/{tileCol}")
        async def get_dataset_tile(
            request: Request,
            tileMatrixSetId: str,
            tileMatrix: int,
            tileRow: int,
            tileCol: int,
            variables: list[str],
            colorscalerange: str,
            style: str = "raster/default",
            width: int = 256,
            height: int = 256,
            f: Literal["image/png", "image/jpeg"] = "image/png",
            dataset: Dataset = Depends(deps.dataset),  # noqa: B008
        ):
            """Get individual tile from this dataset"""
            try:
                bbox, crs = extract_tile_bbox_and_crs(
                    tileMatrixSetId, tileMatrix, tileRow, tileCol
                )
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e

            # parsed_colorscalerange = parse_colorscalerange(colorscalerange)
            # parsed_style, cmap = parse_style(style)
            # parsed_image_format = parse_image_format(f)
            # render_params = QueryParams(
            #     variables=variables,
            #     style=parsed_style,
            #     colorscalerange=parsed_colorscalerange,
            #     cmap=cmap,
            #     crs=crs,
            #     bbox=bbox,
            #     width=width,
            #     height=height,
            #     format=parsed_image_format,
            #     selectors={},
            # )
            # buffer = await pipeline(dataset, render_params)

            # return StreamingResponse(
            #     buffer,
            #     media_type="image/png",
            # )

        return router
