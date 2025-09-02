"""OGC Tiles API XPublish Plugin"""

import io
from enum import Enum
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from xpublish import Dependencies, Plugin, hookimpl

from xarray import Dataset
from xpublish_tiles.lib import TileTooBigError
from xpublish_tiles.logger import logger
from xpublish_tiles.pipeline import pipeline
from xpublish_tiles.render import RenderRegistry
from xpublish_tiles.types import QueryParams
from xpublish_tiles.utils import async_time_debug
from xpublish_tiles.xpublish.tiles.metadata import (
    create_tileset_metadata,
    extract_dataset_extents,
    extract_variable_bounding_box,
)
from xpublish_tiles.xpublish.tiles.tile_matrix import (
    TILE_MATRIX_SET_SUMMARIES,
    TILE_MATRIX_SETS,
    extract_tile_bbox_and_crs,
    get_all_tile_matrix_set_ids,
    get_tile_matrix_limits,
)
from xpublish_tiles.xpublish.tiles.types import (
    TILES_FILTERED_QUERY_PARAMS,
    ConformanceDeclaration,
    DataType,
    Layer,
    Link,
    Style,
    TileJSON,
    TileMatrixSet,
    TileMatrixSets,
    TileQuery,
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

        @router.get("/", response_model=TilesetsList, response_model_exclude_none=True)
        async def get_dataset_tiles_list(dataset: Dataset = Depends(deps.dataset)):  # noqa: B008
            """List of available tilesets for this dataset"""
            # Get dataset variables that can be tiled
            tilesets = []

            # Get dataset metadata
            dataset_attrs = dataset.attrs
            title = dataset_attrs.get("title", "Dataset")
            description = dataset_attrs.get("description", "")
            keywords = dataset_attrs.get("keywords", "")
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(",") if k.strip()]
            elif not isinstance(keywords, list):
                keywords = []

            # Get available styles from registered renderers
            logger.info(f"Getting available styles for dataset '{title}'")

            styles = []
            for renderer_cls in RenderRegistry.all().values():
                # Add default variant alias
                default_variant = renderer_cls.default_variant()
                default_style_info = renderer_cls.describe_style("default")
                default_style_info["title"] = (
                    f"{renderer_cls.style_id().title()} - Default ({default_variant.title()})"
                )
                default_style_info["description"] = (
                    f"Default {renderer_cls.style_id()} rendering (alias for {default_variant})"
                )
                styles.append(
                    Style(
                        id=default_style_info["id"],
                        title=default_style_info["title"],
                        description=default_style_info["description"],
                    )
                )

                # Add all actual variants
                for variant in renderer_cls.supported_variants():
                    style_info = renderer_cls.describe_style(variant)
                    styles.append(
                        Style(
                            id=style_info["id"],
                            title=style_info["title"],
                            description=style_info["description"],
                        )
                    )

            logger.info("loading extents for dataset vars")

            layer_extents = {}
            for var_name in dataset.data_vars.keys():
                extents = extract_dataset_extents(dataset, var_name)
                layer_extents[var_name] = extents

            # Create one tileset entry per supported tile matrix set
            supported_tms = get_all_tile_matrix_set_ids()

            for tms_id in supported_tms:
                if tms_id in TILE_MATRIX_SETS:
                    tms_summary = TILE_MATRIX_SET_SUMMARIES[tms_id]()

                    # Create layers for each data variable
                    layers = []
                    for var_name, var_data in dataset.data_vars.items():
                        extents = layer_extents[var_name]

                        # Extract variable-specific bounding box, fallback to dataset bounds
                        var_bounding_box = extract_variable_bounding_box(
                            dataset, var_name, tms_summary.crs
                        )

                        layer = Layer(
                            id=var_name,
                            title=var_data.attrs.get("long_name", var_name),
                            description=var_data.attrs.get("description", ""),
                            dataType=DataType.COVERAGE,
                            boundingBox=var_bounding_box,
                            crs=tms_summary.crs,
                            links=[
                                Link(
                                    href=f"./{tms_id}/{{tileMatrix}}/{{tileRow}}/{{tileCol}}?variables={var_name}",
                                    rel="item",
                                    type="image/png",
                                    title=f"Tiles for {var_name}",
                                    templated=True,
                                )
                            ],
                            extents=extents,
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
                        keywords=keywords if keywords else None,
                        attribution=dataset_attrs.get("attribution"),
                        license=dataset_attrs.get("license"),
                        version=dataset_attrs.get("version"),
                        pointOfContact=dataset_attrs.get("contact"),
                        mediaTypes=["image/png", "image/jpeg"],
                        styles=styles,
                    )
                    tilesets.append(tileset)

            return TilesetsList(tilesets=tilesets)

        @router.get(
            "/{tileMatrixSetId}",
            response_model=TileSetMetadata,
            response_model_exclude_none=True,
        )
        async def get_dataset_tileset_metadata(
            tileMatrixSetId: str,
            dataset: Dataset = Depends(deps.dataset),  # noqa: B008
        ):
            """Get tileset metadata for this dataset"""
            try:
                return create_tileset_metadata(dataset, tileMatrixSetId)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e

        @router.get(
            "/{tileMatrixSetId}/tilejson.json",
            response_model=TileJSON,
            response_model_exclude_none=True,
        )
        async def get_dataset_tilejson(
            request: Request,
            tileMatrixSetId: str,
            query: Annotated[TileQuery, Query()],
            dataset: Dataset = Depends(deps.dataset),  # noqa: B008
        ):
            """Get TileJSON specification for this dataset and tile matrix set"""
            # Validate that the tile matrix set exists
            if tileMatrixSetId not in TILE_MATRIX_SET_SUMMARIES:
                raise HTTPException(status_code=404, detail="Tile matrix set not found")

            # Extract dimension selectors from query parameters
            selectors = {}
            for param_name, param_value in request.query_params.items():
                # Skip the standard tile query parameters
                if param_name not in TILES_FILTERED_QUERY_PARAMS:
                    # Check if this parameter corresponds to a dataset dimension
                    if param_name in dataset.dims:
                        selectors[param_name] = param_value

            if not query.variables or len(query.variables) == 0:
                raise HTTPException(status_code=422, detail="No variables specified")

            bounds = extract_variable_bounding_box(
                dataset, query.variables[0], "EPSG:4326"
            )

            # Build tile URL template relative to this endpoint
            base_url = str(request.base_url).rstrip("/")
            # dataset path prefix already includes /datasets/{id} by xpublish; request.url.path points to /datasets/{id}/tiles/{tms}/tilejson.json
            # Construct sibling tiles path replacing trailing segment
            tiles_path = request.url.path.rsplit("/", 1)[0]  # drop 'tilejson.json'

            style = query.style[0] if query.style else "raster"
            variant = query.style[1] if query.style else "default"

            # XYZ template
            url_template = f"{base_url}{tiles_path}/{{z}}/{{y}}/{{x}}?variables={','.join(query.variables)}&style={style}/{variant}&width={query.width}&height={query.height}&f={query.f}"
            # Append selectors
            if selectors:
                selector_qs = "&".join(f"{k}={v}" for k, v in selectors.items())
                url_template = f"{url_template}&{selector_qs}"

            # Compute bounds list if available
            bounds_list = None
            if bounds is not None:
                bounds_list = [
                    bounds.lowerLeft[0],
                    bounds.lowerLeft[1],
                    bounds.upperRight[0],
                    bounds.upperRight[1],
                ]

            # Determine min/max zoom from TMS definition
            tms = TILE_MATRIX_SETS[tileMatrixSetId]()
            minzoom = 0
            maxzoom = (
                max(int(m.id) for m in tms.tileMatrices if str(m.id).isdigit())
                if tms.tileMatrices
                else None
            )

            # Compose TileJSON
            return TileJSON(
                tilejson="3.0.0",
                tiles=[url_template],
                name=dataset.attrs.get("title", "Dataset"),
                description=dataset.attrs.get("description"),
                version=dataset.attrs.get("version"),
                scheme="xyz",
                attribution=dataset.attrs.get("attribution"),
                bounds=bounds_list,
                minzoom=minzoom,
                maxzoom=maxzoom,
            )

        @router.get("/{tileMatrixSetId}/{tileMatrix}/{tileRow}/{tileCol}")
        @async_time_debug
        async def get_dataset_tile(
            request: Request,
            tileMatrixSetId: str,
            tileMatrix: int,
            tileRow: int,
            tileCol: int,
            query: Annotated[TileQuery, Query()],
            dataset: Dataset = Depends(deps.dataset),  # noqa: B008
        ):
            """Get individual tile from this dataset"""
            try:
                bbox, crs = extract_tile_bbox_and_crs(
                    tileMatrixSetId, tileMatrix, tileRow, tileCol
                )
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e

            # Extract dimension selectors from query parameters
            selectors = {}
            for param_name, param_value in request.query_params.items():
                # Skip the standard tile query parameters
                if param_name not in TILES_FILTERED_QUERY_PARAMS:
                    # Check if this parameter corresponds to a dataset dimension
                    if param_name in dataset.dims:
                        selectors[param_name] = param_value

            style = query.style[0] if query.style else "raster"
            variant = query.style[1] if query.style else "default"

            render_params = QueryParams(
                variables=query.variables,
                style=style,
                colorscalerange=query.colorscalerange,
                variant=variant,
                crs=crs,
                bbox=bbox,
                width=query.width,
                height=query.height,
                format=query.f,
                selectors=selectors,
            )
            try:
                buffer = await pipeline(dataset, render_params)
                status_code = 200  # only used as a sentinel value
                detail = "OK"
            except TileTooBigError:
                status_code = 413
                detail = f"Tile {tileMatrixSetId}/{tileMatrix}/{tileRow}/{tileCol} request too big. Please choose a higher zoom level."
            except KeyError as ke:
                logger.error(f"KeyError: {ke}")
                status_code = 422
                detail = f"Invalid variable name(s): {query.variables!r}."

            if status_code != 200:
                if not query.render_errors:
                    raise HTTPException(status_code=status_code, detail=detail)
                else:
                    # Use renderer's render_error method for all error types
                    renderer = render_params.get_renderer()
                    buffer = io.BytesIO()
                    renderer.render_error(
                        buffer=buffer,
                        width=query.width,
                        height=query.height,
                        message=detail,
                        format=query.f,
                    )

            return StreamingResponse(
                buffer,
                media_type="image/png",
            )

        return router
