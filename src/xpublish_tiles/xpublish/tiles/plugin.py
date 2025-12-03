"""OGC Tiles API XPublish Plugin"""

import asyncio
import io
import json
from enum import Enum
from typing import Annotated
from urllib.parse import quote

import morecantile
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from xpublish import Dependencies, Plugin, hookimpl

from xarray import Dataset
from xpublish_tiles.grids import guess_grid_system
from xpublish_tiles.lib import (
    IndexingError,
    MissingParameterError,
    TileTooBigError,
    VariableNotFoundError,
    async_run,
)
from xpublish_tiles.logger import (
    get_context_logger,
    logger,
    set_context_logger,
    with_accumulated_logs,
)
from xpublish_tiles.pipeline import pipeline
from xpublish_tiles.render import RenderRegistry
from xpublish_tiles.tiles_lib import get_min_zoom
from xpublish_tiles.types import QueryParams
from xpublish_tiles.utils import normalize_tilejson_bounds
from xpublish_tiles.xpublish.tiles.metadata import (
    create_tileset_for_tms,
    create_tileset_metadata,
    extract_dataset_extents,
    extract_variable_bounding_box,
)
from xpublish_tiles.xpublish.tiles.tile_matrix import (
    TILE_MATRIX_SET_SUMMARIES,
    TILE_MATRIX_SETS,
    extract_tile_bbox_and_crs,
    get_all_tile_matrix_set_ids,
)
from xpublish_tiles.xpublish.tiles.types import (
    TILES_FILTERED_QUERY_PARAMS,
    ConformanceDeclaration,
    Style,
    TileJSON,
    TileMatrixSet,
    TileMatrixSets,
    TileQuery,
    TileSetMetadata,
    TilesetsList,
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
        @with_accumulated_logs(
            log_message_fn=lambda dataset: f"tiles_list {getattr(dataset, '_xpublish_id', 'unknown')}",
            context_fn=lambda dataset: {
                "endpoint": "tiles_list",
                "dataset_id": getattr(dataset, "_xpublish_id", "unknown"),
            },
        )
        async def get_dataset_tiles_list(
            dataset: Dataset = Depends(deps.dataset),
        ):
            """List of available tilesets for this dataset"""
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
                # Skip scalar variables
                if dataset[var_name].ndim == 0:
                    continue
                extents = await extract_dataset_extents(dataset, var_name)
                layer_extents[var_name] = extents

            # Create one tileset entry per supported tile matrix set
            supported_tms = get_all_tile_matrix_set_ids()

            # Execute concurrently in the event loop
            tileset_results = await asyncio.gather(
                *[
                    create_tileset_for_tms(
                        dataset,
                        tms_id,
                        layer_extents,
                        title,
                        description,
                        keywords,
                        dataset_attrs,
                        styles,
                    )
                    for tms_id in supported_tms
                ]
            )
            tilesets = [ts for ts in tileset_results if ts is not None]

            return TilesetsList(tilesets=tilesets)

        @router.get(
            "/{tileMatrixSetId}",
            response_model=TileSetMetadata,
            response_model_exclude_none=True,
        )
        @with_accumulated_logs(
            log_message_fn=lambda tileMatrixSetId,
            dataset: f"tileset_metadata {tileMatrixSetId} {getattr(dataset, '_xpublish_id', 'unknown')}",
            context_fn=lambda tileMatrixSetId, dataset: {
                "tileMatrixSetId": tileMatrixSetId,
                "dataset_id": getattr(dataset, "_xpublish_id", "unknown"),
            },
        )
        async def get_dataset_tileset_metadata(
            tileMatrixSetId: str,
            dataset: Dataset = Depends(deps.dataset),
        ):
            """Get tileset metadata for this dataset"""
            try:
                return await async_run(create_tileset_metadata, dataset, tileMatrixSetId)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e

        @router.get(
            "/{tileMatrixSetId}/tilejson.json",
            response_model=TileJSON,
            response_model_exclude_none=True,
        )
        @with_accumulated_logs(
            log_message_fn=lambda request,
            tileMatrixSetId,
            query,
            dataset: f"tilejson {tileMatrixSetId} {query.variables} {getattr(dataset, '_xpublish_id', 'unknown')}",
            context_fn=lambda request, tileMatrixSetId, query, dataset: {
                "tileMatrixSetId": tileMatrixSetId,
                "variables": query.variables,
                "dataset_id": getattr(dataset, "_xpublish_id", "unknown"),
            },
        )
        async def get_dataset_tilejson(
            request: Request,
            tileMatrixSetId: str,
            query: Annotated[TileQuery, Query()],
            dataset: Dataset = Depends(deps.dataset),
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

            try:
                bounds = await extract_variable_bounding_box(
                    dataset, query.variables[0], "EPSG:4326"
                )
            except VariableNotFoundError as e:
                logger.error("VariableNotFoundError", str(e))
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid variable name(s): {query.variables!r}.",
                ) from None

            logger.info(f"base_url: {request.base_url}")
            logger.info(f"url: {request.url.path}")
            logger.info(f"root_path: {request.scope.get('root_path')}")

            # Build tile URL template relative to this endpoint
            base_url = str(request.base_url).rstrip("/")
            root_path = request.scope.get("root_path", "")

            # dataset path prefix already includes /datasets/{id} by xpublish; request.url.path points to /datasets/{id}/tiles/{tms}/tilejson.json
            # Construct sibling tiles path replacing tailing segment
            tiles_path = request.url.path.replace(root_path, "", 1).rsplit("/", 1)[
                0
            ]  # drop 'tilejson.json'

            style = query.style[0] if query.style else "raster"
            variant = query.style[1] if query.style else "default"

            # XYZ template
            url_template = f"{base_url}{tiles_path}/{{z}}/{{y}}/{{x}}?variables={','.join(query.variables)}&style={style}/{variant}&width={query.width}&height={query.height}&f={query.f}&render_errors={str(query.render_errors).lower()}"
            # Append optional color scale range
            if query.colorscalerange:
                url_template = f"{url_template}&colorscalerange={query.colorscalerange[0]:g},{query.colorscalerange[1]:g}"

            if query.colormap:
                url_template = (
                    f"{url_template}&colormap={quote(json.dumps(query.colormap))}"
                )

            # Append selectors
            if selectors:
                selector_qs = "&".join(f"{k}={v}" for k, v in selectors.items())
                url_template = f"{url_template}&{selector_qs}"

            # Compute bounds list if available
            bounds_list = None
            if bounds is not None:
                bounds_list = normalize_tilejson_bounds(
                    [
                        bounds.lowerLeft[0],
                        bounds.lowerLeft[1],
                        bounds.upperRight[0],
                        bounds.upperRight[1],
                    ]
                )

            # Determine min/max zoom from dataset characteristics
            # Get the original morecantile TMS for minzoom/maxzoom properties
            tms = morecantile.tms.get(tileMatrixSetId)

            # Calculate optimal zoom levels based on grid and data characteristics
            # Get the first variable's grid system
            var_name = query.variables[0]
            grid = await async_run(guess_grid_system, dataset, var_name)
            da = dataset.cf[var_name]

            bound_logger = get_context_logger()
            bound_logger = bound_logger.bind(tms=tms.id)
            set_context_logger(bound_logger)

            # Calculate min/max zoom based on data characteristics
            minzoom = await async_run(get_min_zoom, grid, tms, da)
            maxzoom = tms.maxzoom

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
        @with_accumulated_logs(
            log_message_fn=lambda request,
            tileMatrixSetId,
            tileMatrix,
            tileRow,
            tileCol,
            query,
            dataset: f"{tileMatrixSetId}/{tileMatrix}/{tileRow}/{tileCol} {query.variables} {getattr(dataset, '_xpublish_id', 'unknown')}",
            context_fn=lambda request,
            tileMatrixSetId,
            tileMatrix,
            tileRow,
            tileCol,
            query,
            dataset: {
                "tile": f"{tileMatrixSetId}/{tileMatrix}/{tileRow}/{tileCol}",
                "variables": query.variables,
                "dataset_id": getattr(dataset, "_xpublish_id", "unknown"),
            },
        )
        async def get_dataset_tile(
            request: Request,
            tileMatrixSetId: str,
            tileMatrix: int,
            tileRow: int,
            tileCol: int,
            query: Annotated[TileQuery, Query()],
            dataset: Dataset = Depends(deps.dataset),
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
                colormap=query.colormap,
            )

            try:
                buffer = await pipeline(dataset, render_params)
                status_code = 200
                detail = "OK"
            except TileTooBigError:
                status_code = 413
                detail = f"Tile {tileMatrixSetId}/{tileMatrix}/{tileRow}/{tileCol} request too big. Please choose a higher zoom level."
                bound_logger = get_context_logger()
                bound_logger.error("TileTooBigError", message=detail)
            except VariableNotFoundError as e:
                bound_logger = get_context_logger()
                bound_logger.error("VariableNotFoundError", error=str(e))
                status_code = 422
                detail = f"Invalid variable name(s): {query.variables!r}."
            except IndexingError as e:
                bound_logger = get_context_logger()
                bound_logger.error("IndexingError", error=str(e))
                status_code = 422
                detail = f"Invalid indexer: {selectors!r}."
            except MissingParameterError as e:
                bound_logger = get_context_logger()
                bound_logger.error("MissingParameterError", error=str(e))
                status_code = 422
                detail = f"Missing parameter: {e!s}."
            except Exception as e:
                status_code = 500
                bound_logger = get_context_logger()
                bound_logger.error("Exception", error=str(e))
                detail = str(e)

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

            return Response(buffer.getbuffer(), media_type="image/png")

        return router
