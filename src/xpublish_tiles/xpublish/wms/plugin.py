"""OGC Web Map Service XPublish Plugin"""

import io
import math
from enum import Enum
from typing import Annotated

import cf_xarray  # noqa: F401
import morecantile
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import Response
from PIL import Image
from pydantic_xml import BaseXmlModel
from xpublish import Dependencies, Plugin, hookimpl

import xarray as xr
from xarray import DataTree
from xpublish_tiles.lib import (
    AsyncLoadTimeoutError,
    ColormapError,
    IndexingError,
    MissingParameterError,
    TileTooBigError,
    VariableNotFoundError,
)
from xpublish_tiles.logger import get_context_logger, with_accumulated_logs
from xpublish_tiles.multiscale import get_dataset, get_resolution_level
from xpublish_tiles.pipeline import _infer_datatype, pipeline
from xpublish_tiles.projections import transformer_from_crs
from xpublish_tiles.render import RenderRegistry
from xpublish_tiles.types import ImageFormat, OutputBBox, OutputCRS, QueryParams
from xpublish_tiles.xpublish.wms.types import (
    WMS_FILTERED_QUERY_PARAMS,
    WMSGetCapabilitiesQuery,
    WMSGetFeatureInfoQuery,
    WMSGetLegendGraphicQuery,
    WMSGetMapQuery,
    WMSQuery,
    WMSServiceExceptionReportResponse,
    WMSServiceExceptionResponse,
)
from xpublish_tiles.xpublish.wms.utils import create_capabilities_response

WEB_MERCATOR_TMS = morecantile.tms.get("WebMercatorQuad")

WMS_VERSION = "1.3.0"
WMS_NAMESPACE = "http://www.opengis.net/wms"
OGC_NAMESPACE = "http://www.opengis.net/ogc"


def to_default_namespace_xml(model: BaseXmlModel, namespace: str) -> bytes:
    """Serialize with the WMS namespace as the default namespace.

    pydantic-xml emits ns0: prefixes, which QGIS does not accept.
    """
    xml_content = model.to_xml(xml_declaration=True, encoding="UTF-8", skip_empty=True)
    xml_str = (
        xml_content.decode("utf-8") if isinstance(xml_content, bytes) else xml_content
    )
    xml_str = xml_str.replace("ns0:", "")
    xml_str = xml_str.replace(f'xmlns:ns0="{namespace}"', f'xmlns="{namespace}"')
    if "xmlns:xlink" not in xml_str and "xlink:" in xml_str:
        xml_str = xml_str.replace(
            f'xmlns="{namespace}"',
            f'xmlns="{namespace}" xmlns:xlink="http://www.w3.org/1999/xlink"',
        )
    return xml_str.encode("utf-8")


def wms_exception(message: str, *, status_code: int, code: str | None = None) -> Response:
    """Build a WMS ServiceExceptionReport response."""
    report = WMSServiceExceptionReportResponse(
        exceptions=[WMSServiceExceptionResponse(code=code, text=message)]
    )
    return Response(
        content=to_default_namespace_xml(report, OGC_NAMESPACE),
        media_type="text/xml",
        status_code=status_code,
    )


class WMSPlugin(Plugin):
    name: str = "wms"

    dataset_router_prefix: str = "/wms"
    dataset_router_tags: list[str | Enum] = ["wms"]

    @hookimpl
    def dataset_router(self, deps: Dependencies):
        """Add wms routes to the dataset router"""
        router = APIRouter(
            prefix=self.dataset_router_prefix, tags=self.dataset_router_tags
        )

        @router.get("", include_in_schema=False)
        @router.get("/")
        @with_accumulated_logs(
            log_message_fn=lambda request,
            wms_query,
            datatree: f"wms {wms_query.root.request} {getattr(datatree, '_xpublish_id', 'unknown')}",
            context_fn=lambda request, wms_query, datatree: {
                "endpoint": "wms",
                "request": wms_query.root.request,
                "dataset_id": getattr(datatree, "_xpublish_id", "unknown"),
            },
        )
        async def get_wms(
            request: Request,
            wms_query: Annotated[WMSQuery, Query()],
            datatree: DataTree = Depends(deps.datatree),
        ):
            match wms_query.root:
                case WMSGetCapabilitiesQuery():
                    return await handle_get_capabilities(
                        request, wms_query.root, get_dataset(datatree)
                    )
                case WMSGetMapQuery():
                    return await handle_get_map(request, wms_query.root, datatree)
                case WMSGetFeatureInfoQuery():
                    return wms_exception(
                        "GetFeatureInfo is not yet implemented. Coming Soon!",
                        status_code=501,
                        code="OperationNotSupported",
                    )
                case WMSGetLegendGraphicQuery():
                    return await handle_get_legend_graphic(
                        wms_query.root, get_dataset(datatree)
                    )

        return router


async def handle_get_capabilities(
    request: Request, query: WMSGetCapabilitiesQuery, dataset: xr.Dataset
) -> Response:
    """Handle WMS GetCapabilities requests with content negotiation."""

    # Determine response format from Accept header or format parameter
    accept_header = request.headers.get("accept", "")
    format_param = request.query_params.get("format", "").lower()

    # Default to XML for WMS compliance
    response_format = "xml"

    if format_param:
        if format_param in ["json", "application/json"]:
            response_format = "json"
        elif format_param in ["xml", "text/xml", "application/xml"]:
            response_format = "xml"
    elif "application/json" in accept_header:
        response_format = "json"

    # Get base URL from request
    base_url = str(request.url).split("?")[0]

    # Only 1.3.0 documents are produced; per WMS version negotiation the
    # server answers a 1.1.1 request with the closest version it supports.
    capabilities = create_capabilities_response(
        dataset=dataset,
        base_url=base_url,
        version=WMS_VERSION,
        service_title="XPublish WMS Service",
        service_abstract="Web Map Service powered by XPublish and xarray",
    )

    if response_format == "json":
        return Response(
            content=capabilities.model_dump_json(indent=2, exclude_none=True),
            media_type="application/json",
        )
    else:
        return Response(
            content=to_default_namespace_xml(capabilities, WMS_NAMESPACE),
            media_type="text/xml",
            headers={"Content-Type": "text/xml; charset=utf-8"},
        )


def estimate_zoom(query: WMSGetMapQuery) -> int | None:
    """Approximate the WebMercatorQuad zoom matching the requested resolution,
    used to pick a multiscale overview level for GetMap."""
    transformer = transformer_from_crs(crs_from=query.crs, crs_to="EPSG:3857")
    minx, _, maxx, _ = transformer.transform_bounds(
        query.bbox.west, query.bbox.south, query.bbox.east, query.bbox.north
    )
    if not (math.isfinite(minx) and math.isfinite(maxx)) or maxx <= minx:
        return None
    return WEB_MERCATOR_TMS.zoom_for_res((maxx - minx) / query.width)


def finalize_image(buffer: io.BytesIO, query: WMSGetMapQuery) -> io.BytesIO:
    """Apply TRANSPARENT/BGCOLOR and FORMAT to a rendered PNG.

    The pipeline always renders RGBA PNG; JPEG output and opaque maps are
    flattened onto ``bgcolor`` here.
    """
    if query.format == ImageFormat.PNG and query.transparent:
        return buffer
    buffer.seek(0)
    image = Image.open(buffer).convert("RGBA")
    background = Image.new("RGBA", image.size, query.bgcolor)
    flattened = Image.alpha_composite(background, image).convert("RGB")
    out = io.BytesIO()
    flattened.save(out, format=str(query.format))
    return out


async def handle_get_map(
    request: Request, query: WMSGetMapQuery, datatree: DataTree
) -> Response:
    """Handle WMS GetMap request."""

    if query.version != WMS_VERSION:
        return wms_exception(
            f"WMS version {query.version} is not supported for GetMap; use version={WMS_VERSION}.",
            status_code=400,
        )

    level = get_resolution_level(
        datatree, zoom=estimate_zoom(query), tms=WEB_MERCATOR_TMS
    )
    if level is not None:
        dataset = level.dataset
        resolution_level = level.path if level.path is not None else "root"
    else:
        dataset = datatree.to_dataset()
        resolution_level = None

    # Extract dimension selectors from query parameters
    selectors = {}
    for param_name, param_value in request.query_params.items():
        # Skip the standard tile query parameters
        if param_name not in WMS_FILTERED_QUERY_PARAMS:
            # Check if this parameter corresponds to a dataset dimension
            if param_name in dataset.dims:
                selectors[param_name] = param_value

    # Special handling for time and vertical axes per wms spec
    if query.time or query.elevation:
        cf_axes = dataset.cf.axes
        if query.time:
            time_name = cf_axes.get("T", None)
            if len(time_name):
                selectors[time_name[0]] = query.time
        if query.elevation:
            vertical_name = cf_axes.get("Z", None)
            if vertical_name:
                selectors[vertical_name[0]] = query.elevation

    style = query.styles[0] if query.styles else "raster"
    variant = query.styles[1] if query.styles else "default"

    render_params = QueryParams(
        variables=[query.layers],  # TODO: Support multiple layers
        style=style,
        colorscalerange=query.colorscalerange,
        variant=variant,
        crs=OutputCRS(query.crs),
        bbox=OutputBBox(query.bbox),
        width=query.width,
        height=query.height,
        # JPEG and opaque output are flattened in finalize_image
        format=ImageFormat.PNG,
        selectors=selectors,
        colormap=query.colormap,
        abovemaxcolor=query.abovemaxcolor,
        belowmincolor=query.belowmincolor,
    )

    status_code = 200
    detail = "OK"
    code = None
    buffer = io.BytesIO()
    try:
        buffer = await pipeline(dataset, render_params)
    except TileTooBigError:
        status_code = 413
        detail = "GetMap request too big. Please request a smaller area or fewer pixels."
        get_context_logger().error("TileTooBigError", message=detail)
    except VariableNotFoundError as e:
        status_code = 422
        detail = f"Invalid layer name(s): {query.layers!r}."
        code = "LayerNotDefined"
        get_context_logger().error("VariableNotFoundError", exc_info=e)
    except IndexingError as e:
        status_code = 422
        detail = f"Invalid dimension value: {selectors!r}. {e!s}"
        code = "InvalidDimensionValue"
        get_context_logger().error("IndexingError", exc_info=e)
    except MissingParameterError as e:
        status_code = 422
        detail = f"Missing parameter: {e!s}."
        get_context_logger().error("MissingParameterError", exc_info=e)
    except ColormapError as e:
        status_code = 422
        detail = f"Invalid colormap: {e!s}"
        get_context_logger().error("ColormapError", exc_info=e)
    except AsyncLoadTimeoutError as e:
        status_code = 504
        detail = "Data loading timed out."
        get_context_logger().error("AsyncLoadTimeoutError", exc_info=e)
    except Exception as e:
        status_code = 500
        detail = "Internal server error."
        get_context_logger().error("Exception", exc_info=e)

    if status_code != 200:
        match query.exceptions:
            case "INIMAGE":
                renderer = render_params.get_renderer()
                buffer = io.BytesIO()
                renderer.render_error(
                    buffer=buffer,
                    width=query.width,
                    height=query.height,
                    message=detail,
                    format=ImageFormat.PNG,
                )
            case "BLANK":
                buffer = io.BytesIO()
                Image.new("RGBA", (query.width, query.height)).save(buffer, format="png")
            case _:
                return wms_exception(detail, status_code=status_code, code=code)

    buffer = finalize_image(buffer, query)

    headers = {}
    if resolution_level is not None:
        headers["X-Multiscale-Level"] = resolution_level
    media_type = "image/png" if query.format == ImageFormat.PNG else "image/jpeg"
    return Response(buffer.getbuffer(), media_type=media_type, headers=headers)


async def handle_get_legend_graphic(
    query: WMSGetLegendGraphicQuery, dataset: xr.Dataset
) -> Response:
    """Handle WMS GetLegendGraphic request."""

    if query.layer not in dataset.data_vars:
        return wms_exception(
            f"Layer {query.layer!r} not found in dataset.",
            status_code=422,
            code="LayerNotDefined",
        )

    datatype = _infer_datatype(dataset[query.layer])
    style, variant = query.styles
    renderer = RenderRegistry.get(style)()

    attrs = dataset[query.layer].attrs
    base = attrs.get("long_name") or query.layer
    units = attrs.get("units")
    label = f"{base} [{units}]" if units else base

    buffer = io.BytesIO()
    try:
        renderer.render_legend(
            buffer=buffer,
            width=query.width,
            height=query.height,
            variant=variant,
            datatype=datatype,
            colorscalerange=query.colorscalerange,
            colormap=query.colormap,
            abovemaxcolor=query.abovemaxcolor,
            belowmincolor=query.belowmincolor,
            vertical=query.vertical,
            label=label if query.show_label else None,
            format=query.format,
        )
    except (MissingParameterError, ColormapError) as e:
        return wms_exception(str(e), status_code=422)

    media_type = "image/png" if query.format == ImageFormat.PNG else "image/jpeg"
    return Response(content=buffer.getvalue(), media_type=media_type)
