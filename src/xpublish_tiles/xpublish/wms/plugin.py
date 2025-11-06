"""OGC Web Map Service XPublish Plugin"""

from enum import Enum
from io import BytesIO
from typing import Annotated

import cf_xarray  # noqa: F401
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import Response
from PIL import Image
from xpublish import Dependencies, Plugin, hookimpl

import xarray as xr
from xpublish_tiles.pipeline import pipeline
from xpublish_tiles.types import OutputBBox, OutputCRS, QueryParams
from xpublish_tiles.utils import lower_case_keys
from xpublish_tiles.xpublish.wms.types import (
    WMS_FILTERED_QUERY_PARAMS,
    WMSGetCapabilitiesQuery,
    WMSGetFeatureInfoQuery,
    WMSGetLegendGraphicQuery,
    WMSGetMapQuery,
    WMSQuery,
)
from xpublish_tiles.xpublish.wms.utils import create_capabilities_response


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
        async def get_wms(
            request: Request,
            wms_query: Annotated[WMSQuery, Query()],
            dataset: xr.Dataset = Depends(deps.dataset),
        ):
            query_params = lower_case_keys(request.query_params)
            query_keys = list(query_params.keys())
            extra_query_params = {}
            for query_key in query_keys:
                if query_key not in WMS_FILTERED_QUERY_PARAMS:
                    extra_query_params[query_key] = query_params[query_key]
                    del query_params[query_key]

            match wms_query.root:
                case WMSGetCapabilitiesQuery():
                    return await handle_get_capabilities(request, wms_query.root, dataset)
                case WMSGetMapQuery():
                    return await handle_get_map(request, wms_query.root, dataset)
                case WMSGetFeatureInfoQuery():
                    raise NotImplementedError(
                        "GetFeatureInfo is not yet implemented. Coming Soon!"
                    )
                case WMSGetLegendGraphicQuery():
                    return await handle_get_legend_graphic(wms_query.root)

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

    # Create capabilities response
    capabilities = create_capabilities_response(
        dataset=dataset,
        base_url=base_url,
        version=query.version,
        service_title="XPublish WMS Service",
        service_abstract="Web Map Service powered by XPublish and xarray",
    )

    if response_format == "json":
        # Return JSON response
        return Response(
            content=capabilities.model_dump_json(indent=2, exclude_none=True),
            media_type="application/json",
        )
    else:
        # Return XML response
        xml_content = capabilities.to_xml(
            xml_declaration=True, encoding="UTF-8", skip_empty=True
        )

        # Fix namespace prefixes for QGIS compatibility
        xml_str = (
            xml_content.decode("utf-8") if isinstance(xml_content, bytes) else xml_content
        )

        # Replace ns0: prefixes with default namespace for QGIS compatibility
        xml_str = xml_str.replace("ns0:", "")
        xml_str = xml_str.replace(
            'xmlns:ns0="http://www.opengis.net/wms"', 'xmlns="http://www.opengis.net/wms"'
        )

        # Ensure xlink namespace is present
        if "xmlns:xlink" not in xml_str and "xlink:" in xml_str:
            xml_str = xml_str.replace(
                'xmlns="http://www.opengis.net/wms"',
                'xmlns="http://www.opengis.net/wms" xmlns:xlink="http://www.w3.org/1999/xlink"',
            )

        xml_content = xml_str.encode("utf-8")

        return Response(
            content=xml_content,
            media_type="text/xml",
            headers={"Content-Type": "text/xml; charset=utf-8"},
        )


async def handle_get_map(
    request: Request, query: WMSGetMapQuery, dataset: xr.Dataset
) -> Response:
    """Handle WMS GetMap request."""

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
        format=query.format,
        selectors=selectors,
        colormap=query.colormap,
    )
    buffer = await pipeline(dataset, render_params)

    return Response(buffer.getbuffer(), media_type="image/png")


async def handle_get_legend_graphic(query: WMSGetLegendGraphicQuery) -> Response:
    """Handle WMS GetLegendGraphic request with a dummy PNG response."""

    # Create a simple dummy PNG image
    img = Image.new("RGB", (query.width, query.height), color="white")

    # Save to BytesIO buffer
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
    )
