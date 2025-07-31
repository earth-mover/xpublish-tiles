"""OGC Web Map Service XPublish Plugin"""

from enum import Enum
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import Response
from xpublish import Dependencies, Plugin, hookimpl

import xarray as xr
from xpublish_tiles.utils import lower_case_keys
from xpublish_tiles.xpublish.wms.types import (
    WMS_FILTERED_QUERY_PARAMS,
    WMSGetCapabilitiesQuery,
    WMSGetFeatureInfoQuery,
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
            dataset: xr.Dataset = Depends(deps.dataset),  # noqa: B008
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
                    # TODO: Implement GetMap response
                    return {"message": "GetMap"}
                case WMSGetFeatureInfoQuery():
                    # TODO: Implement GetFeatureInfo response
                    return {"message": "GetFeatureInfo"}

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

                # Fix missing xlink namespace declaration
                xml_str = (
                    xml_content.decode("utf-8")
                    if isinstance(xml_content, bytes)
                    else xml_content
                )
                if "xmlns:xlink" not in xml_str and "xlink:" in xml_str:
                    xml_str = xml_str.replace(
                        'xmlns:ns0="http://www.opengis.net/wms"',
                        'xmlns:ns0="http://www.opengis.net/wms" xmlns:xlink="http://www.w3.org/1999/xlink"',
                    )
                    xml_content = xml_str.encode("utf-8")

                return Response(
                    content=xml_content,
                    media_type="text/xml",
                    headers={"Content-Type": "text/xml; charset=utf-8"},
                )

        return router
