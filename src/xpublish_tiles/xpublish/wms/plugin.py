"""OGC Web Map Service XPublish Plugin"""

from enum import Enum
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from xpublish import Dependencies, Plugin, hookimpl

import xarray as xr
from xpublish_tiles.utils import lower_case_keys
from xpublish_tiles.xpublish.wms.query import (
    WMS_FILTERED_QUERY_PARAMS,
    WMSQuery,
    WMSGetCapabilitiesQuery,
    WMSGetMapQuery,
    WMSGetFeatureInfoQuery,
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
                    # TODO: Implement GetCapabilities response
                    return {"message": "GetCapabilities"}
                case WMSGetMapQuery():
                    # TODO: Implement GetMap response
                    return {"message": "GetMap"}
                case WMSGetFeatureInfoQuery():
                    # TODO: Implement GetFeatureInfo response
                    return {"message": "GetFeatureInfo"}

        return router
