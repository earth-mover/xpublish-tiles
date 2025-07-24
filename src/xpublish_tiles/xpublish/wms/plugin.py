"""OGC Web Map Service XPublish Plugin"""

from enum import Enum

from fastapi import APIRouter
from xpublish import Dependencies, Plugin, hookimpl


class WMSPlugin(Plugin):
    name: str = "wms"

    dataset_router_prefix: str = "/wms"
    dataset_router_tags: list[str | Enum] = ["wms"]

    @hookimpl
    def dataset_router(self, deps: Dependencies):
        """Add all tile routers to the dataset router"""
        router = APIRouter(
            prefix=self.dataset_router_prefix, tags=self.dataset_router_tags
        )

        @router.get("/")
        async def get_wms():
            return {"message": "Hello, WMS!"}

        return router
