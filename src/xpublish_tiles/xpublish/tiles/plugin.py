"""OGC Tiles API XPublish Plugin"""

from enum import Enum

from fastapi import APIRouter
from xpublish import Dependencies, Plugin, hookimpl


class TilesPlugin(Plugin):
    name: str = "tiles"

    dataset_router_prefix: str = "/tiles"
    dataset_router_tags: list[str | Enum] = ["tiles"]

    @hookimpl
    def dataset_router(self, deps: Dependencies):
        """Add all tile routers to the dataset router"""
        router = APIRouter(
            prefix=self.dataset_router_prefix, tags=self.dataset_router_tags
        )

        @router.get("/")
        async def get_tiles():
            return {"message": "Hello, Tiles!"}

        return router
