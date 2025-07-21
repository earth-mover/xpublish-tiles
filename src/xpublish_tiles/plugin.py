from fastapi import APIRouter
from xpublish import Dependencies, Plugin, hookimpl

from xpublish_tiles.routers.xyz import xyz_tiles_router


class TilesPlugin(Plugin):
    name = "tiles"

    app_router_prefix: str = "/tiles"
    app_router_tags: list[str] = ["tiles"]

    dataset_router_prefix: str = "/tiles"
    dataset_router_tags: list[str] = ["tiles"]

    @hookimpl
    def dataset_router(self, deps: Dependencies):
        """Add all tile routers to the dataset router"""
        router = APIRouter(
            prefix=self.dataset_router_prefix, tags=self.dataset_router_tags
        )
        router.include_router(xyz_tiles_router, prefix="/xyz")
        return router
