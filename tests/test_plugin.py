import pytest
import xpublish

from xpublish_tiles.plugin import TilesPlugin


@pytest.fixture(scope="session")
def xpublish_app():
    rest = xpublish.Rest({}, plugins={"tiles": TilesPlugin()})
    return rest.app


def test_app_router(xpublish_app):
    assert xpublish_app.router.routes
