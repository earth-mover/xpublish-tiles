import pytest

from icechunk.xarray import to_icechunk
from xpublish_tiles.datasets import (
    GLOBAL_6KM,
    IFS,
    PARA,
    SENTINEL2_NOCOORDS,
    Dataset,
)

ARRAYLAKE_REPO = "earthmover-integration/tiles-datasets-develop"


@pytest.fixture(
    params=[
        pytest.param(IFS, id="ifs"),
        pytest.param(SENTINEL2_NOCOORDS, id="sentinel2-nocoords"),
        pytest.param(GLOBAL_6KM, id="global_6km"),
        pytest.param(PARA, id="para"),
    ]
)
def dataset(request):
    return request.param


def test_create(dataset: Dataset, repo, where: str, prefix: str, request) -> None:
    if not request.config.getoption("--setup"):
        pytest.skip("test_create only runs when --setup flag is provided")

    ds = dataset.create()
    session = repo.writable_session("main")

    to_icechunk(ds, session, group=dataset.name, mode="w")
    session.commit(f"wrote {dataset.name!r}")
