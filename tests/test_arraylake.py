import pytest

import dask
from icechunk.xarray import to_icechunk
from xpublish_tiles.datasets import (
    EU3035,
    EU3035_HIRES,
    GLOBAL_6KM,
    HRRR,
    IFS,
    PARA,
    SENTINEL2_NOCOORDS,
    Dataset,
)


@pytest.fixture(
    params=[
        pytest.param(IFS, id="ifs"),
        pytest.param(SENTINEL2_NOCOORDS, id="sentinel2-nocoords"),
        pytest.param(GLOBAL_6KM, id="global_6km"),
        pytest.param(PARA, id="para"),
        pytest.param(HRRR, id="hrrr"),
        pytest.param(EU3035, id="eu3035"),
        pytest.param(EU3035_HIRES, id="eu3035_hires"),
    ]
)
def dataset(request):
    return request.param


def test_create(dataset: Dataset, repo, where: str, prefix: str, request) -> None:
    if not request.config.getoption("--setup"):
        pytest.skip("test_create only runs when --setup flag is provided")

    if "arraylake" in where:
        import coiled

        import distributed

        cluster = coiled.Cluster(name="tiles-testing", n_workers=(4, 100))
        client = distributed.Client(cluster)
        scheduler = client
    else:
        scheduler = "threads"

    ds = dataset.create()
    session = repo.writable_session("main")
    with dask.config.set(scheduler=scheduler):
        to_icechunk(ds, session, group=dataset.name, mode="w")
    session.commit(f"wrote {dataset.name!r}")
