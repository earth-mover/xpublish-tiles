import pytest

from icechunk.xarray import to_icechunk
from xpublish_tiles.testing.datasets import (
    EU3035,
    EU3035_HIRES,
    GLOBAL_6KM,
    HRRR,
    IFS,
    PARA,
    PARA_HIRES,
    REDGAUSS_N320,
    SENTINEL2_NOCOORDS,
    UTM33S,
    UTM33S_HIRES,
    UTM50S_HIRES,
    Dataset,
)


@pytest.fixture(
    params=[
        pytest.param(IFS, id="ifs"),
        pytest.param(SENTINEL2_NOCOORDS, id="sentinel2-nocoords"),
        pytest.param(GLOBAL_6KM, id="global_6km"),
        pytest.param(PARA, id="para"),
        pytest.param(PARA_HIRES, id="para_hires"),
        pytest.param(HRRR, id="hrrr"),
        pytest.param(EU3035, id="eu3035"),
        pytest.param(EU3035_HIRES, id="eu3035_hires"),
        pytest.param(UTM33S, id="utm33s"),
        pytest.param(UTM33S_HIRES, id="utm33s_hires"),
        pytest.param(UTM50S_HIRES, id="utm50s_hires"),
        pytest.param(REDGAUSS_N320, id="redgauss_n320"),
    ]
)
def dataset(request):
    return request.param


# This test runs first when --setup is passed. The xdist_group ensures it completes
# before other tests run in parallel.
@pytest.mark.xdist_group(name="repo_creation")
def test_create_local_dataset(dataset: Dataset, repo) -> None:
    ds = dataset.create()
    session = repo.writable_session("main")
    to_icechunk(ds, session, group=dataset.name, mode="w")
    session.commit(f"wrote {dataset.name!r}")
