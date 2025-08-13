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
    PARA_HIRES,
    SENTINEL2_NOCOORDS,
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
    ]
)
def dataset(request):
    return request.param


@pytest.mark.xdist_group(name="repo_creation")
def test_create(dataset: Dataset, repo, where: str, prefix: str, setup_option) -> None:
    if setup_option is None:
        pytest.skip("test_create only runs when --setup flag is provided")

    force_create = setup_option == "force"

    # Check if dataset already exists (unless force_create)
    if not force_create and "arraylake" in where:
        try:
            # Check if the dataset group already exists in the current repo
            snapshot = repo.readonly_session()
            group_exists = dataset.name in snapshot.store
            if group_exists:
                print(
                    f"Dataset {dataset.name} already exists in {prefix}, skipping creation"
                )
                return
        except Exception:
            # If we can't check the group, proceed with creation
            pass

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
