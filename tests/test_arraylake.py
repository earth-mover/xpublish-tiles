from functools import partial

import numpy as np
import pandas as pd
import pytest

from icechunk.xarray import to_icechunk
from tests.datasets import Dataset, Dim, raster_grid, uniform_grid

ARRAYLAKE_REPO = "earthmover-integration/tiles-datasets-develop"


IFS = Dataset(
    # https://app.earthmover.io/earthmover-demos/ecmwf-ifs-oper/array/main/tprate
    group="ifs",
    dims=(
        Dim(
            name="time",
            size=2,
            chunk_size=1,
            data=np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[h]"),
        ),
        Dim(
            name="step",
            size=49,
            chunk_size=5,
            data=pd.to_timedelta(np.arange(0, 49), unit="h"),
        ),
        Dim(name="latitude", size=721, chunk_size=240, data=np.linspace(90, -90, 721)),
        Dim(
            name="longitude", size=1440, chunk_size=360, data=np.linspace(-180, 180, 1440)
        ),
    ),
    dtype=np.float32,
    setup=uniform_grid,
)

SENTINEL2_NOCOORDS = Dataset(
    # https://app.earthmover.io/earthmover-demos/sentinel-datacube-South-America-3-icechunk
    group="s2-no-coords",
    dims=(
        Dim(
            name="time",
            size=1,
            chunk_size=1,
            data=np.array(["2000-01-01"], dtype="datetime64[h]"),
        ),
        Dim(name="latitude", size=20_000, chunk_size=1800, data=None),
        Dim(name="longitude", size=20_000, chunk_size=1800, data=None),
        Dim(name="band", size=3, chunk_size=3, data=np.array(["R", "G", "B"])),
    ),
    dtype=np.uint16,
    setup=partial(
        raster_grid,
        crs="wgs84",
        geotransform="-82.0 0.0002777777777777778 0.0 13.0 0.0 -0.0002777777777777778",
    ),
)

HELIOS = Dataset(
    # https://app.earthmover.io/zeus-ai/helios/array/main/latlon/v2.0/analysis/2025/ghi
    group="helios",
    dims=(
        Dim(
            name="time",
            size=2,
            chunk_size=1,
            data=np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[h]"),
        ),
        Dim(
            name="latitude",
            size=3000,
            chunk_size=500,
            data=np.linspace(-89.97, -89.9700001, 3000),
        ),
        Dim(
            name="longitude",
            size=6000,
            chunk_size=500,
            data=np.linspace(-179.97, 180.0001, 6000),
        ),
        Dim(name="band", size=3, chunk_size=3, data=np.array(["R", "G", "B"])),
    ),
    dtype=np.float32,
    setup=uniform_grid,
)

PARA = Dataset(
    # https://app.earthmover.io/ctrees/lulc_30m_global/array/main/BRA.14_1/classification
    group="para",
    dims=(
        Dim(
            name="x",
            size=52065,
            chunk_size=2000,
            data=np.linspace(-58.988125, -45.972125, 52065),
        ),
        Dim(
            name="y",
            size=50612,
            chunk_size=2000,
            data=np.linspace(2.721625, -9.931125, 50612),
        ),
        Dim(
            name="time",
            size=1,
            chunk_size=1,
            data=np.array(["2018-01-01"], dtype="datetime64[h]"),
        ),
    ),
    dtype=np.int16,
    attrs={
        "flag_meanings": (
            "water ocean forest grassland agriculture urban barren shrubland "
            "wetland cropland tundra ice"
        ),
        "flag_values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    },
    setup=partial(
        raster_grid,
        crs="wgs84",
        geotransform="-58.98825 0.00024999999999999995 0.0 2.72175 0.0 -0.00025000000000000017",
    ),
)


@pytest.fixture(
    params=[
        pytest.param(IFS, id="ifs"),
        pytest.param(SENTINEL2_NOCOORDS, id="sentinel2-nocoords"),
        pytest.param(HELIOS, id="helios"),
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

    to_icechunk(ds, session, group=dataset.group, mode="w")
    session.commit(f"wrote {dataset.group!r}")
