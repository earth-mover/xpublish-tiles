import numpy as np
import numpy.testing as npt
import pytest
from pyproj import CRS

import xarray as xr
from xpublish_tiles.config import config
from xpublish_tiles.extract import POINT_DIM, decompressed_bytes, extract_points
from xpublish_tiles.grids import guess_grid_system
from xpublish_tiles.lib import PointQueryNotSupportedError, RequestTooBigError, cf_get
from xpublish_tiles.projections import transformer_from_crs
from xpublish_tiles.testing.datasets import (
    CURVILINEAR,
    ERA5,
    EU3035,
    GLOBAL_6KM,
    GLOBAL_6KM_360,
    SENTINEL2_NOCOORDS,
)
from xpublish_tiles.types import ContinuousData, ValidatedArray

WGS84 = CRS.from_epsg(4326)


@pytest.mark.parametrize(
    "dataset",
    [GLOBAL_6KM, GLOBAL_6KM_360, ERA5, EU3035, SENTINEL2_NOCOORDS],
    ids=lambda d: d.name,
)
def test_locate_cell_centres(dataset):
    ds = dataset.create()
    name = next(iter(ds.data_vars))
    grid = guess_grid_system(ds, name)
    da = grid.assign_index(cf_get(ds, name))
    rng = np.random.default_rng(0)
    xi = rng.integers(0, da.sizes[grid.Xdim], 20)
    yi = rng.integers(0, da.sizes[grid.Ydim], 20)
    x, y = da[grid.X].values[xi], da[grid.Y].values[yi]

    shifts = (-360, 0, 360) if grid.crs.is_geographic else (0,)
    for shift in shifts:
        indexers = grid.locate(x + shift, y)
        npt.assert_array_equal(indexers[grid.Xdim], xi)
        npt.assert_array_equal(indexers[grid.Ydim], yi)

    far = 95.0 if grid.crs.is_geographic else 1e9
    outside = grid.locate(x[:1], np.array([far]))
    assert all(idx.tolist() == [-1] for idx in outside.values())


async def test_extract_points_matches_sel(air_dataset):
    x = np.array([-95.0, 0.0, 265.0, -160.0])
    y = np.array([45.0, 45.0, 45.0, 75.0])
    result = await extract_points(
        air_dataset,
        variables=["air"],
        selectors={"time": "2013-01-01T00:00:00"},
        crs=WGS84,
        x=x,
        y=y,
    )
    # (0, 45) is outside the North America grid
    assert result[POINT_DIM].values.tolist() == [0, 2, 3]
    expected = air_dataset.air.sel(time="2013-01-01T00:00:00").sel(
        lon=[265.0, 265.0, 200.0], lat=[45.0, 45.0, 75.0], method="nearest"
    )
    npt.assert_array_equal(result.air.values, np.diag(expected.values))


async def test_extract_points_reprojects_query(air_dataset):
    lon, lat = np.array([-95.0, -120.0]), np.array([45.0, 30.0])
    in_4326 = await extract_points(
        air_dataset, variables=["air"], selectors={}, crs=WGS84, x=lon, y=lat
    )
    mercator = CRS.from_epsg(3857)
    x, y = transformer_from_crs(WGS84, mercator).transform(lon, lat)
    in_3857 = await extract_points(
        air_dataset, variables=["air"], selectors={}, crs=mercator, x=x, y=y
    )
    npt.assert_array_equal(in_4326.air.values, in_3857.air.values)


async def test_extract_points_unsupported_grid():
    ds = CURVILINEAR.create()
    with pytest.raises(PointQueryNotSupportedError, match="Curvilinear"):
        await extract_points(
            ds,
            variables=[next(iter(ds.data_vars))],
            selectors={},
            crs=WGS84,
            x=np.array([0.0]),
            y=np.array([0.0]),
        )


async def test_extract_points_too_big(air_dataset):
    with config.set({"max_renderable_size": 1024}), pytest.raises(RequestTooBigError):
        await extract_points(
            air_dataset,
            variables=["air"],
            selectors={},
            crs=WGS84,
            x=np.array([-95.0]),
            y=np.array([45.0]),
            squeeze=False,
        )


def test_decompressed_bytes_counts_whole_chunks(air_dataset):
    da = xr.DataArray(
        np.zeros((10, 20, 30), dtype=np.float32), dims=("time", "lat", "lon")
    )
    # ``member`` was dropped by a scalar selector but is still chunked by 4
    array = ValidatedArray(
        da=da,
        grid=guess_grid_system(air_dataset, "air"),
        datatype=ContinuousData(valid_min=None, valid_max=None),
        chunks={"time": 5, "lat": 10, "lon": 10, "member": 4},
    )
    indexers = {"lon": np.array([0, 1, 25, -1]), "lat": np.array([0, 1, 15, -1])}
    # two distinct 10x10 (lat, lon) chunks; 10 times can span 3 time chunks of 5
    assert decompressed_bytes(array, indexers) == 2 * 100 * 15 * 4 * 4
