from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
import pytest

import dask.array
import xarray as xr
from icechunk.xarray import to_icechunk

ARRAYLAKE_REPO = "earthmover-integration/tiles-datasets-develop"


@dataclass(kw_only=True)
class Dim:
    name: str
    chunk_size: int
    size: int
    data: np.ndarray | None = None


@dataclass(kw_only=True)
class Dataset:
    group: str
    dims: tuple[Dim, ...]
    dtype: np.typing.DTypeLike
    attrs: dict[str, Any] = field(default_factory=dict)
    setup: Callable

    def create(self):
        return self.setup(dims=self.dims, dtype=self.dtype, attrs=self.attrs)


def generate_sine_wave_data(dims: tuple[Dim, ...], dtype: npt.DTypeLike):
    """Generate sine wave data across all dimensions.

    Fits 3 waves along each dimension using coordinate values as inputs.
    For dimensions without coordinates, uses normalized indices.
    """
    chunks = tuple(d.chunk_size for d in dims)

    # Create coordinate arrays for each dimension
    coord_arrays = []
    for i, dim in enumerate(dims):
        # Use provided coordinates or indices
        if dim.data is not None:
            coord_array = np.asarray(dim.data)
        else:
            coord_array = np.arange(dim.size)

        # Handle different data types
        if not np.issubdtype(coord_array.dtype, np.number):
            # For non-numeric coordinates (datetime, string, etc.), use integer offset based on position
            normalized = np.arange(len(coord_array), dtype=np.float64)
            if len(coord_array) > 1:
                normalized = normalized / (len(coord_array) - 1)
        else:
            # Numeric coordinates
            coord_min, coord_max = coord_array.min(), coord_array.max()
            assert (
                coord_max > coord_min
            ), f"Coordinate range must be non-zero for dimension {dim.name}"
            normalized = (coord_array - coord_min) / (coord_max - coord_min)

        # Add dimension-specific offset to avoid identical patterns
        normalized += i * 0.5
        coord_arrays.append(normalized * 6 * np.pi)  # 3 waves = 6π

    # Create dask arrays for coordinates with proper chunking
    dask_coords = []
    for coord_array, chunk_size in zip(coord_arrays, chunks, strict=False):
        dask_coord = dask.array.from_array(coord_array, chunks=chunk_size)
        dask_coords.append(dask_coord)

    # Create meshgrid with dask arrays
    grids = dask.array.meshgrid(*dask_coords, indexing="ij")

    # Start with sin of the first coordinate, then multiply by sin of remaining coordinates
    sine_data = dask.array.sin(grids[0])
    for grid in grids[1:]:
        sine_data = sine_data * dask.array.sin(grid)

    return sine_data.astype(dtype)


def generate_flag_values_data(
    dims: tuple[Dim, ...], dtype: npt.DTypeLike, flag_values: list
):
    """Generate random integers from flag_values for categorical data."""
    shape = tuple(d.size for d in dims)
    chunks = tuple(d.chunk_size for d in dims)

    # Create random choice from flag_values
    flag_array = np.array(flag_values, dtype=dtype)
    random_indices = np.random.choice(len(flag_values), size=shape)
    data = flag_array[random_indices]

    return dask.array.from_array(data, chunks=chunks)


def uniform_grid(*, dims: tuple[Dim, ...], dtype: npt.DTypeLike, attrs: dict[str, Any]):
    # Check if this is categorical data with flag_values
    if "flag_values" in attrs:
        data_array = generate_flag_values_data(dims, dtype, attrs["flag_values"])
    else:
        # Generate sine wave data for continuous data
        data_array = generate_sine_wave_data(dims, dtype)

    ds = xr.Dataset(
        {
            "foo": (tuple(d.name for d in dims), data_array, attrs),
        },
        coords={d.name: d.data for d in dims if d.data is not None},
    )
    # coord vars always single chunk?
    for dim in dims:
        if dim.data is not None:
            ds.variables[dim.name].encoding = {"chunks": dim.size}

    return ds


def raster_grid(
    *,
    dims: tuple[Dim, ...],
    dtype: npt.DTypeLike,
    attrs: dict[str, Any],
    crs: Any,
    geotransform: str,
) -> xr.Dataset:
    ds = uniform_grid(dims=dims, dtype=dtype, attrs=attrs)
    crs = pyproj.CRS.from_user_input(crs)
    ds.coords["spatial_ref"] = ((), 0, crs.to_cf())
    ds.spatial_ref.attrs["GeoTransform"] = geotransform
    return ds


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
