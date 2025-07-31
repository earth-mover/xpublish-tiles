from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pyproj

import dask.array
import xarray as xr


@dataclass(kw_only=True)
class Dim:
    name: str
    chunk_size: int
    size: int
    data: np.ndarray | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


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
        coords={d.name: (d.name, d.data, d.attrs) for d in dims if d.data is not None},
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


def create_global_dataset(
    *,
    lat_ascending: bool = True,
    lon_0_360: bool = False,
    nlat: int = 720,
    nlon: int = 1441,
) -> xr.Dataset:
    """Create a global dataset with configurable coordinate ordering.

    Args:
        lat_ascending: If True, latitudes go from -90 to 90; if False, from 90 to -90
        lon_0_360: If True, longitudes go from 0 to 360; if False, from -180 to 180
        nlat: Number of latitude points
        nlon: Number of longitude points

    Returns:
        xr.Dataset: Global dataset with specified coordinate ordering
    """
    lats = np.linspace(-90, 90, nlat)
    if not lat_ascending:
        lats = lats[::-1]

    if lon_0_360:
        lons = np.linspace(0, 360, nlon)
    else:
        lons = np.linspace(-180, 180, nlon)

    dims = [
        Dim(
            name="latitude",
            size=nlat,
            chunk_size=nlat,
            data=lats,
            attrs={"standard_name": "latitude"},
        ),
        Dim(
            name="longitude",
            size=nlon,
            chunk_size=nlon,
            data=lons,
            attrs={"standard_name": "longitude"},
        ),
    ]
    return uniform_grid(dims=tuple(dims), dtype=np.float32, attrs={})
