from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
from pyproj.aoi import BBox

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
    name: str
    dims: tuple[Dim, ...]
    dtype: np.typing.DTypeLike
    attrs: dict[str, Any] = field(default_factory=dict)
    setup: Callable

    def create(self):
        ds = self.setup(dims=self.dims, dtype=self.dtype, attrs=self.attrs)
        ds.attrs["name"] = self.name
        return ds


def generate_tanh_wave_data(dims: tuple[Dim, ...], dtype: npt.DTypeLike):
    """Generate smooth tanh wave data across all dimensions.

    Fits 3 waves along each dimension using coordinate values as inputs.
    Uses tanh to create smooth, bounded patterns in [-1, 1] range.
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

    # Create smooth patterns using tanh of summed sine waves
    # tanh naturally bounds to [-1, 1] and creates smooth, flowing patterns
    sine_sum = dask.array.zeros_like(grids[0])
    for grid in grids:
        sine_sum = sine_sum + dask.array.sin(grid)

    # Use tanh to compress the sum into [-1, 1] range smoothly
    # The factor 0.8 prevents saturation, keeping gradients smooth
    sine_data = dask.array.tanh(0.8 * sine_sum)

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
        # Generate tanh wave data for continuous data
        data_array = generate_tanh_wave_data(dims, dtype)

    attrs["valid_max"] = 1
    attrs["valid_min"] = -1
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
    bbox: BBox | None = None,
) -> xr.Dataset:
    ds = uniform_grid(dims=dims, dtype=dtype, attrs=attrs)
    crs = pyproj.CRS.from_user_input(crs)
    ds.coords["spatial_ref"] = ((), 0, crs.to_cf())
    ds.spatial_ref.attrs["GeoTransform"] = geotransform

    # Add bounding box to dataset attributes if provided
    if bbox is not None:
        ds.attrs["bbox"] = bbox

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


HRRR_CRS_WKT = "".join(
    [
        'PROJCRS["unknown",BASEGEOGCRS["unknown",DATUM["unknown",ELLIPSOID["unk',
        'nown",6371229,0,LENGTHUNIT["metre",1,ID["EPSG",9001]]]],PRIMEM["Greenw',
        'ich",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8901]]],CONVER',
        'SION["unknown",METHOD["Lambert Conic Conformal',
        '(2SP)",ID["EPSG",9802]],PARAMETER["Latitude of false origin",38.5,ANGL',
        'EUNIT["degree",0.0174532925199433],ID["EPSG",8821]],PARAMETER["Longitu',
        'de of false origin",262.5,ANGLEUNIT["degree",0.0174532925199433],ID["E',
        'PSG",8822]],PARAMETER["Latitude of 1st standard parallel",38.5,ANGLEUN',
        'IT["degree",0.0174532925199433],ID["EPSG",8823]],PARAMETER["Latitude',
        'of 2nd standard parallel",38.5,ANGLEUNIT["degree",0.0174532925199433],',
        'ID["EPSG",8824]],PARAMETER["Easting at false',
        'origin",0,LENGTHUNIT["metre",1],ID["EPSG",8826]],PARAMETER["Northing',
        'at false origin",0,LENGTHUNIT["metre",1],ID["EPSG",8827]]],CS[Cartesia',
        'n,2],AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1,ID["EPSG",9001]]],A',
        'XIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]',
    ]
)


IFS = Dataset(
    # https://app.earthmover.io/earthmover-demos/ecmwf-ifs-oper/array/main/tprate
    name="ifs",
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
    name="s2-no-coords",
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

GLOBAL_6KM = Dataset(
    name="global_6km",
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
    name="para",
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

transformer = pyproj.Transformer.from_crs(HRRR_CRS_WKT, 4326, always_xy=True)
x0, y0 = transformer.transform(237.280472, 21.138123, direction="INVERSE")

HRRR = Dataset(
    name="hrrr",
    dims=(
        Dim(
            name="x",
            size=1799,
            chunk_size=2000,
            data=x0 + np.arange(1799) * 3000,
        ),
        Dim(
            name="y",
            size=1059,
            chunk_size=2000,
            data=y0 + np.arange(1059) * 3000,
        ),
        Dim(
            name="time",
            size=1,
            chunk_size=1,
            data=np.array(["2018-01-01"], dtype="datetime64[h]"),
        ),
        Dim(
            name="step",
            size=1,
            chunk_size=1,
            data=pd.to_timedelta(np.arange(0, 2), unit="h"),
        ),
    ),
    dtype=np.float32,
    setup=partial(
        raster_grid,
        crs=HRRR_CRS_WKT,
        geotransform=None,
        bbox=BBox(west=-134.095480, south=21.138123, east=-60.917193, north=52.6156533),
    ),
)

EU3035 = Dataset(
    name="eu3035",
    dims=(
        Dim(name="x", size=28741, chunk_size=2000, data=None),
        Dim(name="y", size=33584, chunk_size=2000, data=None),
    ),
    dtype=np.float32,
    setup=partial(
        raster_grid,
        crs="epsg:3035",
        geotransform="2635780.0 120.0 0.0 5416000.0 0.0 -120.0",
        bbox=BBox(
            west=-16.0, south=32.0, east=40.0, north=84.0
        ),  # Approximate EU coverage
    ),
)
