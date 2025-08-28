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
from xpublish_tiles.testing.tiles import (
    ETRS89_TILES,
    ETRS89_TILES_EDGE_CASES,
    HRRR_TILES,
    HRRR_TILES_EDGE_CASES,
    PARA_TILES,
    PARA_TILES_EDGE_CASES,
    SOUTH_AMERICA_BENCHMARK_TILES,
    UTM33S_TILES,
    UTM33S_TILES_EDGE_CASES,
    WEBMERC_TILES,
    WEBMERC_TILES_EDGE_CASES,
    WGS84_TILES,
    WGS84_TILES_EDGE_CASES,
)


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
    edge_case_tiles: list = field(default_factory=list)
    tiles: list = field(default_factory=list)
    benchmark_tiles: list[str] = field(default_factory=list)

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
    """Generate discretized tanh wave data with noise using flag_values for categorical data."""
    # Generate tanh wave data (returns values in [-1, 1] range)
    tanh_data = generate_tanh_wave_data(dims, np.float32)

    # Add random noise that preserves the sign
    # Generate noise proportional to the absolute value to avoid sign changes
    shape = tuple(d.size for d in dims)
    chunks = tuple(d.chunk_size for d in dims)

    # Create random noise array with same chunking
    rng = dask.array.random.default_rng(seed=1234)
    noise = rng.uniform(-0.8, 0.8, size=shape, chunks=chunks)

    # Scale noise by absolute value to preserve sign and prevent crossing zero
    abs_tanh = np.abs(tanh_data)
    scaled_noise = noise * abs_tanh * 1.2  # Scale factor to control noise intensity

    # Apply noise while ensuring we stay within [-1, 1] bounds
    noisy_tanh = tanh_data + scaled_noise
    noisy_tanh = np.clip(noisy_tanh, -1, 1)

    # Discretize to N levels based on number of flag values
    num_categories = len(flag_values)
    # First normalize to [0, 1], then scale to [0, num_categories-1], then round to integers
    normalized = (noisy_tanh + 1) / 2  # Map [-1, 1] to [0, 1]
    scaled = normalized * (num_categories - 1)  # Map [0, 1] to [0, num_categories-1]
    indices = np.round(scaled).astype(int)  # Round and convert to int

    # Clip to ensure indices are in valid range
    indices = np.clip(indices, 0, num_categories - 1)

    # Map indices to actual flag values
    flag_array = np.array(flag_values, dtype=dtype)
    array = dask.array.map_blocks(
        lambda chunk, flags: flags[chunk],
        indices,
        flag_array,
        meta=indices,
    )

    # Use advanced indexing to map indices to flag values
    return array


def uniform_grid(*, dims: tuple[Dim, ...], dtype: npt.DTypeLike, attrs: dict[str, Any]):
    # Check if this is categorical data with flag_values
    if "flag_values" in attrs:
        data_array = generate_flag_values_data(dims, dtype, attrs["flag_values"])
    else:
        # Generate tanh wave data for continuous data
        data_array = generate_tanh_wave_data(dims, dtype)

    if "flag_values" not in attrs:
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
    if geotransform:
        ds.spatial_ref.attrs["GeoTransform"] = geotransform

    # Add bounding box to dataset attributes if provided
    if bbox is not None:
        ds.attrs["bbox"] = bbox

    return ds


def curvilinear_grid(
    *,
    dims: tuple[Dim, ...],
    dtype: npt.DTypeLike,
    attrs: dict[str, Any],
) -> xr.Dataset:
    """Create a curvilinear grid dataset with 2D lat/lon coordinates.

    Uses uniform_grid to create the base dataset, then adds curvilinear
    lat/lon coordinates using quadratic functions.
    """
    # Create base uniform grid
    ds = uniform_grid(dims=dims, dtype=dtype, attrs=attrs)

    # Extract x and y dimensions - should be the first two dims
    x_dim, y_dim = dims[0], dims[1]
    xsize, ysize = x_dim.size, y_dim.size

    # Create coordinate arrays
    x = np.arange(xsize)
    y = np.arange(ysize)

    # Generate curvilinear lat/lon using tanh functions for smooth, bounded coordinates
    # Normalize coordinates to [-1, 1] range
    x_norm = 2 * (x - xsize / 2) / xsize
    y_norm = 2 * (y - ysize / 2) / ysize

    X, Y = np.meshgrid(x_norm, y_norm, indexing="ij")

    # Base coordinates - create a regular grid first
    lon_base = (
        -100.0 + X * 15.0
    )  # Base longitude range: -115 to -85 (30 degrees, over US)
    lat_base = 40.0 + Y * 15.0  # Base latitude range: 25 to 55 (30 degrees, over US)

    # Add curvilinear distortion using tanh for smooth, bounded curves
    # tanh ensures distortions stay bounded and don't cause overlaps
    # Increased warping especially in latitude to make it less rectangular
    # Special handling for lower-right corner to wrap it further north
    lon_distortion = 3.0 * np.tanh(X + 0.8 * Y)  # Stronger longitude warping
    lat_distortion = 5.0 * np.tanh(Y + 0.6 * X + 0.4 * X * Y)  # Base latitude warping

    # Add extra northward curvature to the lower-right corner (positive X, negative Y)
    lower_right_mask = (X > 0) & (Y < 0)
    lat_distortion = np.where(
        lower_right_mask,
        lat_distortion
        + 8.0 * np.tanh(X * -Y * 2.0),  # Strong northward curve in lower-right
        lat_distortion,
    )

    lon = lon_base + lon_distortion
    lat = lat_base + lat_distortion

    # Add curvilinear coordinates to the dataset
    ds.coords["lat"] = ((x_dim.name, y_dim.name), lat, {"standard_name": "latitude"})
    ds.coords["lon"] = ((x_dim.name, y_dim.name), lon, {"standard_name": "longitude"})

    ds["foo"].attrs["coordinates"] = "lat lon"

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

# fmt: off
GLOBAL_BENCHMARK_TILES = [
    "3/2/1", "3/3/3", "3/2/2", "3/3/2", "3/3/1", "3/2/3", "3/1/2", "3/1/1", "3/4/2",
    "3/4/1", "3/1/3", "3/4/3", "3/2/0", "3/3/0", "3/2/4", "3/3/4", "3/1/0", "3/0/2",
    "3/4/0", "3/1/4", "3/5/2", "3/0/1", "3/4/4", "3/0/3", "3/5/3", "3/2/5", "3/0/0",
    "3/3/5", "3/0/4", "3/0/5", "3/1/5", "3/2/6", "3/2/7", "3/3/7", "3/0/7", "3/1/7",
    "3/0/6", "3/1/6", "3/3/6", "3/4/5", "2/0/3", "4/6/7", "4/5/6", "4/6/6", "4/6/5",
    "4/5/7", "4/7/6", "4/5/5", "4/7/7", "4/4/6", "4/6/8", "4/4/7", "4/5/8", "4/7/5",
    "4/4/5", "4/7/8", "4/4/8", "4/6/4", "4/5/4", "4/8/6", "4/5/9", "4/8/7", "4/6/9",
    "4/7/4", "4/8/5", "4/4/4", "4/7/9", "4/8/8", "4/4/9", "4/8/4", "4/6/3", "4/5/3",
    "4/9/7", "4/8/9", "4/7/3", "4/9/5", "4/6/10", "4/5/10", "4/4/3", "4/9/8", "4/7/10",
    "4/4/10", "4/8/3", "4/9/4", "4/6/2", "4/5/2", "4/7/2", "4/4/2", "4/6/11", "4/5/11",
    "4/4/11", "4/5/1", "4/4/1", "4/4/12", "5/12/13", "5/13/13", "5/12/12", "5/11/13",
    "5/11/12", "5/13/12", "5/12/14", "5/13/14", "5/12/11", "5/13/11", "5/11/11", "5/11/14",
    "5/14/13", "5/10/13", "5/14/12", "5/10/12", "5/14/14", "5/14/11", "5/12/15", "5/12/10",
    "5/10/11", "5/10/14", "5/13/15", "5/13/10", "5/14/15", "5/11/10", "5/11/15", "5/14/10",
    "5/10/15", "5/10/10", "5/12/9", "5/11/9", "5/11/16", "5/15/10", "5/15/15", "5/13/9",
    "5/11/8", "5/14/9", "5/12/8", "5/10/9", "5/13/8", "5/14/8", "5/15/13", "5/11/7",
    "5/15/8", "5/10/8", "5/12/7", "5/11/6", "5/13/7", "5/15/12", "5/14/7", "5/15/7",
    "5/12/6", "5/11/5", "5/15/11", "5/13/6", "5/15/6", "5/14/6", "5/12/5",
]
# fmt: on


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
        Dim(
            name="latitude",
            size=721,
            chunk_size=240,
            data=np.linspace(90, -90, 721),
        ),
        Dim(
            name="longitude",
            size=1440,
            chunk_size=360,
            data=np.linspace(-180 + 0.125, 180 - 0.125, 1440),
        ),
    ),
    dtype=np.float32,
    setup=uniform_grid,
    edge_case_tiles=WGS84_TILES_EDGE_CASES + WEBMERC_TILES_EDGE_CASES,
    tiles=WGS84_TILES + WEBMERC_TILES,
    benchmark_tiles=GLOBAL_BENCHMARK_TILES,
)

ERA5 = Dataset(
    # https://app.earthmover.io/earthmover-demos/ecmwf-ifs-oper/array/main/tprate
    name="era5",
    dims=(
        Dim(
            name="time",
            size=2,
            chunk_size=1,
            data=np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[h]"),
        ),
        Dim(
            name="latitude",
            size=721,
            chunk_size=240,
            data=np.linspace(90, -90, 721),
        ),
        Dim(
            name="longitude",
            size=1440,
            chunk_size=360,
            data=np.linspace(0, 359.75, 1440),
        ),
    ),
    dtype=np.float32,
    setup=uniform_grid,
    edge_case_tiles=WGS84_TILES_EDGE_CASES + WEBMERC_TILES_EDGE_CASES,
    tiles=WGS84_TILES + WEBMERC_TILES,
    benchmark_tiles=GLOBAL_BENCHMARK_TILES,
)


SENTINEL2_NOCOORDS = Dataset(
    # https://app.earthmover.io/earthmover-demos/sentinel-datacube-South-America-3-icechunk
    name="sentinel",
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
    edge_case_tiles=WGS84_TILES_EDGE_CASES + WEBMERC_TILES_EDGE_CASES,
    tiles=WGS84_TILES + WEBMERC_TILES,
    benchmark_tiles=SOUTH_AMERICA_BENCHMARK_TILES,
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
            data=np.linspace(-89.97, 89.97, 3000),
        ),
        Dim(
            name="longitude",
            size=6000,
            chunk_size=500,
            data=np.linspace(-179.97, 179.97, 6000),
        ),
        Dim(name="band", size=3, chunk_size=3, data=np.array(["R", "G", "B"])),
    ),
    dtype=np.float32,
    setup=uniform_grid,
    edge_case_tiles=WGS84_TILES_EDGE_CASES + WEBMERC_TILES_EDGE_CASES,
    tiles=WGS84_TILES + WEBMERC_TILES,
    benchmark_tiles=GLOBAL_BENCHMARK_TILES,
)

# fmt: off
UTM33S_BENCHMARK_TILES = [
    "3/4/4", "4/8/8", "4/9/8", "5/17/17", "5/18/17", "6/34/36", "6/35/34",
    "7/71/68", "7/72/68", "8/143/137", "8/144/137", "8/145/137", "9/286/272",
    "9/286/273", "9/286/274", "9/286/275", "9/287/272", "9/287/273", "9/287/274",
    "9/287/275", "9/288/272", "9/288/273", "9/288/274", "9/288/275", "9/289/272",
    "9/289/273", "9/289/274", "9/289/275", "9/290/272", "9/290/273", "9/290/274",
    "9/290/275", "9/285/272", "9/285/273", "9/285/274", "9/285/275", "10/574/545",
    "10/574/546", "10/574/547", "10/574/548", "10/574/549", "10/575/545", "10/575/546",
    "10/575/547", "10/575/548", "10/575/549", "10/576/545", "10/576/546", "10/576/547",
    "10/576/548", "10/576/549", "10/577/545", "10/577/546", "10/577/547", "10/577/548",
    "10/577/549", "10/578/545", "10/578/546", "10/578/547", "10/578/548", "10/578/549",
]
UTM33S_BENCHMARK_TILES= ["3/4/4"] * 100

EU3035_BENCHMARK_TILES = [
    "4/3/8", "4/4/7", "4/4/8", "4/4/9", "4/5/9", "4/5/8", "4/4/6", "4/4/10", "4/3/7",
    "4/3/9", "4/6/8", "4/5/7", "4/6/9", "4/6/7", "4/3/10", "4/3/6", "3/2/5", "3/1/5",
    "3/1/4", "3/3/3", "3/1/3", "2/0/1", "3/2/3", "2/1/1", "2/1/2", "4/5/10", "4/2/8",
    "4/5/6", "4/6/10", "4/2/9", "4/2/7", "4/6/6", "4/2/10", "4/2/6", "4/4/11", "4/4/5",
    "4/5/11", "4/5/5", "4/3/11", "4/3/5", "4/6/11", "4/2/11", "4/6/5", "4/2/5", "3/3/5",
    "3/2/2", "3/1/2", "3/3/2",
]

EU3035_HIRES_BENCHMARK_TILES = [
    "6/17/34", "6/17/33", "6/19/33", "6/18/33", "6/18/34", "6/19/34", "6/17/35", "6/18/32",
    "6/18/35", "6/19/35", "6/19/32", "6/20/34", "6/17/32", "6/20/33", "6/16/33", "6/16/34",
    "6/18/36", "6/20/35", "6/16/35", "6/20/32", "6/19/36", "6/17/36", "6/16/32", "6/18/31",
    "6/19/31", "6/17/31", "6/20/36", "6/16/36", "6/16/31", "6/20/31",
]

PARA_BENCHMARK_TILES = [
    "9/263/178", "9/262/178", "9/262/179", "9/264/179", "9/263/179", "9/263/180",
    "9/264/178", "9/261/179", "9/262/180", "9/263/177", "9/262/177", "9/261/178",
    "9/264/180", "9/261/180", "9/264/177", "9/263/181", "9/262/181", "9/261/177",
    "9/265/179", "9/265/178", "9/264/181", "9/265/180", "9/262/176", "9/263/176",
]

HRRR_BENCHMARK_TILES = [
    "4/6/2", "4/5/4", "4/6/3", "4/5/2", "4/6/4", "4/5/3", "4/7/4", "4/5/5", "4/7/2",
    "4/5/1", "4/6/5", "3/2/3", "3/2/0", "3/2/1", "3/2/2", "3/3/1", "3/3/2", "2/1/1",
    "2/1/0", "1/0/0", "5/10/7", "5/10/6", "5/11/6", "5/12/7", "5/11/7", "5/12/6",
    "5/11/5", "5/11/8", "5/12/5", "5/12/8", "5/13/6", "5/10/5", "5/13/7", "5/10/8",
    "5/13/5", "5/13/8", "5/11/4", "5/12/4", "5/10/4", "5/13/4", "4/4/3", "4/4/4",
    "4/4/2", "4/7/3", "4/6/1", "4/3/3", "4/4/5", "4/4/1", "4/3/4", "4/3/2", "4/7/5",
    "4/7/1", "4/5/6", "4/3/5", "4/6/6", "4/3/1", "4/6/0", "4/5/0", "4/4/6", "4/4/0",
    "4/7/0", "4/3/6", "4/7/6", "4/3/0", "3/3/0", "3/1/1", "3/1/2", "3/3/3", "3/1/0",
    "3/1/3", "2/0/0", "2/0/1",
]

# fmt: on

PARA = Dataset(
    name="para",
    dims=(
        Dim(
            name="x",
            size=2000,
            chunk_size=1000,
            data=np.linspace(-58.988125, -45.972125, 2000),
        ),
        Dim(
            name="y",
            size=3000,
            chunk_size=1000,
            data=np.linspace(2.721625, -9.931125, 3000),
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
            "wetland cropland"
        ),
        "flag_values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "flag_colors": "#1f77b4 #17becf #2ca02c #8c564b #ff7f0e #d62728 #bcbd22 #9467bd #e377c2 #7f7f7f",
    },
    setup=partial(
        raster_grid,
        crs="wgs84",
        geotransform="-58.988125 0.006508 0.0 2.721625 0.0 -0.004217583333333333",
        bbox=BBox(west=-58.988125, south=-9.931125, east=-45.972125, north=2.721625),
    ),
    edge_case_tiles=PARA_TILES_EDGE_CASES,
    tiles=PARA_TILES,
    benchmark_tiles=PARA_BENCHMARK_TILES,
)

PARA_HIRES = Dataset(
    name="para_hires",
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
        "flag_colors": "#1f77b4 #17becf #2ca02c #8c564b #ff7f0e #d62728 #bcbd22 #9467bd #e377c2 #7f7f7f #c5b0d5 #ffffff",
    },
    setup=partial(
        raster_grid,
        crs="wgs84",
        geotransform="-58.988125 0.00025 0.0 2.721625 0.0 -0.00025",
        bbox=BBox(west=-58.988125, south=-9.931125, east=-45.972125, north=2.721625),
    ),
    edge_case_tiles=PARA_TILES_EDGE_CASES,
    tiles=PARA_TILES,
    benchmark_tiles=PARA_BENCHMARK_TILES,
)

transformer = pyproj.Transformer.from_crs(HRRR_CRS_WKT, 4326, always_xy=True)
x0, y0 = transformer.transform(237.280472, 21.138123, direction="INVERSE")
x0 = round(x0, 6)
y0 = round(y0, 6)

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
    edge_case_tiles=HRRR_TILES_EDGE_CASES,
    tiles=HRRR_TILES,
    benchmark_tiles=HRRR_BENCHMARK_TILES,
)

EU3035 = Dataset(
    name="eu3035",
    dims=(
        Dim(name="x", size=3000, chunk_size=1000, data=None),
        Dim(name="y", size=3000, chunk_size=1000, data=None),
    ),
    dtype=np.float32,
    setup=partial(
        raster_grid,
        crs="epsg:3035",
        geotransform="2635780.0 1200.0 0.0 5416000.0 0.0 -1200.0",
        bbox=BBox(
            west=-31.39, south=36.96, east=55.51, north=67.12
        ),  # Geographic extent of projected grid
    ),
    edge_case_tiles=ETRS89_TILES_EDGE_CASES,
    tiles=ETRS89_TILES,
    benchmark_tiles=EU3035_BENCHMARK_TILES,
)

EU3035_HIRES = Dataset(
    name="eu3035_hires",
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
    edge_case_tiles=ETRS89_TILES_EDGE_CASES,
    tiles=ETRS89_TILES,
    benchmark_tiles=EU3035_HIRES_BENCHMARK_TILES,
)

UTM33S = Dataset(
    name="utm33s",
    dims=(
        Dim(name="x", size=2000, chunk_size=2000, data=None),
        Dim(name="y", size=5000, chunk_size=2000, data=None),
    ),
    dtype=np.float32,
    setup=partial(
        raster_grid,
        crs="epsg:32733",
        geotransform="166021.44 333.98 0.0 10000000.00 0.0 -1776.62",  # UTM Zone 33S coordinates
        # bbox=BBox(west=-1.763744, south=-80.013566, east=31.763881, north=0.0),
        # bbox=BBox(west=-1.763744, south=-80.013566, east=31.763881, north=0.0),
        bbox=BBox(west=12.0, south=-80.0, east=18.0, north=0),
    ),
    edge_case_tiles=UTM33S_TILES_EDGE_CASES,
    tiles=UTM33S_TILES,
    benchmark_tiles=UTM33S_BENCHMARK_TILES,
)

UTM33S_HIRES = Dataset(
    name="utm33s_hires",
    dims=(
        Dim(name="x", size=27000, chunk_size=2000, data=None),
        Dim(name="y", size=75000, chunk_size=2000, data=None),
    ),
    dtype=np.float32,
    setup=partial(
        raster_grid,
        crs="epsg:32733",
        geotransform="688070.98 0.5 0.0 6809115.47 0.0 -0.5",  # Northern Cape, SA at 0.5m resolution
        bbox=BBox(
            west=16.927608, south=-29.170151, east=17.072642, north=-28.829823
        ),  # Northern Cape patch
    ),
    edge_case_tiles=UTM33S_TILES_EDGE_CASES,
    tiles=UTM33S_TILES,
    benchmark_tiles=UTM33S_BENCHMARK_TILES,
)


FORECAST = xr.decode_cf(
    xr.Dataset.from_dict(
        {
            "coords": {
                "L": {
                    "dims": ("L",),
                    "attrs": {
                        "long_name": "Lead",
                        "standard_name": "forecast_period",
                        "units": "months",
                    },
                    "data": [0, 1],
                },
                "M": {
                    "dims": ("M",),
                    "attrs": {
                        "standard_name": "realization",
                        "long_name": "Ensemble Member",
                        "units": "unitless",
                    },
                    "data": [0, 1, 2],
                },
                "S": {
                    "dims": ("S",),
                    "attrs": {
                        "calendar": "360_day",
                        "long_name": "Forecast Start Time",
                        "standard_name": "forecast_reference_time",
                        "units": "months since 1960-01-01",
                    },
                    "data": [0, 1, 2, 3],
                },
                "X": {
                    "dims": ("X",),
                    "attrs": {
                        "standard_name": "longitude",
                        "units": "degree_east",
                    },
                    "data": [0, 1, 2, 3, 4],
                },
                "Y": {
                    "dims": ("Y",),
                    "attrs": {
                        "standard_name": "latitude",
                        "units": "degree_north",
                    },
                    "data": [0, 1, 2, 3, 4, 5],
                },
            },
            "attrs": {"Conventions": "IRIDL"},
            "dims": {"L": 2, "M": 3, "S": 4, "X": 5, "Y": 6},
            "data_vars": {
                "sst": {
                    "dims": ("S", "L", "M", "Y", "X"),
                    "attrs": {
                        "PDS_TimeRange": 3,
                        "center": "US Weather Service - National Met. Center",
                        "units": "Celsius_scale",
                        "scale_min": -69.97389221191406,
                        "scale_max": 43.039306640625,
                        "long_name": "Sea Surface Temperature",
                        "standard_name": "sea_surface_temperature",
                    },
                    "data": np.arange(np.prod((4, 2, 3, 6, 5))).reshape((4, 2, 3, 6, 5)),
                }
            },
        }
    )
)

CURVILINEAR = Dataset(
    name="curvilinear",
    dims=(
        Dim(name="xi_rho", size=400, chunk_size=200, data=None),
        Dim(name="eta_rho", size=200, chunk_size=100, data=None),
        Dim(
            name="s_rho",
            size=2,
            chunk_size=2,
            data=np.array([0, -1]),
            attrs={
                "long_name": "S-coordinate at RHO-points",
                "valid_min": -1.0,
                "valid_max": 0.0,
                "standard_name": "ocean_s_coordinate_g2",
                "formula_terms": "s: s_rho C: Cs_r eta: zeta depth: h depth_c: hc",
                "field": "s_rho, scalar",
            },
        ),
    ),
    dtype=np.float64,
    setup=curvilinear_grid,
)


POPDS = xr.Dataset(
    {
        "TEMP": (
            ("nlat", "nlon"),
            np.ones((20, 30)) * 15,
            {
                "coordinates": "TLONG TLAT",
                "standard_name": "sea_water_potential_temperature",
            },
        )
    },
    coords={
        "TLONG": (
            ("nlat", "nlon"),
            np.ones((20, 30)),
            {"units": "degrees_east"},
        ),
        "TLAT": (
            ("nlat", "nlon"),
            2 * np.ones((20, 30)),
            {"units": "degrees_north"},
        ),
        "ULONG": (
            ("nlat", "nlon"),
            0.5 * np.ones((20, 30)),
            {"units": "degrees_east"},
        ),
        "ULAT": (
            ("nlat", "nlon"),
            2.5 * np.ones((20, 30)),
            {"units": "degrees_north"},
        ),
        "UVEL": (
            ("nlat", "nlon"),
            np.ones((20, 30)) * 15,
            {"coordinates": "ULONG ULAT", "standard_name": "sea_water_x_velocity"},
        ),
        "nlon": ("nlon", np.arange(30), {"axis": "X"}),
        "nlat": ("nlat", np.arange(20), {"axis": "Y"}),
    },
)
