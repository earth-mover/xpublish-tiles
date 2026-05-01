"""Library utility functions for xpublish-tiles."""

import asyncio
import io
import math
import operator
from collections.abc import Hashable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache, partial
from itertools import product
from typing import TYPE_CHECKING, Any, cast

import matplotlib.colors as mcolors
import numba
import numpy as np
import pyproj
import toolz as tlz
from PIL import Image
from pyproj import CRS
from pyproj.aoi import BBox
from skimage.restoration import unwrap_phase

import xarray as xr
from xpublish_tiles.config import config
from xpublish_tiles.utils import NUMBA_THREADING_LOCK

if TYPE_CHECKING:
    from xpublish_tiles.grids import (
        GridMetadata,
        GridSystem,
        Slicer,
        Slicers,
    )
from xpublish_tiles.logger import logger

WGS84_SEMI_MAJOR_AXIS = np.float64(6378137.0)  # from proj
M_PI = 3.14159265358979323846  # from proj
M_2_PI = 6.28318530717958647693  # from proj


def unwrap(data: np.ndarray, *, width: float, axis: int | None = None) -> np.ndarray:
    """Remove ±width discontinuities from ``data``.

    When ``axis`` is given, uses ``np.unwrap`` along that axis.
    Use this whenever the discontinuity is per-row/per-column

    When ``axis`` is None, falls back to ``skimage.restoration.unwrap_phase``
    for cases where the discontinuity has 2D structure (e.g. tripole).
    """
    if axis is not None:
        return np.unwrap(data, period=width, axis=axis)
    factor = 2 * np.pi / width
    un = unwrap_phase(data * factor)
    un /= factor
    return un


@dataclass(frozen=True)
class PadDimension:
    """Helper class to encapsulate padding parameters for a dimension."""

    name: str
    size: int
    left_pad: int = field(default_factory=lambda: config.get("default_pad"))
    right_pad: int = field(default_factory=lambda: config.get("default_pad"))
    wraparound: bool = False
    prevent_overlap: bool = False
    fill: bool = False


@dataclass
class Fill:
    size: int


@dataclass
class CoarsenedCoordinateIndices:
    """Index arrays for coarsening ``total`` cells by ``factor``.

    All indices are offset by ``offset`` (default 0), so they index
    directly into the original edge/coordinate arrays.
    ``starts`` is computed eagerly; ``centers()`` and ``ends()`` compute
    on demand.
    """

    total: int
    factor: int
    offset: int = 0
    starts: np.ndarray = field(init=False)

    def __post_init__(self):
        n = math.ceil(self.total / self.factor)
        self.starts = np.arange(n) * self.factor + self.offset

    def centers(self) -> np.ndarray:
        return np.minimum(self.starts + self.factor // 2, self.offset + self.total - 1)

    def ends(self) -> np.ndarray:
        return np.minimum(self.starts + self.factor, self.offset + self.total) - 1


def _coarsen_indices_impl(
    slicers: dict[str, list],
    dims: list[str],
    coarsen_factors: dict[str, int],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute left/right edge indices into the original edge arrays for each coarsened cell.

    Uses :class:`CoarsenedCoordinateIndices` for the stride math.  For multi-slice
    (wraparound) dims the slices are flattened first, because a coarsened
    cell can span the boundary between slices.
    """
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for dim in dims:
        slices = [s for s in slicers[dim] if isinstance(s, slice)]
        factor = coarsen_factors.get(dim, 1)

        if len(slices) == 1:
            s = slices[0]
            w = CoarsenedCoordinateIndices(s.stop - s.start, factor, offset=s.start)
            starts, ends = w.starts, w.ends()
        else:
            orig = np.concatenate([np.arange(s.start, s.stop) for s in slices])
            w = CoarsenedCoordinateIndices(orig.size, factor)
            starts = orig[w.starts]
            ends = orig[w.ends()]

        result[dim] = (starts, ends)
    return result


def crs_repr(crs: CRS | None) -> str:
    """Generate a concise representation string for a CRS object.

    Args:
        crs: pyproj CRS object or None

    Returns:
        String representation of the CRS
    """
    if crs is None:
        return "None"

    # Try to get EPSG code first, fallback to shorter description
    try:
        if hasattr(crs, "to_epsg") and crs.to_epsg():
            return f"<CRS: EPSG:{crs.to_epsg()}>"
        else:
            # Use the name if available, otherwise authority:code
            crs_name = getattr(crs, "name", str(crs)[:50] + "...")
            return f"<CRS: {crs_name}>"
    except Exception:
        # Fallback to generic representation
        return "<CRS>"


class BenchmarkImportError(ImportError):
    """Raised when the user attempts to benchmark without needed dependencies."""

    def __init__(self):
        super().__init__(
            "Additional dependencies are required for benchmarking."
            " Please install xpublish-tiles[testing]."
        )


class TileTooBigError(Exception):
    """Raised when a tile request would result in too much data to render."""

    pass


class VariableNotFoundError(Exception):
    """Raised when the user-requested variable cannot be found."""

    pass


class IndexingError(Exception):
    """Raised when an invalid coordinate is passed for selection."""

    pass


class MissingParameterError(Exception):
    """Raised when an expected parameter (e.g. colorscalerange) is not passed."""

    pass


class AsyncLoadTimeoutError(Exception):
    """Raised when async data loading times out."""

    pass


THREAD_POOL_NUM_THREADS = config.get("num_threads")
logger.info("setting up thread pool with num threads: %s", THREAD_POOL_NUM_THREADS)
EXECUTOR = ThreadPoolExecutor(
    max_workers=THREAD_POOL_NUM_THREADS,
    thread_name_prefix="xpublish-tiles-pool",
)

# Dictionary to store semaphores per event loop
_semaphores: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}
_data_load_semaphores: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}


def _get_semaphore(loop) -> asyncio.Semaphore:
    """Get or create a semaphore for the current event loop."""
    if loop is None:
        loop = asyncio.get_event_loop()
    if loop not in _semaphores:
        _semaphores[loop] = asyncio.Semaphore(config.get("num_threads"))
    return _semaphores[loop]


def get_data_load_semaphore() -> asyncio.Semaphore:
    """Get or create a data load semaphore for the current event loop."""
    loop = asyncio.get_running_loop()
    if loop not in _data_load_semaphores:
        _data_load_semaphores[loop] = asyncio.Semaphore(
            config.get("num_concurrent_data_loads")
        )
    return _data_load_semaphores[loop]


async def async_run(func, *args, **kwargs):
    """Run a function in the thread pool executor with semaphore limiting."""
    loop = asyncio.get_running_loop()
    semaphore = _get_semaphore(loop)
    async with semaphore:
        return await loop.run_in_executor(EXECUTOR, func, *args, **kwargs)


def sync_load_async(obj: xr.DataArray | xr.Dataset) -> None:
    """Run ``obj.load_async()`` from sync code, updating ``obj`` in place.

    Lets xarray handle concurrency internally (``Dataset.load_async`` gathers
    its variables). Safe regardless of whether an event loop is already
    running on the calling thread: when a loop is detected we delegate to a
    worker thread that gets a fresh loop via ``asyncio.run``.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(obj.load_async())
        return
    EXECUTOR.submit(lambda: asyncio.run(obj.load_async())).result()


# 4326 with order of axes reversed.
OTHER_4326 = pyproj.CRS.from_user_input("WGS 84 (CRS84)")

# https://pyproj4.github.io/pyproj/stable/advanced_examples.html#caching-pyproj-objects
transformer_from_crs = lru_cache(partial(pyproj.Transformer.from_crs, always_xy=True))


# benchmarked with
# import numpy as np
# import pyproj
# from src.xpublish_tiles.lib import transform_blocked

# x = np.linspace(2635840.0, 3874240.0, 500)
# y = np.linspace(5415940.0, 2042740, 500)

# transformer = pyproj.Transformer.from_crs(3035, 4326, always_xy=True)
# grid = np.meshgrid(x, y)


# %timeit transform_blocked(*grid, chunk_size=(20, 20), transformer=transformer)
# %timeit transform_blocked(*grid, chunk_size=(100, 100), transformer=transformer)
# %timeit transform_blocked(*grid, chunk_size=(250, 250), transformer=transformer)
# %timeit transform_blocked(*grid, chunk_size=(500, 500), transformer=transformer)
# %timeit transformer.transform(*grid)
#
# 500 x 500 grid:
# 19.1 ms ± 1.64 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 10.9 ms ± 113 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# 13.8 ms ± 222 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# 48.6 ms ± 318 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 49.6 ms ± 3.38 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
#
# 2000 x 2000 grid:
# 302 ms ± 21.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 156 ms ± 1.36 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 155 ms ± 2.75 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 156 ms ± 5.07 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 772 ms ± 27 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
def get_transform_chunk_size(da: xr.DataArray):
    """Get the chunk size for coordinate transformations dynamically."""
    chunk_size = config.get("transform_chunk_size")
    # This way the chunks are C-contiguous and we avoid a memory copy inside pyproj \m/
    return (max(chunk_size * chunk_size // da.shape[-1], 1), da.shape[-1])


def is_degree_geographic(crs: CRS) -> bool:
    """True for any geographic CRS with lon/lat axes in degrees (EPSG:4326,
    CRS84, custom spherical datums like HEALPix's, etc.). The 4326-fastpath
    in :func:`transform_coordinates` uses this to skip the pyproj roundtrip
    and just wrap lon to [-180, 180], which is valid for all such CRSes —
    any residual datum shift is sub-meter and below pixel resolution.
    """
    return crs.is_geographic and all(ax.unit_name == "degree" for ax in crs.axis_info)


def epsg4326to3857(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = WGS84_SEMI_MAJOR_AXIS

    x = np.asarray(lon, dtype=np.float64, copy=True)
    y = np.asarray(lat, dtype=np.float64, copy=True)

    # Only normalize longitude values that are outside the [-180, 180] range
    # This preserves precision for values already in the valid range
    # pyproj accepts both -180 and 180 as valid values without wrapping
    needs_normalization = (x > 180) | (x < -180)

    np.deg2rad(x, out=x)
    if np.any(needs_normalization):
        # Only normalize the values that need it to preserve precision
        # doing it this way matches proj
        x[needs_normalization] = ((x[needs_normalization] + M_PI) % (2 * M_PI)) - M_PI
    # Clamp latitude to avoid infinity at poles in-place
    # Web Mercator is only valid between ~85.05 degrees
    # Given our padding, we may be sending in data at latitudes poleward of MAX_LAT
    # MAX_LAT = 85.051128779806604  # atan(sinh(pi)) * 180 / pi
    # np.clip(y, -MAX_LAT, MAX_LAT, out=y)

    # Y coordinate: use more stable formula for large latitudes
    # Using: y = a * asinh(tan(φ)) for better numerical stability
    # following the proj formula
    # https://github.com/OSGeo/PROJ/blob/ff43c46b19802f5953a1546b05f59c5b9ee65795/src/projections/merc.cpp#L14
    # https://proj.org/en/stable/operations/projections/merc.html#forward-projection
    # Note: WebMercator uses the "spherical form"
    np.deg2rad(y, out=y)
    np.tan(y, out=y)
    np.arcsinh(y, out=y)

    x *= a
    y *= a

    return x, y


def aeqd_to_4326(
    x_m: np.ndarray,
    y_m: np.ndarray,
    center_lat: float,
    center_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast azimuthal equidistant (meters) to EPSG:4326 (degrees) conversion.

    Uses flat-earth approximation. Accurate to 0.3% at 300km from center.
    ~200x faster than pyproj for large arrays. Modifies x_m and y_m in place.
    """
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * np.cos(np.radians(center_lat))
    x_m /= meters_per_deg_lon
    x_m += center_lon
    y_m /= meters_per_deg_lat
    y_m += center_lat
    return x_m, y_m


def slices_from_chunks(chunks):
    """Slightly modified from dask.array.core.slices_from_chunks to be lazy."""
    cumdims = [tlz.accumulate(operator.add, bds, 0) for bds in chunks]
    slices = (
        (slice(s, s + dim) for s, dim in zip(starts, shapes, strict=False))
        for starts, shapes in zip(cumdims, chunks, strict=False)
    )
    return product(*slices)


def transform_chunk(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    slices: tuple[slice, slice],
    transformer: pyproj.Transformer,
    x_out: np.ndarray,
    y_out: np.ndarray,
    inplace: bool = False,
) -> None:
    """Transform a chunk of coordinates."""
    row_slice, col_slice = slices
    x_chunk = x_grid[row_slice, col_slice]
    y_chunk = y_grid[row_slice, col_slice]
    assert x_chunk.flags["C_CONTIGUOUS"]
    assert y_chunk.flags["C_CONTIGUOUS"]
    x_transformed, y_transformed = transformer.transform(
        x_chunk, y_chunk, inplace=inplace
    )
    if not inplace:
        x_out[row_slice, col_slice] = x_transformed
        y_out[row_slice, col_slice] = y_transformed


async def transform_blocked(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    transformer: pyproj.Transformer,
    chunk_size: tuple[int, int],
    inplace: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Blocked transformation using thread pool."""

    shape = x_grid.shape
    if inplace:
        x_out, y_out = x_grid, y_grid
    else:
        x_out = np.empty(shape, dtype=x_grid.dtype)
        y_out = np.empty(shape, dtype=y_grid.dtype)

    chunk_rows, chunk_cols = chunk_size

    # Generate chunks for each dimension
    row_chunks = [min(chunk_rows, shape[0] - i) for i in range(0, shape[0], chunk_rows)]
    col_chunks = [min(chunk_cols, shape[1] - j) for j in range(0, shape[1], chunk_cols)]

    chunks = (row_chunks, col_chunks)
    await asyncio.gather(
        *[
            async_run(
                transform_chunk,
                x_grid,
                y_grid,
                slices,
                transformer,
                x_out,
                y_out,
                inplace,
            )
            for slices in slices_from_chunks(chunks)
        ]
    )
    return x_out, y_out


def check_transparent_pixels(image_bytes):
    """Check the percentage of transparent pixels in a PNG image."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    arr = np.array(img)
    transparent_mask = arr[:, :, 3] == 0
    transparent_count = np.sum(transparent_mask)
    total_pixels = arr.shape[0] * arr.shape[1]

    return (transparent_count / total_pixels) * 100


async def transform_coordinates(
    subset: xr.DataArray,
    grid_x_name: str,
    grid_y_name: str,
    transformer: pyproj.Transformer,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Transform coordinates from input CRS to output CRS.

    This function broadcasts the X and Y coordinates and then transforms them
    using either chunked or direct transformation based on the data size.

    It attempts to preserve rectilinear-ness when possible: 4326 -> 3857

    Parameters
    ----------
    subset : xr.DataArray
        The subset data array containing coordinates to transform
    grid_x_name : str
        Name of the X coordinate dimension
    grid_y_name : str
        Name of the Y coordinate dimension
    transformer : pyproj.Transformer
        The coordinate transformer
    chunk_size : tuple[int, int], optional
        Chunk size for blocked transformation, by default from config

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        Transformed X and Y coordinate arrays
    """

    inx, iny = subset[grid_x_name], subset[grid_y_name]

    assert transformer.source_crs is not None
    assert transformer.target_crs is not None

    # the ordering of these two fastpaths is important
    # we want to normalize to -180 -> 180 always
    if is_degree_geographic(transformer.source_crs) and is_degree_geographic(
        transformer.target_crs
    ):
        # pyproj does not normalize these, and inputs can arrive in either the
        # 0–360 or -180–180 convention (e.g. HEALPix corner vertices come out
        # in 0–360). Preserve dtype; shift any out-of-range values.
        newdata = inx.data.copy()
        np.subtract(newdata, 360, out=newdata, where=newdata >= 180)
        np.add(newdata, 360, out=newdata, where=newdata < -180)
        return inx.copy(data=newdata), iny

    if transformer.source_crs == transformer.target_crs:
        return inx, iny

    # preserve rectilinear-ness by reimplementing this (easy) transform
    if (inx.ndim == 1 and iny.ndim == 1) and (
        transformer == transformer_from_crs(4326, 3857)
        or transformer == transformer_from_crs(OTHER_4326, 3857)
    ):
        newx, newy = epsg4326to3857(inx.data, iny.data)
        _clamp_infinite(newx)
        _clamp_infinite(newy)
        return inx.copy(data=newx), iny.copy(data=newy)

    # Broadcast coordinates
    # FIXME: dropping indexes is a workaround for broadcasting RasterIndex
    bx, by = xr.broadcast(
        inx.drop_indexes(inx.dims, errors="ignore"),
        iny.drop_indexes(iny.dims, errors="ignore"),
    )
    assert bx.dims == by.dims

    chunk_size = get_transform_chunk_size(bx)
    if bx.size > math.prod(chunk_size):
        # Ensure we have C-contiguous float64 arrays (required by pyproj).
        # np.asarray avoids copying if already float64 and C-contiguous.
        # This is numpy 2.X behaviour
        newX = np.asarray(bx.data, order="C", dtype=np.float64)
        newY = np.asarray(by.data, order="C", dtype=np.float64)
        await transform_blocked(
            newX,
            newY,
            transformer,
            chunk_size,
            inplace=True,
        )
    else:
        newX, newY = await async_run(transformer.transform, bx.data, by.data)

    if not transformer.target_crs.is_geographic:
        _clamp_infinite(newX)
        _clamp_infinite(newY)

    return bx.copy(data=newX), by.copy(data=newY)


def _clamp_infinite(arr: np.ndarray) -> None:
    """In-place: replace ±inf with ±float max.

    Poles map to ±inf under Web Mercator (and similar projections). Datashader
    drops polygons touching non-finite vertices, leaving holes in the polar
    caps. Clamping to finite values lets those cells still rasterize.
    """
    # A value large relative to projected extents (Web Mercator ≈ ±2e7) but
    # well inside float64 precision so downstream polygon math (datashader's
    # ray-casting / bbox tests) stays numerically well-behaved.
    _BIG = 1e15
    np.nan_to_num(arr, copy=False, nan=np.nan, posinf=_BIG, neginf=-_BIG)


def _prevent_slice_overlap(indexers: list[slice]) -> list[slice]:
    """
    Prevent overlapping slices by adjusting stop positions.

    This mimics the original logic: if a slice's stop position would overlap
    with a previously added slice's start, adjust the stop to prevent overlap.
    This is used for anti-meridian longitude selections where slices may be
    processed in an order that could cause overlaps.
    """
    if len(indexers) <= 1:
        return indexers
    result = []
    for indexer in indexers:
        start, stop, step = indexer.start, indexer.stop, indexer.step
        if len(result) > 0 and stop >= result[-1].start:
            stop = result[-1].start
        result.append(slice(start, stop, step))
    return result


def pad_slicers(
    slicers: "Slicers",
    *,
    dimensions: list[PadDimension] | None = None,
) -> "Slicers":
    """
    Apply padding to slicers for specified dimensions.

    Parameters
    ----------
    slicers : dict[str, list[slice | Fill]]
        Dictionary mapping dimension names to lists of slices or Fill objects
    dimensions : list[PadDimension]
        List of dimension padding information

    Returns
    -------
    dict[str, list[slice | Fill]]
        Dictionary mapping dimension names to lists of padded slices or Fill objects
    """
    if not dimensions:
        return slicers.copy()

    result = {}
    # Handle each specified dimension
    for dim in dimensions:
        if dim.name not in slicers:
            continue

        dim_slicers = cast(list[slice], slicers[dim.name])
        indexers = [slice(*idxr.indices(dim.size)) for idxr in dim_slicers]

        # Prevent overlap if requested (before padding)
        if dim.prevent_overlap:
            indexers = _prevent_slice_overlap(indexers)

        # Apply padding
        first, last = indexers[0], indexers[-1]
        left_edge = first.start - dim.left_pad
        right_edge = last.stop + dim.right_pad

        indexers_with_fill: list[slice | Fill] = []
        if len(indexers) == 1:
            indexers_with_fill = [slice(max(0, left_edge), min(dim.size, right_edge))]
        else:
            indexers_with_fill = [
                slice(max(0, left_edge), first.stop),
                *indexers[1:-1],
                slice(last.start, min(dim.size, right_edge)),
            ]

        # Apply wraparound if enabled for this dimension
        if dim.wraparound:
            if left_edge < 0:
                # Padding would extend below 0, wrap from end
                indexers_with_fill = [slice(left_edge, None), *indexers_with_fill]
            if right_edge > dim.size:
                # Padding would extend beyond size, wrap from beginning
                indexers_with_fill = indexers_with_fill + [
                    slice(0, right_edge - dim.size)
                ]
        elif dim.fill:
            # Note: This is unused at the moment since we skip padding for coarsening
            left_over = left_edge if left_edge < 0 else 0
            right_over = max(right_edge - dim.size, 0)
            if left_over:
                indexers_with_fill = [Fill(abs(left_over)), *indexers_with_fill]
            if right_over:
                indexers_with_fill = [*indexers_with_fill, Fill(abs(right_over))]

        result[dim.name] = indexers_with_fill

    # Pass through any other dimensions unchanged
    for key, value in slicers.items():
        if key not in result:
            result[key] = value

    return result


def normalize_slicers(
    slicers: "Slicers",
    dim_sizes: "Mapping[Hashable, int]",
) -> "Slicers":
    return {
        dim: [
            slice(*s.indices(dim_sizes[dim])) if isinstance(s, slice) else s
            for s in entries
        ]
        for dim, entries in slicers.items()
    }


def apply_default_pad(slicers, da, grid):
    """Apply default padding for edge safety (floating-point roundoff protection).

    This is only necessary because we cannot provide cell edges to datashader.
    Instead, it re-infers cell edges given cell centers as inputs.

    Parameters
    ----------
    slicers : dict[str, list[slice | Fill | UgridIndexer | HealpixIndexer]]
        Raw slicers from grid.sel()
    da : xr.DataArray
        Data array (for dimension sizes)
    grid : GridSystem2D
        Grid system information

    Returns
    -------
    dict[str, list[slice | Fill | UgridIndexer | HealpixIndexer]]
        Slicers with default_pad applied
    """
    default_padders = [
        PadDimension(
            name=grid.Xdim, size=da.sizes[grid.Xdim], wraparound=grid.lon_spans_globe
        ),
        PadDimension(name=grid.Ydim, size=da.sizes[grid.Ydim], wraparound=False),
    ]
    return pad_slicers(slicers, dimensions=default_padders)


def slicers_to_pad_instruction(slicers, datatype) -> dict[str, Any]:
    from xpublish_tiles.types import DiscreteData

    pad_kwargs = {}
    pad_widths = {}
    for dim in slicers:
        pad_width = []
        sl = slicers[dim]
        pad_width.append(sl[0].size if isinstance(sl[0], Fill) else 0)
        pad_width.append(sl[-1].size if isinstance(sl[-1], Fill) else 0)
        if pad_width != [0, 0]:
            pad_widths[dim] = pad_width
    if pad_widths:
        pad_kwargs["pad_width"] = pad_widths
        pad_kwargs["mode"] = "edge" if isinstance(datatype, DiscreteData) else "constant"
    return pad_kwargs


def polygons_from_rings(rings: np.ndarray):
    """Build a spatialpandas PolygonArray from a ``(N, M, 2)`` rings buffer.

    Each row is one polygon with M vertices (including the closing vertex).

    We choose to use `spatialpandas` instead of `geopandas` because it's internal
    data structure (ragged arrays backed by pyarrow buffers) is what the renderer
    requires internally.

    Spatialpandas' ``PolygonArray`` stores geometries natively in ragged
    form backed by pyarrow buffers, so we construct it directly:
    - ``inner``: the flat coord buffer, zero-copy via ``pa.py_buffer``
    - ``rings_arr``: list-array whose offsets are ``0, 2M, 4M, …``
    - ``polys``: list-array of rings, one ring per polygon

    Orientation (CW vs CCW) is not normalized — datashader's winding-number
    rasterizer is orientation-agnostic, so we skip the per-polygon reorient.
    """
    import pyarrow as pa
    from spatialpandas.geometry import PolygonArray

    n, m, _ = rings.shape
    flat = np.ascontiguousarray(rings.reshape(-1), dtype=np.float64)
    ring_stride = m * 2

    inner = pa.Array.from_buffers(pa.float64(), flat.size, [None, pa.py_buffer(flat)], 0)
    ring_offsets = np.arange(0, flat.size + 1, ring_stride, dtype=np.int32)
    rings_arr = pa.ListArray.from_arrays(pa.array(ring_offsets), inner)
    poly_offsets = np.arange(n + 1, dtype=np.int32)
    polys = pa.ListArray.from_arrays(pa.array(poly_offsets), rings_arr)
    return PolygonArray(polys)


@numba.njit(parallel=True, cache=True, boundscheck=False)
def fill_rings_from_corners(out, corner_x, corner_y):
    """Fill a (n0, n1, 5, 2) ring array from a (n0+1, n1+1) corner grid.

    Ring order: (i,j), (i,j+1), (i+1,j+1), (i+1,j), (i,j) [close].
    """
    n0, n1 = out.shape[0], out.shape[1]
    for i in numba.prange(n0):  # ty: ignore[not-iterable]
        for j in range(n1):
            out[i, j, 0, 0] = corner_x[i, j]
            out[i, j, 0, 1] = corner_y[i, j]
            out[i, j, 1, 0] = corner_x[i, j + 1]
            out[i, j, 1, 1] = corner_y[i, j + 1]
            out[i, j, 2, 0] = corner_x[i + 1, j + 1]
            out[i, j, 2, 1] = corner_y[i + 1, j + 1]
            out[i, j, 3, 0] = corner_x[i + 1, j]
            out[i, j, 3, 1] = corner_y[i + 1, j]
            out[i, j, 4, 0] = corner_x[i, j]
            out[i, j, 4, 1] = corner_y[i, j]


def apply_range_colors(
    cmap: mcolors.Colormap,
    abovemaxcolor: str | None,
    belowmincolor: str | None,
) -> mcolors.Colormap:
    """Apply over/under colors to a colormap for out-of-range values.

    Args:
        cmap: The colormap to modify
        abovemaxcolor: Color for values above colorscalerange max.
            "extend" or None = use max palette color (default behavior)
            "transparent" = fully transparent
            Otherwise = color value (hex or named)
        belowmincolor: Color for values below colorscalerange min.
            "extend" or None = use min palette color (default behavior)
            "transparent" = fully transparent
            Otherwise = color value (hex or named)

    Returns:
        A copy of the colormap with over/under colors applied
    """
    cmap = cmap.copy()

    if abovemaxcolor is not None and abovemaxcolor != "extend":
        if abovemaxcolor == "transparent":
            cmap.set_over((0, 0, 0, 0))
        else:
            cmap.set_over(abovemaxcolor)

    if belowmincolor is not None and belowmincolor != "extend":
        if belowmincolor == "transparent":
            cmap.set_under((0, 0, 0, 0))
        else:
            cmap.set_under(belowmincolor)

    return cmap


def create_colormap_from_dict(colormap_dict: dict[str, str]) -> mcolors.Colormap:
    """Create a matplotlib colormap from a dictionary of index->color mappings."""
    # Sort by numeric keys to ensure proper order
    sorted_items = sorted(colormap_dict.items(), key=lambda x: int(x[0]))

    # Extract positions (normalized 0-1) and colors
    positions = []
    colors = []

    for key, color in sorted_items:
        position = int(key) / 255.0  # Normalize to 0-1 range
        positions.append(position)
        colors.append(color)

    if positions[0] != 0 and positions[-1] != 1:
        # this is a matplotlib requirement
        raise ValueError("Provided colormap keys must contain 0 and 255.")

    return mcolors.LinearSegmentedColormap.from_list(
        "custom", list(zip(positions, colors, strict=True)), N=256
    )


def create_listed_colormap_from_dict(
    colormap_dict: dict[str, str], flag_values: Sequence[Hashable]
) -> dict[Hashable, str]:
    """Create a matplotlib ListedColormap from a dictionary of flag_value->color mappings.

    For categorical data, the colormap must have exactly as many entries as flag_values.
    Every key in the colormap must correspond to a flag_value.
    Keys should be string representations of the flag values, and values should be hex colors.
    """
    # Validate that all colormap keys are in flag_values
    flag_values_str = {str(v) for v in flag_values}
    colormap_keys = set(colormap_dict.keys())

    # Check for colormap keys that don't correspond to any flag_value
    invalid_keys = colormap_keys - flag_values_str
    if invalid_keys:
        raise ValueError(
            f"colormap contains keys not in flag_values: {sorted(invalid_keys)}. "
            f"Valid flag_values: {sorted(flag_values_str)}"
        )

    # Check for flag_values that don't have colormap entries
    missing_keys = flag_values_str - colormap_keys
    if missing_keys:
        raise ValueError(
            f"colormap is missing entries for flag_values: {sorted(missing_keys)}. "
            f"All flag_values must have corresponding colors."
        )

    # Build colormap in the order of flag_values
    colors = {flag_value: colormap_dict[str(flag_value)] for flag_value in flag_values}
    return colors


@numba.jit(nopython=True, parallel=True, cache=True)
def _coarsen_nanmean_2d(arr, fy, fx, out):
    """Coarsen with nanmean, handling incomplete edge windows."""
    ny_out, nx_out = out.shape
    H, W = arr.shape

    for i in numba.prange(ny_out):  # ty: ignore[not-iterable]
        y_start = i * fy
        y_end = min((i + 1) * fy, H)

        for j in range(nx_out):
            x_start = j * fx
            x_end = min((j + 1) * fx, W)

            total = 0.0
            count = 0
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    val = arr[y, x]
                    if not np.isnan(val):
                        total += val
                        count += 1

            out[i, j] = total / count if count > 0 else np.nan


def maybe_cast_data(data: xr.DataArray) -> xr.DataArray:
    """Upcast small floats (e.g. float16) to float32 for numba compatibility."""
    dtype = data.dtype
    totype = str(dtype.str)
    if dtype.kind == "f" and dtype.itemsize < 4:
        totype = totype[:-1] + "4"
    return data.astype(totype, copy=False)


def coarsen_mean_pad(da: xr.DataArray, factors: dict[str, int]) -> xr.DataArray:
    """Memory-efficient coarsen with boundary='pad' and nanmean."""
    da = maybe_cast_data(da)
    dims = da.dims
    arr = da.data
    H, W = arr.shape

    fy, fx = tuple(factors.get(str(dim), 1) for dim in dims)
    out = np.empty((math.ceil(H / fy), math.ceil(W / fx)), dtype=np.float64)
    with NUMBA_THREADING_LOCK:
        _coarsen_nanmean_2d(arr, fy, fx, out)
    return xr.DataArray(out, dims=dims, name=da.name)


def _get_indexer_size(sl: "Slicer", dim_size: int | None = None) -> int:
    """Get the size of an indexer (slice, Fill, UgridIndexer, or ndarray)."""
    from xpublish_tiles.grids import HealpixIndexer, UgridIndexer

    if isinstance(sl, Fill | UgridIndexer):
        return sl.size
    elif isinstance(sl, HealpixIndexer):
        return sl.size(dim_size)
    elif isinstance(sl, slice):
        start = sl.start if sl.start is not None else 0
        if sl.stop is not None:
            stop = sl.stop
        elif dim_size is not None:
            stop = dim_size
        else:
            raise ValueError("dim_size is required for open-ended slices")
        return stop - start
    else:
        raise TypeError(f"Unknown indexer type: {type(sl)!r}")


def _iter_subset_shapes(
    slicers: "Slicers",
    da: xr.DataArray,
    grid: "GridSystem",
):
    """Iterate over individual subset shapes from slicers.

    Yields tuple shapes for each subset that will be created.
    For GridSystem2D, yields (x_size, y_size) for each X slice.
    For Triangular, yields (size,) for the single slice.
    """
    from xpublish_tiles.grids import FacetedGridSystem, FacetedIndexer, Triangular

    if isinstance(grid, Triangular):
        yield (_get_indexer_size(next(iter(slicers[grid.dim])), da.sizes[grid.dim]),)
        return

    if isinstance(grid, FacetedGridSystem):
        indexer = slicers[grid.face_dim][0]
        assert isinstance(indexer, FacetedIndexer)
        for face_slicers in indexer.selections:
            face_slice = face_slicers[grid.face_dim][0]
            assert isinstance(face_slice, slice)
            face = grid.faces[face_slice.start]
            y_size = _get_indexer_size(face_slicers[face.Ydim][0], da.sizes[face.Ydim])
            for sl in face_slicers[face.Xdim]:
                yield (_get_indexer_size(sl, da.sizes[face.Xdim]), y_size)
        return

    yslice = None
    for candidate in slicers[grid.Ydim]:
        if isinstance(candidate, slice):
            yslice = candidate
            break

    if yslice is None:
        yslice = slicers[grid.Ydim][0]

    y_size = _get_indexer_size(yslice, da.sizes[grid.Ydim])

    for sl in slicers[grid.Xdim]:
        x_size = _get_indexer_size(sl, da.sizes[grid.Xdim])
        yield (x_size, y_size)


def check_data_is_renderable_size(
    slicers: "Slicers",
    da: xr.DataArray,
    grid: "GridSystem",
    alternate: "GridMetadata",
    *,
    style: str = "raster",
) -> bool:
    """Check if given slicers produce data of renderable size without loading data.

    Parameters
    ----------
    slicers : dict[str, list[slice | Fill | UgridIndexer | HealpixIndexer]]
        Slicers for data selection
    da : xr.DataArray
        Data array (only metadata used, no data loaded)
    grid : GridSystem
        Grid system information
    alternate : GridMetadata
        Alternate grid metadata

    Returns
    -------
    bool
        True if data is within renderable size limits, False if too big
    """
    has_alternate = alternate.crs != grid.crs
    factor = 3 if has_alternate else 1

    total_size = sum(math.prod(shape) for shape in _iter_subset_shapes(slicers, da, grid))
    if style == "polygons":
        total_size *= grid.npoints_per_geometry
    return total_size * da.dtype.itemsize <= factor * config.get("max_renderable_size")


def max_render_shape(
    *, style: str, width: int = 256, height: int = 256
) -> tuple[int, int]:
    """Compute the per-axis max data shape for coarsening, given the render style.

    For raster: ``max_pixel_factor * tile_size`` per axis.
    For polygons: derived from ``max_num_geometries`` so that
    ``product(max_shape) <= max_num_geometries``.
    """
    if style == "polygons":
        max_num = config.get("max_num_geometries")
        aspect = width / height
        max_h = int(math.sqrt(max_num / aspect))
        max_w = int(max_h * aspect)
        return (max_w, max_h)
    pixel_factor = config.get("max_pixel_factor")
    return (pixel_factor * width, pixel_factor * height)


def round_bbox(bbox: BBox) -> BBox:
    # https://github.com/developmentseed/morecantile/issues/175
    # the precision in morecantile tile bounds isn't perfect,
    # a good way to test is `tms.bounds(Tile(0,0,0))` which should
    # match the spec exactly: https://docs.ogc.org/is/17-083r4/17-083r4.html#toc48
    # Example: tests/test_pipeline.py::test_pipeline_tiles[-90->90,0->360-wgs84_prime_meridian(2/2/1)]
    return BBox(
        west=round(bbox.west, 8),
        south=round(bbox.south, 8),
        east=round(bbox.east, 8),
        north=round(bbox.north, 8),
    )


def sum_tuples(*tuples):
    return tuple(sum(values) for values in zip(*tuples, strict=False))
