"""Library utility functions for xpublish-tiles."""

import asyncio
import io
import math
import operator
from collections.abc import Hashable, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from functools import lru_cache, partial
from itertools import product
from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
import numpy as np
import pyproj
import toolz as tlz
from PIL import Image
from pyproj import CRS

import xarray as xr
from xpublish_tiles.config import config

if TYPE_CHECKING:
    from xpublish_tiles.grids import UgridIndexer
from xpublish_tiles.logger import logger

WGS84_SEMI_MAJOR_AXIS = np.float64(6378137.0)  # from proj
M_PI = 3.14159265358979323846  # from proj
M_2_PI = 6.28318530717958647693  # from proj


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


THREAD_POOL_NUM_THREADS = config.get("num_threads")
logger.info("setting up thread pool with num threads: %s", THREAD_POOL_NUM_THREADS)
EXECUTOR = ThreadPoolExecutor(
    max_workers=THREAD_POOL_NUM_THREADS,
    thread_name_prefix="xpublish-tiles-pool",
)

# Dictionary to store semaphores per event loop
_semaphores: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}


def _get_semaphore(loop) -> asyncio.Semaphore:
    """Get or create a semaphore for the current event loop."""
    if loop is None:
        loop = asyncio.get_event_loop()
    if loop not in _semaphores:
        _semaphores[loop] = asyncio.Semaphore(config.get("num_threads"))
    return _semaphores[loop]


async def async_run(func, *args, **kwargs):
    """Run a function in the thread pool executor with semaphore limiting."""
    loop = asyncio.get_running_loop()
    semaphore = _get_semaphore(loop)
    async with semaphore:
        return await loop.run_in_executor(EXECUTOR, func, *args, **kwargs)


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


def is_4326_like(crs: CRS) -> bool:
    return crs == pyproj.CRS.from_epsg(4326) or crs == OTHER_4326


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


def transform_blocked(
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
    wait(
        [
            EXECUTOR.submit(
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


def transform_coordinates(
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
    if is_4326_like(transformer.source_crs) and is_4326_like(transformer.target_crs):
        # for some reason pyproj does not normalize these to -180->180
        newdata = inx.data.copy()
        newdata[newdata >= 180] -= 360
        return inx.copy(data=newdata), iny

    if transformer.source_crs == transformer.target_crs:
        return inx, iny

    # preserve rectilinear-ness by reimplementing this (easy) transform
    if (inx.ndim == 1 and iny.ndim == 1) and (
        transformer == transformer_from_crs(4326, 3857)
        or transformer == transformer_from_crs(OTHER_4326, 3857)
    ):
        newx, newy = epsg4326to3857(inx.data, iny.data)
        return inx.copy(data=newx), iny.copy(data=newy)

    # Broadcast coordinates
    # FIXME: dropping indexes is a workaround for broadcasting RasterIndex
    bx, by = xr.broadcast(
        inx.drop_indexes(inx.dims, errors="ignore"),
        iny.drop_indexes(iny.dims, errors="ignore"),
    )
    assert bx.dims == by.dims

    chunk_size = get_transform_chunk_size(bx)
    # Choose transformation method based on data size
    if bx.size > math.prod(chunk_size):
        newX, newY = transform_blocked(
            bx.data.copy(order="C"),
            by.data.copy(order="C"),
            transformer,
            chunk_size,
            True,  # inplace
        )
    else:
        newX, newY = transformer.transform(bx.data, by.data)

    return bx.copy(data=newX), by.copy(data=newY)


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
    slicers: "dict[str, list[slice | Fill | UgridIndexer]]",
    *,
    dimensions: list[PadDimension] | None = None,
) -> "dict[str, list[slice | Fill | UgridIndexer]]":
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

        dim_slicers = slicers[dim.name]
        # Convert to proper slice objects with dimension size
        indexers = [slice(*idxr.indices(dim.size)) for idxr in dim_slicers]  # type: ignore[misc]

        # Prevent overlap if requested (before padding)
        if dim.prevent_overlap:
            indexers = _prevent_slice_overlap(indexers)

        # Apply padding
        first, last = indexers[0], indexers[-1]
        left_edge = first.start - dim.left_pad
        right_edge = last.stop + dim.right_pad

        indexers_with_fill: list[slice | Fill]
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
            if indexers_with_fill[0].start == 0:
                # Starts at beginning, add wraparound from end
                indexers_with_fill = [slice(-dim.left_pad, None), *indexers_with_fill]
            if indexers_with_fill[-1].stop >= dim.size - 1:
                # Ends at end, add wraparound from beginning
                indexers_with_fill = indexers_with_fill + [slice(0, dim.right_pad)]
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
