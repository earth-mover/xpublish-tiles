"""Library utility functions for xpublish-tiles."""

import asyncio
import io
import logging
import math
import operator
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import pyproj
import toolz as tlz
from PIL import Image

import xarray as xr


class NoCoverageError(Exception):
    """Raised when a tile has no overlap with the dataset bounds."""

    pass


class TileTooBigError(Exception):
    """Raised when a tile request would result in too much data to render."""

    pass


logger = logging.getLogger(__name__)

EXECUTOR = ThreadPoolExecutor(
    max_workers=16, thread_name_prefix="xpublish-tiles-threadpool"
)

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
CHUNKED_TRANSFORM_CHUNK_SIZE = (250, 250)


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
) -> None:
    """Transform a chunk of coordinates."""
    row_slice, col_slice = slices
    x_chunk = x_grid[row_slice, col_slice]
    y_chunk = y_grid[row_slice, col_slice]
    x_transformed, y_transformed = transformer.transform(x_chunk, y_chunk)
    x_out[row_slice, col_slice] = x_transformed
    y_out[row_slice, col_slice] = y_transformed


def transform_blocked(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    transformer: pyproj.Transformer,
    chunk_size: tuple[int, int] = (250, 250),
) -> tuple[np.ndarray, np.ndarray]:
    """Blocked transformation using thread pool."""

    if transformer.source_crs == transformer.target_crs:
        return x_grid, y_grid

    start_time = time.perf_counter()

    shape = x_grid.shape
    x_out = np.empty(shape, dtype=x_grid.dtype)
    y_out = np.empty(shape, dtype=y_grid.dtype)

    chunk_rows, chunk_cols = chunk_size

    # Generate chunks for each dimension
    row_chunks = [min(chunk_rows, shape[0] - i) for i in range(0, shape[0], chunk_rows)]
    col_chunks = [min(chunk_cols, shape[1] - j) for j in range(0, shape[1], chunk_cols)]

    chunks = (row_chunks, col_chunks)

    # Use slices_from_chunks to generate slices lazily
    futures = [
        EXECUTOR.submit(
            transform_chunk, x_grid, y_grid, slices, transformer, x_out, y_out
        )
        for slices in slices_from_chunks(chunks)
    ]

    for future in futures:
        future.result()

    total_time = time.perf_counter() - start_time
    logger.info(
        "transform_blocked completed in %.4f seconds (shape=%s, chunks=%d)",
        total_time,
        shape,
        len(futures),
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
    chunk_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, xr.DataArray, xr.DataArray]:
    """
    Transform coordinates from input CRS to output CRS.

    This function broadcasts the X and Y coordinates and then transforms them
    using either chunked or direct transformation based on the data size.

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
        Chunk size for blocked transformation, by default CHUNKED_TRANSFORM_CHUNK_SIZE

    Returns
    -------
    tuple[np.ndarray, np.ndarray, xr.DataArray, xr.DataArray]
        Transformed X and Y coordinates, and the broadcasted coordinate arrays
    """
    if chunk_size is None:
        chunk_size = CHUNKED_TRANSFORM_CHUNK_SIZE

    # Broadcast coordinates
    bx, by = xr.broadcast(subset[grid_x_name], subset[grid_y_name])

    # Choose transformation method based on data size
    if bx.size > math.prod(chunk_size):
        loop = asyncio.get_event_loop()
        newX, newY = await loop.run_in_executor(
            EXECUTOR,
            transform_blocked,
            bx.data,
            by.data,
            transformer,
            chunk_size,
        )
    else:
        newX, newY = transformer.transform(bx.data, by.data)

    return newX, newY, bx, by
