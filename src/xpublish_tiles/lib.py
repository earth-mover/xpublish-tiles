"""Library utility functions for xpublish-tiles."""

import io
import operator
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import pyproj
import toolz as tlz
from PIL import Image


class NoCoverageError(Exception):
    """Raised when a tile has no overlap with the dataset bounds."""

    pass


EXECUTOR = ThreadPoolExecutor(
    max_workers=16, thread_name_prefix="xpublish-tiles-threadpool"
)


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
