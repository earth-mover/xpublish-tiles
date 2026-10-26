"""Exact data extraction: the non-rendering counterpart of ``pipeline``."""

import asyncio
import math
from collections.abc import Mapping
from functools import partial
from typing import Any

import numpy as np
from pyproj import CRS

import xarray as xr
from xpublish_tiles.config import config
from xpublish_tiles.grids import GridSystem, PointIndexers
from xpublish_tiles.lib import (
    AsyncLoadTimeoutError,
    RequestTooBigError,
    async_run,
    get_data_load_semaphore,
)
from xpublish_tiles.logger import log_duration
from xpublish_tiles.pipeline import apply_query
from xpublish_tiles.projections import transformer_from_crs
from xpublish_tiles.types import ValidatedArray
from xpublish_tiles.utils import cf_ref_attr

POINT_DIM = "point"


def locate_points(
    array: ValidatedArray, *, crs: CRS, x: np.ndarray, y: np.ndarray
) -> PointIndexers:
    """Locate the points ``(x, y)``, given in ``crs``, on the array's grid."""
    grid = array.grid
    gx, gy = transformer_from_crs(crs_from=crs, crs_to=grid.crs).transform(x, y)
    return grid.locate(np.asarray(gx), np.asarray(gy))


def isel_points(array: ValidatedArray, indexers: PointIndexers) -> xr.DataArray:
    """Lazily pick the located cells along a new ``point`` dim.

    Points outside the grid are dropped; the ``point`` coordinate holds each kept
    point's position in the request.
    """
    (inside,) = np.nonzero(next(iter(indexers.values())) >= 0)
    da = array.da.isel(
        {dim: xr.Variable(POINT_DIM, idx[inside]) for dim, idx in indexers.items()}
    ).assign_coords({POINT_DIM: inside})
    return da.drop_vars(
        [
            name
            for name, coord in da.coords.items()
            if _is_structural(da, coord, array.grid)
        ]
    )


def _chunk_span(size: int, chunk: int) -> int:
    """Upper bound on the elements decompressed to read ``size`` contiguous
    elements in ``chunk``-sized chunks, wherever the run starts."""
    return (-(-(size - 1) // chunk) + 1) * chunk


def decompressed_bytes(array: ValidatedArray, indexers: PointIndexers) -> int:
    """Upper bound on the bytes decompressed to read the located cells.

    Every distinct chunk the points fall in is read whole, along with the full
    chunk extent of every other dim, including dims a selector already dropped
    (see ``decompressed_size_bytes``). Without chunk metadata, only the selected
    elements count.
    """
    da, chunks = array.da, array.chunks
    (inside,) = np.nonzero(next(iter(indexers.values())) >= 0)
    point_chunks = [chunks.get(dim, 1) for dim in indexers]
    chunk_ids = np.stack(
        [idx[inside] // c for idx, c in zip(indexers.values(), point_chunks, strict=True)]
    )
    total = np.unique(chunk_ids, axis=1).shape[1] * math.prod(point_chunks)
    for dim, size in da.sizes.items():
        if (name := str(dim)) not in indexers:
            total *= _chunk_span(size, chunks[name]) if name in chunks else size
    for dim, chunk in chunks.items():
        if dim not in da.dims:
            total *= chunk
    return total * da.dtype.itemsize


def _is_structural(da: xr.DataArray, coord: xr.DataArray, grid: GridSystem) -> bool:
    """Grid machinery that says nothing about a point: the grid mapping, encoded
    geometries, and X/Y unless they hold per-point cell centres. Z always stays."""
    if coord.name == cf_ref_attr(da, "grid_mapping"):
        return True
    if coord.name == grid.Z:
        return False
    if coord.name in (grid.X, grid.Y):
        return POINT_DIM not in coord.dims
    return grid.is_structural_variable(coord)


async def extract_points(
    ds: xr.Dataset,
    *,
    variables: list[str],
    selectors: Mapping[str, Any],
    crs: CRS,
    x: np.ndarray,
    y: np.ndarray,
    squeeze: bool = True,
    allow_ranges: bool = False,
) -> xr.Dataset:
    """Load ``variables`` at the points ``(x, y)``, given in ``crs``.

    ``squeeze`` and ``allow_ranges`` control the non-spatial dims, as in
    ``apply_query``.
    """
    validated = await async_run(
        partial(
            apply_query,
            ds,
            variables=variables,
            selectors=dict(selectors),
            squeeze=squeeze,
            allow_ranges=allow_ranges,
        )
    )
    arrays = []
    total_bytes = 0
    for array in validated.values():
        indexers = locate_points(array, crs=crs, x=x, y=y)
        arrays.append(isel_points(array, indexers))
        total_bytes += decompressed_bytes(array, indexers)
    max_bytes = config.get("max_renderable_size")
    if total_bytes > max_bytes:
        raise RequestTooBigError(
            f"Request needs {total_bytes / 1024**2:.0f}MB of data, more than the "
            f"{max_bytes / 1024**2:.0f}MB limit. Request a shorter range or fewer points."
        )
    timeout = config.get("async_load_timeout_per_tile")
    async with get_data_load_semaphore():
        try:
            with log_duration("load points", key="load"):
                async with asyncio.timeout(timeout):
                    loaded = await asyncio.gather(*(a.load_async() for a in arrays))
        except TimeoutError:
            raise AsyncLoadTimeoutError(
                f"Async data loading timed out after {timeout}s. Server may be overloaded."
            ) from None
    return xr.merge(loaded, join="outer")
