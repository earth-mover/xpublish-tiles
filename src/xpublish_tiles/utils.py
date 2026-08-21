import contextlib
import functools
import importlib.util
import threading
import time
from typing import Any

import cf_xarray  # noqa: F401
import numpy as np

import xarray as xr
from xpublish_tiles.logger import log_duration, logger


def _has_threadsafe_numba_layer() -> bool:
    """Whether numba will select a threading layer that is safe for concurrent
    entry into parallel regions from multiple Python threads.

    tbb and omp are threadsafe; the workqueue fallback is not, so without one
    of the safe layers we serialize numba sections with a lock instead.
    https://numba.readthedocs.io/en/stable/user/threading-layer.html
    """
    from numba import config as numba_config

    forced = getattr(numba_config, "THREADING_LAYER", "default")
    if forced in ("tbb", "omp", "safe", "threadsafe"):
        return True
    if forced != "default":
        return False
    if importlib.util.find_spec("tbb") is not None:
        return True
    try:
        # must actually import (not find_spec): on macOS the omppool extension
        # exists but fails to dlopen because the numba wheel lacks a usable
        # libomp rpath (https://github.com/numba/numba/issues/10492)
        importlib.import_module("numba.np.ufunc.omppool")
    except ImportError:
        return False
    return True


# Our own parallel=True kernels are only worth it when they need no lock to run:
# paying a process-wide lock to get ~4x on one array is a bad trade for a server
# that is already parallel across requests.
NUMBA_PARALLEL = _has_threadsafe_numba_layer()
NUMBA_THREADING_LOCK = contextlib.nullcontext() if NUMBA_PARALLEL else threading.Lock()


def cf_ref_attr(var: xr.DataArray, name: str) -> Any | None:
    """Return a CF reference attribute like ``node_coordinates``.

    xarray's ``decode_coords="all"`` relocates CF reference attributes from
    ``.attrs`` to ``.encoding`` when it promotes the referenced variables to
    coordinates, so check both.
    """
    value = var.attrs.get(name)
    if value is None:
        value = var.encoding.get(name)
    return value


def xarray_object_key(
    obj: xr.DataArray | xr.Dataset,
    *,
    cf_coords: dict | None = None,
) -> tuple:
    """Cache key fragment: sorted dims whose size > 1, excluding the time dim
    when the time coordinate is 1D.

    ``cf_coords`` may be the parent dataset's ``ds.cf.coordinates`` mapping;
    passing it avoids an expensive per-call ``obj.cf.coordinates`` lookup on
    the hot path.
    """
    coords = cf_coords if cf_coords is not None else obj.cf.coordinates
    time_dims: set = set()
    for name in coords.get("time", []):
        if name in obj.coords and obj.coords[name].ndim == 1:
            time_dims.add(obj.coords[name].dims[0])
    return tuple(
        sorted(
            dim for dim, size in obj.sizes.items() if size > 1 and dim not in time_dims
        )
    )


def lower_case_keys(d: Any) -> dict[str, Any]:
    """Convert keys to lowercase, handling both dict and QueryParams objects"""
    if hasattr(d, "items"):
        return {k.lower(): v for k, v in d.items()}
    else:
        # Handle other dict-like objects
        return {k.lower(): v for k, v in dict(d).items()}


def time_debug(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound_logger = kwargs.get("bound_logger")
        with log_duration(func.__name__, emoji="⏱️", logger=bound_logger):
            return func(*args, **kwargs)

    return wrapper


def async_time_debug(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        bound_logger = kwargs.get("bound_logger")
        with log_duration(func.__name__, emoji="⏱️", logger=bound_logger):
            return await func(*args, **kwargs)

    return wrapper


def normalize_longitude_deg(lon: float) -> float:
    """Normalize a longitude value to the [-180, 180] range.

    Examples:
    - 190 -> -170
    - 360 -> 0
    - -190 -> 170
    - 180, -180 remain unchanged
    """
    # Use modulo arithmetic to wrap, then shift into [-180, 180]
    return ((float(lon) + 180.0) % 360.0) - 180.0


def normalize_tilejson_bounds(
    bounds: list[float] | tuple[float, float, float, float],
) -> list[float]:
    """Normalize a TileJSON bounds array to use [-180, 180] longitudes.

    Input: [west, south, east, north] possibly with 0..360 longitudes.
    Output: [west, south, east, north] with longitudes in [-180, 180].

    Special cases:
    - If the span is ~360° (full world), return [-180, 180]
    - If normalization yields west > east (dateline crossing), return [-180, 180]
    """
    west0, south, east0, north = bounds  # type: ignore[misc]

    # Full-world coverage in 0..360 representation
    if (float(east0) - float(west0)) >= 360.0 - 1e-6:
        return [-180.0, float(south), 180.0, float(north)]

    # Explicit 0..360 dateline-crossing case (east0 < west0)
    if float(east0) < float(west0):
        return [-180.0, float(south), 180.0, float(north)]

    w = normalize_longitude_deg(west0)
    e = normalize_longitude_deg(east0)

    if w > e:
        # Dateline-crossing case cannot be represented as a single [w,e] in TileJSON
        # Use full extent to signal global coverage
        w, e = -180.0, 180.0

    return [w, float(south), e, float(north)]


def format_number_for_url(value: float) -> str:
    """Format a number for inclusion in a URL query string.

    ``%g`` is unusable here: it emits scientific notation whose ``+`` exponent
    sign is decoded as a space by query string parsers (``1e+06`` -> ``1e 06``),
    and it silently rounds to 6 significant digits.

    Examples
    --------
    >>> [format_number_for_url(v) for v in (0.0, -3.0, 1.5)]
    ['0', '-3', '1.5']
    >>> [format_number_for_url(v) for v in (1e6, -1e6, 1234567.5, 1e21, 1e-7)]
    ['1000000', '-1000000', '1234567.5', '1000000000000000000000', '0.0000001']
    >>> [format_number_for_url(v) for v in (float("nan"), float("inf"))]
    ['nan', 'inf']
    """
    if not np.isfinite(value):
        return str(float(value))
    return np.format_float_positional(value, trim="-")


@contextlib.contextmanager
def time_operation(message: str = "Operation"):
    """Context manager for timing operations with custom messages."""
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    perf_time = (end_time - start_time) * 1000
    logger.debug(f"{message}: {perf_time:.2f} ms")


@contextlib.asynccontextmanager
async def async_time_operation(message: str = "Async Operation"):
    """Async context manager for timing operations with custom messages."""
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    perf_time = (end_time - start_time) * 1000
    logger.debug(f"{message}: {perf_time:.2f} ms")
