import contextlib
import functools
import importlib.util
import threading
import time
from typing import Any

from xpublish_tiles.logger import log_duration, logger

# Only use lock if tbb is not available
HAS_TBB = importlib.util.find_spec("tbb") is not None
NUMBA_THREADING_LOCK = contextlib.nullcontext() if HAS_TBB else threading.Lock()


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
