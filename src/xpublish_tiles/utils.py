import contextlib
import functools
import importlib.util
import threading
import time
from typing import Any

from xpublish_tiles.logger import logger

# Only use lock if tbb is not available
HAS_TBB = importlib.util.find_spec("tbb") is not None
LOCK = contextlib.nullcontext() if HAS_TBB else threading.Lock()


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
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        perf_time = (end_time - start_time) * 1000
        logger.debug(f"{func.__name__}: {perf_time} ms")
        return result

    return wrapper


def async_time_debug(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        perf_time = (end_time - start_time) * 1000
        logger.debug(f"{func.__name__}: {perf_time} ms")
        return result

    return wrapper


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
