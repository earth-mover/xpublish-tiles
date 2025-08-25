import functools
import time
from typing import Any

from xpublish_tiles.logger import logger


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
