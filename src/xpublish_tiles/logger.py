"""
Logging setup and shared logger for the xpublish_tiles package.
"""

import contextlib
import contextvars
import logging
import time

import structlog


class LogAccumulator:
    """Accumulates log entries for later processing without outputting them."""

    def __init__(self):
        self.logs = []

    def __call__(self, logger, method_name, event_dict):
        # Store the log entry
        self.logs.append(event_dict)
        # Suppress output by raising DropEvent
        raise structlog.DropEvent


# Context variable to hold the current bound logger
_context_logger: contextvars.ContextVar[structlog.stdlib.BoundLogger | None] = (
    contextvars.ContextVar("context_logger")
)


# Set up a shared logger for the xpublish_tiles package (for backward compatibility)
logger = logging.getLogger("xpublish_tiles")

# Configure structlog to integrate with standard logging
timestamper = structlog.processors.TimeStamper(fmt="iso")
shared_processors = [
    # Removed add_log_level and add_logger_name to clean up output
    timestamper,
]

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,  # Respects standard logging level
        *shared_processors,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


# Create a custom console renderer that only outputs what we want
class CleanConsoleRenderer:
    """Console renderer that only outputs the event message and context fields."""

    def __call__(self, logger, name, event_dict):
        # Extract the main event message
        event = event_dict.get("event", "")

        # Get the timestamp
        timestamp = event_dict.get("timestamp", "")
        timestamp_str = f"   {timestamp} " if timestamp else ""

        # Create context fields (excluding standard structlog fields)
        context_parts = []
        for key, value in event_dict.items():
            if key not in ("event", "timestamp", "level", "logger", "exc_info"):
                context_parts.append(f"{key}={value}")

        context_str = " ".join(context_parts)
        if context_str:
            context_str = " " + context_str

        return f"{timestamp_str}{event}{context_str}"


# Create a formatter for structlog that works with standard logging
formatter = structlog.stdlib.ProcessorFormatter(
    processor=CleanConsoleRenderer(),
    foreign_pre_chain=shared_processors,
)


def setup_logging(log_level=logging.INFO):
    """Set up logging with structlog integration."""
    # Create a silent handler that does nothing
    silent_handler = logging.NullHandler()

    # Configure the xpublish_tiles logger to be silent by default
    logger.setLevel(log_level)
    logger.handlers = []
    logger.addHandler(silent_handler)

    # Configure root logger to be silent by default
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = []
    root_logger.addHandler(silent_handler)

    # Silence other noisy loggers
    for logger_name in [
        "matplotlib",
        "numba",
        "datashader",
        "asyncio",
        "PIL",
        "numcodecs",
        "zarr",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


# Create structlog logger factory
def get_logger(name=None):
    """Get a structlog logger bound to the given name."""
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger("xpublish_tiles")


def get_context_logger():
    """Get the current context logger, or create a default one if none is set."""
    context_logger = _context_logger.get(None)
    if context_logger is not None:
        return context_logger
    # Return a default logger if no context is set
    return structlog.get_logger("xpublish_tiles")


def set_context_logger(bound_logger: structlog.stdlib.BoundLogger):
    """Set the context logger for the current context."""
    _context_logger.set(bound_logger)


@contextlib.contextmanager
def log_duration(message: str, emoji: str = "⏱️", logger=None):
    """
    Context manager to log the duration of a code block with a custom message.

    Args:
        message: Custom message to log with timing
        emoji: Emoji to prefix the message (optional)

    Usage:
        with log_duration("loading data", "📥"):
            result = await load_data()
    """
    if logger is None:
        logger = get_context_logger()
    start_time = time.perf_counter()
    try:
        yield
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(f"{emoji} ({duration_ms:.0f}ms) {message}")
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"{emoji} ({duration_ms:.0f}ms) {message} (failed)", error=str(e))
        raise
