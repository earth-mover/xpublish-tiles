"""Optional Datadog tracing; every helper is a no-op without the `datadog` extra."""

try:
    from ddtrace.trace import tracer  # ty: ignore[unresolved-import]
except ImportError:
    tracer = None


def root_span():
    """Root span of the current trace, or None without ddtrace or an active trace."""
    return tracer.current_root_span() if tracer is not None else None
