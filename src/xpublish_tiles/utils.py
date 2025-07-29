from typing import Any


def lower_case_keys(d: Any) -> dict[str, Any]:
    """Convert keys to lowercase, handling both dict and QueryParams objects"""
    if hasattr(d, "items"):
        return {k.lower(): v for k, v in d.items()}
    else:
        # Handle other dict-like objects
        return {k.lower(): v for k, v in dict(d).items()}


def parse_colorscalerange(raw_value: str) -> tuple[float, float]:
    """Unpack a color scale range string into a tuple of floats"""
    try:
        min_val, max_val = map(float, raw_value.split(","))
        return min_val, max_val
    except ValueError as e:
        raise ValueError(f"Invalid color scale range format: {raw_value}") from e
