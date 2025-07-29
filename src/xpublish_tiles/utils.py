from typing import Any

from xpublish_tiles.types import ImageFormat, Style


def lower_case_keys(d: Any) -> dict[str, Any]:
    """Convert keys to lowercase, handling both dict and QueryParams objects"""
    if hasattr(d, "items"):
        return {k.lower(): v for k, v in d.items()}
    else:
        # Handle other dict-like objects
        return {k.lower(): v for k, v in dict(d).items()}


def parse_colorscalerange(raw_value: str) -> tuple[float, float]:
    """Unpack a color scale range string from "min,max" into a tuple of floats"""
    try:
        min_val, max_val = map(float, raw_value.split(","))
        return min_val, max_val
    except ValueError as e:
        raise ValueError(f"Invalid color scale range format: {raw_value}") from e


def parse_style(raw_value: str) -> tuple[Style, str]:
    """Parse a style string from "style/colormap" into a tuple of Style and colormap"""
    try:
        style, value = raw_value.split("/", 1)
        return Style[style.upper()], value
    except ValueError as e:
        raise ValueError(f"Invalid style format: {raw_value}") from e
    except KeyError as e:
        raise ValueError(f"Invalid style format: {raw_value}") from e


def parse_image_format(raw_value: str) -> ImageFormat:
    """Parse an image format string from "format" or "image/format" into a string"""
    try:
        if "/" in raw_value:
            _, format_str = raw_value.split("/", 1)
        else:
            format_str = raw_value
        return ImageFormat(format_str.lower())
    except ValueError as e:
        raise ValueError(f"Invalid image format: {raw_value}") from e
    except KeyError as e:
        raise ValueError(f"Invalid image format: {raw_value}") from e
