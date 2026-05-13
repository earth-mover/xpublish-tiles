import json
from typing import cast

from pyproj import CRS
from pyproj.aoi import BBox
from pyproj.exceptions import CRSError

from xpublish_tiles.types import ImageFormat


def validate_colorscalerange(v: str | list[str] | None) -> tuple[float, float] | None:
    if v is None:
        return None
    elif not isinstance(v, str):
        if len(v) == 0:
            raise ValueError("colorscalerange must be a non-empty list")
        v = v[0]

    try:
        values = v.split(",")
    except AttributeError as e:
        raise ValueError(
            "colorscalerange must be a string or a list of strings delimited by commas"
        ) from e

    if len(values) != 2:
        raise ValueError("colorscalerange must be in the format 'min,max'")

    try:
        min_val = float(values[0])
        max_val = float(values[1])
    except ValueError as e:
        raise ValueError(
            "colorscalerange must be in the format 'min,max' where min and max are valid floats",
        ) from e
    return (min_val, max_val)


def validate_levels(v: str | list[str] | None) -> tuple[float, ...] | None:
    """Parse comma-separated float levels for vector/contours.

    Returns a strictly-increasing tuple, or None if no levels were given.
    Two or more levels are required (one band needs two boundaries).
    """
    if v is None:
        return None
    if not isinstance(v, str):
        if len(v) == 0:
            return None
        v = v[0]
    v = v.strip()
    if not v:
        return None
    try:
        parsed = tuple(float(x) for x in v.split(","))
    except ValueError as e:
        raise ValueError(
            "levels must be a comma-separated list of floats, e.g. '0,5,10,15'",
        ) from e
    if len(parsed) < 2:
        raise ValueError("levels must contain at least 2 values to define a band")
    if any(parsed[i + 1] <= parsed[i] for i in range(len(parsed) - 1)):
        raise ValueError("levels must be strictly increasing")
    return parsed


def validate_smoothing(v: str | list[str] | None) -> float | None:
    """Parse vector/contours pre-blur sigma (in grid cells)."""
    if v is None:
        return None
    if not isinstance(v, str):
        if len(v) == 0:
            return None
        v = v[0]
    v = v.strip()
    if not v:
        return None
    try:
        parsed = float(v)
    except ValueError as e:
        raise ValueError("smoothing must be a non-negative float") from e
    if parsed < 0:
        raise ValueError("smoothing must be a non-negative float")
    return parsed


def validate_bbox(v: str | None) -> BBox | None:
    if v is None:
        return None

    values = v.split(",") if isinstance(v, str) else v
    if len(values) != 4:
        raise ValueError("bbox must be in the format 'minx,miny,maxx,maxy'")

    try:
        bbox = cast(tuple[float, float, float, float], tuple(float(x) for x in values))
    except ValueError as e:
        raise ValueError(
            "bbox must be in the format 'minx,miny,maxx,maxy' where minx, miny, maxx and maxy are valid floats in the provided CRS",
        ) from e

    return BBox(*bbox)


def validate_style(v: str | list[str] | None) -> tuple[str, str] | None:
    if v is None:
        return None
    elif not isinstance(v, str):
        if len(v):
            v = v[0]
        else:
            raise ValueError(
                "style must be in the format 'stylename/palettename'. A common default for this is 'raster/default'"
            )

    # An empty string is valid, but not None
    if not v:
        return None

    values = v.split("/")
    if len(values) != 2:
        raise ValueError(
            "style must be in the format 'stylename/palettename'. A common default for this is 'raster/default'",
        )

    style_name = values[0].lower()
    variant = values[1]

    # Validate that the style is registered
    from xpublish_tiles.render import RenderRegistry

    try:
        renderer_cls = RenderRegistry.get(style_name)
    except ValueError as e:
        available_styles = list(RenderRegistry.all().keys())
        raise ValueError(
            f"style '{style_name}' is not valid. Available styles are: {', '.join(available_styles)}",
        ) from e

    # Validate that the variant is supported (or is "default")
    if variant != "default":
        supported_variants = renderer_cls.supported_variants()
        if variant not in supported_variants:
            raise ValueError(
                f"variant '{variant}' is not supported for style '{style_name}'. "
                f"Supported variants are: {', '.join(['default'] + supported_variants)}",
            )

    return style_name, variant


def validate_image_format(v: str | None) -> ImageFormat | None:
    if v is None:
        return None
    try:
        return ImageFormat(v.lower().strip())
    except ValueError as e:
        raise ValueError(
            f"format {v!r} is not valid. Options are: {', '.join(ImageFormat.__members__.keys())}",
        ) from e


def validate_legend_format(v: str | None):
    """Accept image/png, image/jpeg, application/json, or shorthand."""
    from xpublish_tiles.types import LegendFormat

    if v is None:
        return None
    if "/" in v:
        _, format_str = v.split("/", 1)
    else:
        format_str = v
    try:
        return LegendFormat(format_str.lower())
    except ValueError as e:
        raise ValueError(
            f"legend format {format_str} is not valid. Options are: {', '.join(LegendFormat.__members__.keys())}",
        ) from e


def validate_crs(v: str | None) -> CRS | None:
    if v is None:
        return None
    try:
        return CRS.from_user_input(v)
    except CRSError as e:
        raise ValueError(
            f"crs {v} is not valid",
        ) from e


def validate_range_color(v: str | None) -> str | None:
    """Validate abovemaxcolor or belowmincolor parameter.

    Accepted values:
    - None: Return None (use default behavior)
    - "extend": Use the max/min palette color (current default behavior)
    - "transparent": Fully transparent
    - Hex color (#RRGGBB or #RRGGBBAA)
    - Named color (any matplotlib-recognized color name)

    Returns:
        Normalized lowercase string or None

    Raises:
        ValueError: If the color value is invalid
    """
    if v is None:
        return None

    v_lower = v.lower().strip()

    if v_lower in ("extend", "transparent"):
        return v_lower

    # Try to validate as a color using matplotlib
    from matplotlib.colors import to_rgba

    try:
        to_rgba(v_lower)
        return v_lower
    except ValueError:
        raise ValueError(
            f"Invalid color value '{v}'. Must be 'extend', 'transparent', "
            "a hex color (#RRGGBB or #RRGGBBAA), or a valid named color."
        ) from None


def validate_color(v: str | None) -> str | None:
    """Validate a generic color string (hex, named, or 'transparent').

    Unlike ``validate_range_color`` this does not accept the sentinel ``"extend"``.
    """
    if v is None:
        return None

    v_lower = v.lower().strip()
    if v_lower == "transparent":
        return v_lower

    from matplotlib.colors import to_rgba

    try:
        to_rgba(v_lower)
        return v_lower
    except ValueError:
        raise ValueError(
            f"Invalid color value '{v}'. Must be 'transparent', "
            "a hex color (#RRGGBB or #RRGGBBAA), or a valid named color."
        ) from None


def validate_colormap(v: str | dict | None) -> dict[str, str] | None:
    """Validate and parse custom colormap parameter.

    Args:
        v: Colormap input - can be None, a JSON string, or a dict

    Returns:
        Parsed colormap dict with string keys (0-255) and hex color values (#RRGGBB) or None

    Raises:
        ValueError: If colormap format is invalid
    """
    if v is None:
        return None

    # If it's already a dict, validate it
    if isinstance(v, dict):
        colormap = v
    elif isinstance(v, str):
        # Try to parse as JSON string
        try:
            colormap = json.loads(v)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError("colormap must be a valid JSON-encoded dictionary") from e
    else:
        raise ValueError("colormap must be a dictionary or JSON string")

    if not isinstance(colormap, dict):
        raise ValueError("colormap must be a dictionary")

    # Convert all keys to strings and validate format
    validated_colormap = {}
    minkey, maxkey = 256, -1
    for key, value in colormap.items():
        # Convert numeric keys to strings
        str_key = str(key)

        # Validate key is numeric (0-255)
        try:
            numeric_key = int(str_key)
            if not 0 <= numeric_key <= 255:
                raise ValueError(
                    f"colormap keys must be integers between 0 and 255, got {numeric_key}"
                )
            minkey = min(minkey, numeric_key)
            maxkey = max(maxkey, numeric_key)
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"colormap keys must be numeric, got '{key}'") from e
            raise

        # Validate value is a valid color (hex or rgb)
        if not isinstance(value, str):
            raise ValueError(
                f"colormap values must be strings, got {type(value).__name__} for key {key}"
            )

        # Validation for hex colors only (#RRGGBB)
        value = value.strip()
        if not (value.startswith("#") and len(value) == 7):
            raise ValueError(
                f"colormap value '{value}' for key {key} must be a hex color (#RRGGBB)"
            )

        validated_colormap[str_key] = value

    if minkey != 0 or maxkey != 255:
        raise ValueError(
            "colormap keys must include 0 and 255 as minimum and maximum."
            f"Detected minimum={minkey!r} and maximum={maxkey!r}"
        )

    return validated_colormap
