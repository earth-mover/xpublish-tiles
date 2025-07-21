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
        if "/" in v:
            _, format_str = v.split("/", 1)
        else:
            format_str = v
        return ImageFormat(format_str.lower())
    except ValueError as e:
        raise ValueError(
            f"image format {format_str} is not valid. Options are: {', '.join(ImageFormat.__members__.keys())}",
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
