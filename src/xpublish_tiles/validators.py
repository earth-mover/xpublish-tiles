from typing import cast

from pyproj.aoi import BBox

from xpublish_tiles.types import ImageFormat, Style


def validate_colorscalerange(v: str | None) -> tuple[float, float] | None:
    if v is None:
        return None

    values = v.split(",")
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


def validate_style(v: str | None) -> tuple[Style, str] | None:
    if v is None:
        return None

    values = v.split("/")
    if len(values) != 2:
        raise ValueError(
            "style must be in the format 'stylename/palettename'. A common default for this is 'raster/default'",
        )

    try:
        style = Style(values[0].lower())
    except ValueError as e:
        raise ValueError(
            f"style {values[0]} is not valid. Options are: {', '.join(Style.__members__.keys())}",
        ) from e

    return style, values[1]


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
