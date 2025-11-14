"""Shared test utilities."""

from pyproj import CRS
from pyproj.aoi import BBox

from xpublish_tiles.types import ImageFormat, OutputBBox, OutputCRS, QueryParams


def create_query_params(
    tile,
    tms,
    *,
    colorscalerange=None,
    size=256,
    style="raster",
    variant="viridis",
    colormap=None,
):
    """Create QueryParams instance using test tiles and TMS."""
    epsg_code = tms.crs.to_epsg()
    if epsg_code is None:
        target_crs = CRS.from_user_input(tms.crs)
    else:
        target_crs = CRS.from_epsg(epsg_code)

    native_bounds = tms.xy_bounds(tile)
    bbox = BBox(
        west=native_bounds[0],
        south=native_bounds[1],
        east=native_bounds[2],
        north=native_bounds[3],
    )

    return QueryParams(
        variables=["foo"],
        crs=OutputCRS(target_crs),
        bbox=OutputBBox(bbox),
        selectors={},
        style=style,
        width=size,
        height=size,
        variant=variant,
        colorscalerange=colorscalerange,
        colormap=colormap,
        format=ImageFormat.PNG,
    )
