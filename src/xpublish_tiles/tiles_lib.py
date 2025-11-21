"""Tile-related utility functions for grids."""

import morecantile
import numpy as np
from pyproj import CRS
from pyproj.aoi import BBox

import xarray as xr
from xpublish_tiles.grids import GridSystem, Triangular
from xpublish_tiles.lib import transformer_from_crs
from xpublish_tiles.utils import time_debug


@time_debug
def get_max_zoom(grid: GridSystem, tms: morecantile.TileMatrixSet) -> int:
    """Calculate maximum zoom level based on grid spacing and TMS.

    Takes the lower left corner of the grid bounding box, adds the minimum
    grid spacing (dXmin, dYmin), transforms the resulting box to the TMS CRS,
    and calculates the appropriate zoom level using tms.zoom_for_res().

    Parameters
    ----------
    grid : Grid
        The grid to calculate zoom for
    tms : morecantile.TileMatrixSet
        The tile matrix set to calculate zoom for

    Returns
    -------
    int
        Maximum appropriate zoom level for this grid
    """
    if isinstance(grid, Triangular):
        # no dXmin, dYmin defined, punt for now
        return tms.maxzoom
    ll_box = BBox(
        west=grid.bbox.west,
        south=grid.bbox.south,
        east=grid.bbox.west + grid.dXmin,
        north=grid.bbox.south + grid.dYmin,
    )

    tms_crs = CRS.from_wkt(tms.crs.to_wkt())
    transformer = transformer_from_crs(grid.crs, tms_crs)

    west_coords = [ll_box.west, ll_box.east, ll_box.west, ll_box.east]
    south_coords = [ll_box.south, ll_box.south, ll_box.north, ll_box.north]

    x_transformed, y_transformed = transformer.transform(west_coords, south_coords)
    dx_transformed = np.max(x_transformed) - np.min(x_transformed)
    dy_transformed = np.max(y_transformed) - np.min(y_transformed)

    min_spacing = min(dx_transformed, dy_transformed)
    zoom = tms.zoom_for_res(min_spacing, zoom_level_strategy="upper")
    return zoom


@time_debug
def get_min_zoom(
    grid: GridSystem, tms: morecantile.TileMatrixSet, da: xr.DataArray
) -> int:
    """Calculate minimum zoom level that avoids TileTooBigError.

    This method finds the zoom level below which no tile would trigger
    the TileTooBigError check in apply_slicers.

    Parameters
    ----------
    grid : Grid
        The grid to calculate zoom for
    tms : morecantile.TileMatrixSet
        The tile matrix set to calculate zoom for
    da : xr.DataArray
        Data array (only metadata used, no data loaded).
        Required since we use `Grid.sel`.

    Returns
    -------
    int
        Minimum safe zoom level for this grid and data
    """
    from xpublish_tiles.pipeline import check_data_is_renderable_size

    tms_crs = CRS.from_wkt(tms.crs.to_wkt())

    tms_to_wgs84 = transformer_from_crs(tms_crs, 4326)
    tms_xy_bounds = tms.xy_bbox
    geo_left, geo_bottom, geo_right, geo_top = tms_to_wgs84.transform_bounds(
        tms_xy_bounds.left,
        tms_xy_bounds.bottom,
        tms_xy_bounds.right,
        tms_xy_bounds.top,
    )
    tms_geo_bounds = morecantile.BoundingBox(
        left=geo_left, bottom=geo_bottom, right=geo_right, top=geo_top
    )

    grid_to_wgs84 = transformer_from_crs(grid.crs, 4326)

    bbox_lons = [grid.bbox.west, grid.bbox.east, grid.bbox.west, grid.bbox.east]
    bbox_lats = [grid.bbox.south, grid.bbox.south, grid.bbox.north, grid.bbox.north]
    wgs84_lons, wgs84_lats = grid_to_wgs84.transform(bbox_lons, bbox_lats)

    wgs84_west, wgs84_east = min(wgs84_lons), max(wgs84_lons)
    wgs84_south, wgs84_north = min(wgs84_lats), max(wgs84_lats)

    west = max(wgs84_west, tms_geo_bounds.left)
    east = min(wgs84_east, tms_geo_bounds.right)
    south = max(wgs84_south, tms_geo_bounds.bottom)
    north = min(wgs84_north, tms_geo_bounds.top)

    test_points = [
        (west, south),
        (east, south),
        (west, north),
        (east, north),
        ((west + east) / 2, (south + north) / 2),
    ]

    alternate = grid.pick_alternate_grid(tms_crs, coarsen_factors={})
    transformer = transformer_from_crs(tms_crs, grid.crs)

    for zoom in range(tms.minzoom, tms.maxzoom + 1):
        all_tiles_renderable = True

        for lon, lat in test_points:
            tile = tms.tile(lon, lat, zoom)
            bounds = tms.xy_bounds(tile)
            left, bottom, right, top = transformer.transform_bounds(
                bounds.left, bounds.bottom, bounds.right, bounds.top
            )

            tile_bbox = BBox(west=left, south=bottom, east=right, north=top)
            slicers = grid.sel(bbox=tile_bbox)

            if not check_data_is_renderable_size(slicers, da, grid, alternate):
                all_tiles_renderable = False
                break

        if all_tiles_renderable:
            return zoom

    return tms.minzoom
