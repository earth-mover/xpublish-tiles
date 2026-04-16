"""Tile-related utility functions for grids."""

import threading

import cachetools
import morecantile
import numpy as np
from pyproj import CRS
from pyproj.aoi import BBox

import xarray as xr
from xpublish_tiles.grids import GridSystem, GridSystem2D, Triangular
from xpublish_tiles.lib import (
    apply_default_pad,
    check_data_is_renderable_size,
    normalize_slicers,
    transformer_from_crs,
)
from xpublish_tiles.utils import time_debug, xarray_object_key

_MIN_ZOOM_CACHE = cachetools.LRUCache(maxsize=8192)
_MIN_ZOOM_LOCK = threading.Lock()


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
def _compute_min_zoom(
    grid: GridSystem,
    tms: morecantile.TileMatrixSet,
    da: xr.DataArray,
    *,
    style: str,
) -> int:
    tms_crs = CRS.from_wkt(tms.crs.to_wkt())

    geo_left, geo_bottom, geo_right, geo_top = tms.bbox
    if geo_left > geo_right:
        # Handle antimeridian-crossing TMS where left > right
        # e.g. NZTM2000
        geo_right += 360
    tms_geo_bounds = morecantile.BoundingBox(
        left=geo_left, bottom=geo_bottom, right=geo_right, top=geo_top
    )

    grid_to_wgs84 = transformer_from_crs(grid.crs, 4326)

    # Sample points along the grid's actual boundary in its native CRS, then
    # transform to WGS84. This avoids the axis-aligned-bbox overestimate you get
    # from transforming only the 4 corners — which places "corner" test points
    # outside the grid's real footprint for non-rectilinear projections (e.g.
    # EPSG:3035), producing tile bboxes that don't overlap the grid.
    n = 4
    edge_ys = np.linspace(grid.bbox.south, grid.bbox.north, n)
    edge_xs = np.linspace(grid.bbox.west, grid.bbox.east, n)
    native_xs = np.concatenate(
        [
            np.full(n, grid.bbox.west),  # west edge
            np.full(n, grid.bbox.east),  # east edge
            edge_xs[1:-1],  # south edge (excl. corners)
            edge_xs[1:-1],  # north edge (excl. corners)
            [(grid.bbox.west + grid.bbox.east) / 2],  # center
        ]
    )
    native_ys = np.concatenate(
        [
            edge_ys,
            edge_ys,
            np.full(n - 2, grid.bbox.south),
            np.full(n - 2, grid.bbox.north),
            [(grid.bbox.south + grid.bbox.north) / 2],
        ]
    )
    wgs84_lons, wgs84_lats = grid_to_wgs84.transform(native_xs, native_ys)

    # Clip to TMS geo bounds so tms.tile(lon, lat, zoom) is valid.
    wgs84_lons = np.clip(wgs84_lons, tms_geo_bounds.left, tms_geo_bounds.right)
    wgs84_lats = np.clip(wgs84_lats, tms_geo_bounds.bottom, tms_geo_bounds.top)
    test_points = list(zip(wgs84_lons.tolist(), wgs84_lats.tolist(), strict=True))

    alternate = grid.pick_alternate_grid(tms_crs, coarsen_factors={})
    transformer = transformer_from_crs(tms_crs, grid.crs)

    def all_renderable(zoom: int) -> bool:
        unique_tiles = {
            (tile.x, tile.y)
            for tile in (tms.tile(lon, lat, zoom) for lon, lat in test_points)
        }
        for x, y in unique_tiles:
            bounds = tms.xy_bounds(morecantile.Tile(x=x, y=y, z=zoom))
            left, bottom, right, top = transformer.transform_bounds(
                bounds.left, bounds.bottom, bounds.right, bounds.top
            )
            # Handle antimeridian-crossing tiles where left > right after transform
            if grid.crs.is_geographic and left > right:
                right += 360

            tile_bbox = BBox(west=left, south=bottom, east=right, north=top)
            slicers = grid.sel(bbox=tile_bbox)
            if isinstance(grid, GridSystem2D):
                slicers = apply_default_pad(slicers, da, grid)
                slicers = normalize_slicers(slicers, dict(da.sizes))
            if not check_data_is_renderable_size(
                slicers, da, grid, alternate, style=style
            ):
                return False
        return True

    # Renderability is monotonic in zoom (higher zoom → smaller tiles → fewer
    # cells). Binary-search for the smallest zoom that is fully renderable.
    lo, hi = tms.minzoom, tms.maxzoom
    if all_renderable(lo):
        return lo
    if not all_renderable(hi):
        return tms.minzoom
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if all_renderable(mid):
            hi = mid
        else:
            lo = mid
    return hi


def get_min_zoom(
    grid: GridSystem,
    tms: morecantile.TileMatrixSet,
    da: xr.DataArray,
    style: str,
    xpublish_id: str | None = None,
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
    xpublish_id : str | None
        Optional dataset identifier for caching. When provided,
        results are cached per (xpublish_id, spatial_dims, tms.id).

    Returns
    -------
    int
        Minimum safe zoom level for this grid and data
    """
    if xpublish_id is not None:
        cache_key: tuple | None = (xpublish_id, xarray_object_key(da), tms.id, style)
    else:
        cache_key = None

    if cache_key is not None and cache_key in _MIN_ZOOM_CACHE:
        return _MIN_ZOOM_CACHE[cache_key]

    with _MIN_ZOOM_LOCK:
        if cache_key is not None and cache_key in _MIN_ZOOM_CACHE:
            return _MIN_ZOOM_CACHE[cache_key]

        result = _compute_min_zoom(grid, tms, da, style=style)

        if cache_key is not None:
            _MIN_ZOOM_CACHE[cache_key] = result

        return result
