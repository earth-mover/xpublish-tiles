"""Projection fastpaths: closed-form or factored replacements for pyproj.

Everything here is pure numpy/pyproj on raw arrays. The xarray-aware,
async orchestration lives in :mod:`xpublish_tiles.lib`.
"""

from functools import lru_cache, partial

import numpy as np
import pyproj
from pyproj import CRS

WGS84_SEMI_MAJOR_AXIS = np.float64(6378137.0)  # from proj
M_PI = 3.14159265358979323846  # from proj
M_2_PI = 6.28318530717958647693  # from proj

# 4326 with order of axes reversed.
OTHER_4326 = pyproj.CRS.from_user_input("WGS 84 (CRS84)")
WEB_MERCATOR = pyproj.CRS.from_epsg(3857)

# https://pyproj4.github.io/pyproj/stable/advanced_examples.html#caching-pyproj-objects
transformer_from_crs = lru_cache(partial(pyproj.Transformer.from_crs, always_xy=True))


def is_degree_geographic(crs: CRS) -> bool:
    """True for any geographic CRS with lon/lat axes in degrees (EPSG:4326,
    CRS84, custom spherical datums like HEALPix's, etc.). The 4326-fastpath
    in :func:`xpublish_tiles.lib.transform_coordinates` uses this to skip the
    pyproj roundtrip and just wrap lon to [-180, 180], which is valid for all
    such CRSes — any residual datum shift is sub-meter and below pixel
    resolution.
    """
    return crs.is_geographic and all(ax.unit_name == "degree" for ax in crs.axis_info)


# Below this, treating the source lon/lat as WGS84 lon/lat is sub-pixel at any
# zoom we serve, so the numpy fastpaths may skip pyproj's datum step.
MAX_DATUM_SHIFT_METERS = 1.0
_PROBE_SIDE = 5


@lru_cache
def has_null_datum_shift(crs: CRS) -> bool:
    """True when `crs` lon/lat may be treated as WGS84 lon/lat.

    Probes the datum step over the CRS's area of use rather than pattern-matching
    operation names: proj picks the operation per point, and a "null" one (e.g.
    EPSG:1188 NAD83->WGS84) is indistinguishable from a ballpark fallback taken
    because a shift grid is missing. Either way, what matters is the displacement.
    """
    lon, lat = _area_of_use_grid(crs, _PROBE_SIDE)
    x, y = transformer_from_crs(crs, 4326).transform(lon, lat)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return False
    dlon = (x - lon)[finite]
    dlon = (dlon + 180.0) % 360.0 - 180.0
    shift = np.hypot(dlon * np.cos(np.deg2rad(lat[finite])), (y - lat)[finite]) * 111320
    return bool(shift.max() < MAX_DATUM_SHIFT_METERS)


def _area_of_use_grid(crs: CRS, side: int) -> tuple[np.ndarray, np.ndarray]:
    """Flattened lon/lat sample grid covering the CRS's area of use."""
    area = crs.area_of_use
    west, south, east, north = (
        (area.west, area.south, area.east, area.north)
        if area is not None
        else (-180.0, -85.0, 180.0, 85.0)
    )
    if east < west:
        east += 360.0
    lon, lat = np.meshgrid(np.linspace(west, east, side), np.linspace(south, north, side))
    return lon.ravel(), lat.ravel()


def epsg4326to3857(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = WGS84_SEMI_MAJOR_AXIS

    x = np.asarray(lon, dtype=np.float64, copy=True)
    y = np.asarray(lat, dtype=np.float64, copy=True)

    # Only normalize longitude values that are outside the [-180, 180] range
    # This preserves precision for values already in the valid range
    # pyproj accepts both -180 and 180 as valid values without wrapping
    needs_normalization = (x > 180) | (x < -180)

    np.deg2rad(x, out=x)
    if np.any(needs_normalization):
        # Only normalize the values that need it to preserve precision
        # doing it this way matches proj
        x[needs_normalization] = ((x[needs_normalization] + M_PI) % (2 * M_PI)) - M_PI
    # Clamp latitude to avoid infinity at poles in-place
    # Web Mercator is only valid between ~85.05 degrees
    # Given our padding, we may be sending in data at latitudes poleward of MAX_LAT
    # MAX_LAT = 85.051128779806604  # atan(sinh(pi)) * 180 / pi
    # np.clip(y, -MAX_LAT, MAX_LAT, out=y)

    # Y coordinate: use more stable formula for large latitudes
    # Using: y = a * asinh(tan(φ)) for better numerical stability
    # following the proj formula
    # https://github.com/OSGeo/PROJ/blob/ff43c46b19802f5953a1546b05f59c5b9ee65795/src/projections/merc.cpp#L14
    # https://proj.org/en/stable/operations/projections/merc.html#forward-projection
    # Note: WebMercator uses the "spherical form"
    np.deg2rad(y, out=y)
    np.tan(y, out=y)
    np.arcsinh(y, out=y)

    x *= a
    y *= a

    return x, y


def aeqd_to_4326(
    x_m: np.ndarray,
    y_m: np.ndarray,
    center_lat: float,
    center_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast azimuthal equidistant (meters) to EPSG:4326 (degrees) conversion.

    Uses flat-earth approximation. Accurate to 0.3% at 300km from center.
    ~200x faster than pyproj for large arrays. Modifies x_m and y_m in place.
    """
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * np.cos(np.radians(center_lat))
    x_m /= meters_per_deg_lon
    x_m += center_lon
    y_m /= meters_per_deg_lat
    y_m += center_lat
    return x_m, y_m
