"""Projection fastpaths: closed-form or factored replacements for pyproj.

Everything here is pure numpy/pyproj on raw arrays. The xarray-aware,
async orchestration lives in :mod:`xpublish_tiles.lib`.
"""

import warnings
from functools import lru_cache, partial

import numba
import numpy as np
import pyproj
from pyproj import CRS

from xpublish_tiles.utils import NUMBA_THREADING_LOCK

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


# --- conic source -> normal-cylindrical target -------------------------------
#
# A conic projection is polar about the cone apex: rho depends only on latitude,
# theta only on longitude. A normal-aspect cylindrical target (Web Mercator,
# plate carree) has X depending only on longitude and Y only on latitude. So the
# composed map factors: X is affine in theta, Y is a function of rho alone. Every
# transcendental collapses to one dimension, and the 2D work left is a hypot, an
# arctan2 and a table lookup.
#
# Phase 2, not implemented: the mirror case, a geographic source into a conic
# target, which is what the CanadianNAD83_LCC tile matrix set needs. There theta
# is 1D in lon and rho is 1D in lat, so x = rho*sin(theta), y = rho0 - rho*cos(theta)
# is an outer product — both reductions stay 1D and only four elementwise
# multiplies remain. Bigger win than this direction, same apex/n recovery.

CONIC_PROJ_NAMES = frozenset({"aea", "lcc"})
# Interpolation error budget for the Y(rho) table. Well below a pixel at any zoom.
CONIC_TOLERANCE_METERS = 0.01
_CONIC_VALIDATION_SIDE = 16
_CONIC_COARSE_NODES = 1024
_CONIC_MAX_NODES = 1 << 17
# The area of use understates the projected extent a raster can occupy.
_CONIC_RANGE_MARGIN = 0.3


@numba.njit(parallel=True, cache=True, boundscheck=False)
def _conic_kernel(x, y, apex_x, apex_y, slope, offset, lo, inv_h, y_nodes, out_x, out_y):
    """One fused pass: no temporaries, no binary search.

    ``sqrt(dx*dx + dy*dy)`` replaces ``np.hypot`` (the overflow guards cost more
    than they buy at projected-coordinate magnitudes) and the uniform node
    spacing turns the table lookup into an index computation. Returns a count of
    points outside the table so the caller can fall back.
    """
    flat_x = x.ravel()
    flat_y = y.ravel()
    flat_out_x = out_x.ravel()
    flat_out_y = out_y.ravel()
    last = y_nodes.size - 2
    outside = 0
    for k in numba.prange(flat_x.size):  # ty: ignore[not-iterable]
        dx = flat_x[k] - apex_x
        dy = apex_y - flat_y[k]
        flat_out_x[k] = np.arctan2(dx, dy) * slope + offset
        scaled = (np.sqrt(dx * dx + dy * dy) - lo) * inv_h
        i = int(scaled)
        if i < 0:
            i = 0
            outside += 1
        elif i > last:
            i = last
            outside += 1
        frac = scaled - i
        below = y_nodes[i]
        flat_out_y[k] = below + frac * (y_nodes[i + 1] - below)
    return outside


@numba.njit(parallel=True, cache=True, boundscheck=False)
def _conic_kernel_1d(
    x, y, apex_x, apex_y, slope, offset, lo, inv_h, y_nodes, out_x, out_y
):
    """Rectilinear source: take the 1D axes and skip broadcasting them.

    The outer product is formed inside the loop, so nothing the size of the
    output is allocated for the inputs.
    """
    last = y_nodes.size - 2
    outside = 0
    for i in numba.prange(x.size):  # ty: ignore[not-iterable]
        dx = x[i] - apex_x
        for j in range(y.size):
            dy = apex_y - y[j]
            out_x[i, j] = np.arctan2(dx, dy) * slope + offset
            scaled = (np.sqrt(dx * dx + dy * dy) - lo) * inv_h
            k = int(scaled)
            if k < 0:
                k = 0
                outside += 1
            elif k > last:
                k = last
                outside += 1
            below = y_nodes[k]
            out_y[i, j] = below + (scaled - k) * (y_nodes[k + 1] - below)
    return outside


class ConicToCylindrical:
    """Factored conic -> normal-cylindrical transform.

    ``X = x_slope * theta + x_offset`` exactly; ``Y = interp(rho)`` from a table
    dense enough for :data:`CONIC_TOLERANCE_METERS`.
    """

    __slots__ = ("apex_x", "apex_y", "inv_h", "rho_lo", "x_offset", "x_slope", "y_nodes")

    def __init__(self, apex_x, apex_y, x_slope, x_offset, rho_lo, rho_hi, y_nodes):
        self.apex_x = apex_x
        self.apex_y = apex_y
        self.x_slope = x_slope
        self.x_offset = x_offset
        self.rho_lo = rho_lo
        self.inv_h = (y_nodes.size - 1) / (rho_hi - rho_lo)
        self.y_nodes = y_nodes

    def transform(
        self, x: np.ndarray, y: np.ndarray, *, grid: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Returns None when any rho falls outside the table, so callers fall back.

        With ``grid``, 1D x and y are the axes of a rectilinear grid and the
        output has shape ``(x.size, y.size)``. Otherwise x and y are matching
        arrays of points. Two 1D arrays are ambiguous between the two, hence the
        explicit flag.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rectilinear = grid and x.ndim == 1 and y.ndim == 1
        shape = (x.size, y.size) if rectilinear else x.shape
        out_x = np.empty(shape, dtype=np.float64)
        out_y = np.empty(shape, dtype=np.float64)
        kernel = _conic_kernel_1d if rectilinear else _conic_kernel
        with NUMBA_THREADING_LOCK:
            outside = kernel(
                x,
                y,
                self.apex_x,
                self.apex_y,
                self.x_slope,
                self.x_offset,
                self.rho_lo,
                self.inv_h,
                self.y_nodes,
                out_x,
                out_y,
            )
        return None if outside else (out_x, out_y)


def _cone_geometry(crs: CRS) -> tuple[float, float, float] | None:
    """(apex_x, apex_y, n) in the CRS's own units, from its parameters.

    Snyder's conic constants. The ellipsoid is always metric while the grid may
    not be (US survey feet), hence the unit scaling.
    """
    with warnings.catch_warnings():
        # to_dict() warns that a PROJ string loses information. We only read the
        # conic parameters, which it keeps.
        warnings.simplefilter("ignore", UserWarning)
        params = crs.to_dict()
    if params.get("proj") not in CONIC_PROJ_NAMES:
        return None
    ellipsoid = crs.ellipsoid
    if ellipsoid is None or ellipsoid.semi_major_metre is None:
        return None
    unit = crs.axis_info[0].unit_conversion_factor
    a = ellipsoid.semi_major_metre / unit
    inverse_flattening = ellipsoid.inverse_flattening
    f = 1.0 / inverse_flattening if inverse_flattening else 0.0
    e = np.sqrt(2 * f - f * f)

    lat_1 = params.get("lat_1", 0.0)
    phi1 = np.deg2rad(lat_1)
    phi2 = np.deg2rad(params.get("lat_2", lat_1))
    phi0 = np.deg2rad(params.get("lat_0", 0.0))
    x_0 = params.get("x_0", 0.0) / unit
    y_0 = params.get("y_0", 0.0) / unit

    def m(phi):
        return np.cos(phi) / np.sqrt(1 - e * e * np.sin(phi) ** 2)

    def q(phi):
        s = np.sin(phi)
        if e == 0:
            return 2 * s
        return (1 - e * e) * (
            s / (1 - e * e * s * s) - np.log((1 - e * s) / (1 + e * s)) / (2 * e)
        )

    def t(phi):
        s = np.sin(phi)
        if e == 0:
            return np.tan(np.pi / 4 - phi / 2)
        return np.tan(np.pi / 4 - phi / 2) / ((1 - e * s) / (1 + e * s)) ** (e / 2)

    tangent = abs(phi1 - phi2) < 1e-12
    if params["proj"] == "aea":
        n = (
            np.sin(phi1)
            if tangent
            else (m(phi1) ** 2 - m(phi2) ** 2) / (q(phi2) - q(phi1))
        )
        if n == 0:
            return None
        rho0 = a * np.sqrt(m(phi1) ** 2 + n * q(phi1) - n * q(phi0)) / n
    else:
        n = (
            np.sin(phi1)
            if tangent
            else (np.log(m(phi1)) - np.log(m(phi2))) / (np.log(t(phi1)) - np.log(t(phi2)))
        )
        if n == 0:
            return None
        rho0 = a * (m(phi1) / (n * t(phi1) ** n)) * t(phi0) ** n
    if not np.isfinite(rho0):
        return None
    return x_0, y_0 + rho0, float(n)


@lru_cache
def conic_to_cylindrical(source_crs: CRS, target_crs: CRS) -> ConicToCylindrical | None:
    """Build the factored transform, or None if the map does not factor.

    The parameters give the cone geometry; only a numeric check tells us whether
    the *composed* map really is a function of (rho, theta). It is not when the
    datum step is position-dependent (e.g. BD72 -> WGS84 misses by metres), and
    the parameters alone cannot show that.
    """
    geometry = _cone_geometry(source_crs)
    if geometry is None:
        return None
    apex_x, apex_y, _ = geometry

    to_source = transformer_from_crs(source_crs.geodetic_crs, source_crs)
    to_target = transformer_from_crs(source_crs, target_crs)
    lon, lat = _area_of_use_grid(source_crs, _CONIC_VALIDATION_SIDE)
    sx, sy = to_source.transform(lon, lat)
    tx, ty = to_target.transform(sx, sy)
    finite = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(tx) & np.isfinite(ty)
    if finite.sum() < 4:
        return None
    sx, sy, tx, ty = sx[finite], sy[finite], tx[finite], ty[finite]

    dx, dy = sx - apex_x, apex_y - sy
    theta = np.arctan2(dx, dy)
    rho = np.hypot(dx, dy)
    x_slope, x_offset = np.polyfit(theta, tx, 1)

    # The area of use is a lon/lat box, so a projected raster on this CRS reaches
    # past it — its corners alone do. Pad generously, then clip back to where the
    # target is finite (rho grows towards the far pole, where Web Mercator Y
    # diverges). Anything still outside falls back to pyproj, so the margin is a
    # performance choice, not a correctness one.
    span = rho.max() - rho.min()
    lo = max(rho.min() - _CONIC_RANGE_MARGIN * span, 0.0)
    hi = rho.max() + _CONIC_RANGE_MARGIN * span
    clipped = _finite_rho_range(to_target, apex_x, apex_y, lo, hi, rho.min(), rho.max())
    if clipped is None:
        return None
    lo, hi = clipped

    coarse = _sample_y_of_rho(to_target, apex_x, apex_y, lo, hi, _CONIC_COARSE_NODES)
    # Linear interpolation error is |Y''| h^2 / 8, so it falls as h^2. One coarse
    # table sizes the final one.
    curvature = np.abs(np.diff(coarse, 2)).max()
    count = _CONIC_COARSE_NODES
    if curvature > 0:
        needed = _CONIC_COARSE_NODES * np.sqrt((curvature / 8) / CONIC_TOLERANCE_METERS)
        count = int(np.clip(needed, _CONIC_COARSE_NODES, _CONIC_MAX_NODES))
    y_nodes = _sample_y_of_rho(to_target, apex_x, apex_y, lo, hi, count)
    if not np.all(np.isfinite(y_nodes)):
        return None

    factored = ConicToCylindrical(
        apex_x, apex_y, float(x_slope), float(x_offset), lo, hi, y_nodes
    )
    check = factored.transform(sx, sy)
    if check is None:
        return None
    if max(np.abs(check[0] - tx).max(), np.abs(check[1] - ty).max()) > (
        CONIC_TOLERANCE_METERS
    ):
        return None
    return factored


def _sample_y_of_rho(
    to_target: pyproj.Transformer,
    apex_x: float,
    apex_y: float,
    lo: float,
    hi: float,
    count: int,
) -> np.ndarray:
    """Y along the theta=0 ray, i.e. the central meridian, as a function of rho."""
    rho = np.linspace(lo, hi, count)
    _, y = to_target.transform(np.full_like(rho, apex_x), apex_y - rho)
    return y


def _finite_rho_range(
    to_target: pyproj.Transformer,
    apex_x: float,
    apex_y: float,
    lo: float,
    hi: float,
    must_cover_lo: float,
    must_cover_hi: float,
) -> tuple[float, float] | None:
    """Shrink [lo, hi] to the finite run around the range we must cover."""
    rho = np.linspace(lo, hi, _CONIC_COARSE_NODES)
    y = _sample_y_of_rho(to_target, apex_x, apex_y, lo, hi, _CONIC_COARSE_NODES)
    bad = np.flatnonzero(~np.isfinite(y))
    if bad.size:
        below = bad[rho[bad] < must_cover_lo]
        above = bad[rho[bad] > must_cover_hi]
        if below.size != bad.size - above.size:
            return None  # a hole inside the range we need
        lo = rho[below[-1] + 1] if below.size else lo
        hi = rho[above[0] - 1] if above.size else hi
    return (lo, hi) if hi > lo else None
