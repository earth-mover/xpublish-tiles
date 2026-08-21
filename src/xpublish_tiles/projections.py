"""Projection fastpaths: closed-form or factored replacements for pyproj.

Everything here is pure numpy/pyproj on raw arrays. The xarray-aware,
async orchestration lives in :mod:`xpublish_tiles.lib`.
"""

import warnings
from dataclasses import dataclass
from functools import lru_cache, partial

import numba
import numpy as np
import pyproj
from pyproj import CRS

from xpublish_tiles.utils import NUMBA_THREADING_LOCK

_XY = tuple[np.ndarray, np.ndarray]

WGS84_SEMI_MAJOR_AXIS = np.float64(6378137.0)  # from proj
M_PI = 3.14159265358979323846  # from proj
M_2_PI = 6.28318530717958647693  # from proj

# 4326 with order of axes reversed.
OTHER_4326 = pyproj.CRS.from_user_input("WGS 84 (CRS84)")
WEB_MERCATOR = pyproj.CRS.from_epsg(3857)

# Below this, treating the source lon/lat as WGS84 lon/lat is sub-pixel at any
# zoom we serve, so the numpy fastpaths may skip pyproj's datum step.
MAX_DATUM_SHIFT_METERS = 1.0
_PROBE_SIDE = 5

# https://pyproj4.github.io/pyproj/stable/advanced_examples.html#caching-pyproj-objects
transformer_from_crs = lru_cache(partial(pyproj.Transformer.from_crs, always_xy=True))


def is_degree_geographic(crs: CRS) -> bool:
    """True for any geographic CRS with lon/lat axes in degrees (EPSG:4326,
    CRS84, custom spherical datums like HEALPix's, etc.). The 4326-fastpath
    in :func:`xpublish_tiles.lib.transform_coordinates` uses this to skip the
    pyproj roundtrip and just wrap lon to [-180, 180], which is valid for all
    such CRSes. We assume any residual datum shift is sub-meter and below pixel
    resolution.
    """
    return crs.is_geographic and all(ax.unit_name == "degree" for ax in crs.axis_info)


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
# A conic projection is polar about the cone apex. That mean ρ(latitude and
# θ(longitude). A normal-aspect cylindrical target (Web Mercator,
# plate carree) has X(longitude) and Y(latitude). So the
# composed map factors: X is affine in θ, Y is a function of ρ alone.
# Then every transcendental collapses to one dimension,
# and the 2D work left is a hypot, an arctan2 and a table lookup.
#
# Phase 2, not implemented: the mirror case, a geographic source into a conic
# target, which is what the CanadianNAD83_LCC tile matrix set needs. There theta
# is 1D in lon and rho is 1D in lat, so x = rho*sin(theta), y = rho0 - rho*cos(theta)
# is an outer product — both reductions stay 1D and only four elementwise
# multiplies remain. Bigger win than this direction, same apex/n recovery.

# Conic source CRSes that factor cleanly into Web Mercator. Being on this list
# asserts two things, both re-checked against pyproj by tests/test_projections.py:
# the composed map really is a function of (rho, theta), and a table of
# _CONIC_NODES nodes holds CONIC_TOLERANCE_METERS across the area of use. Neither
# is safe to assume for an arbitrary conic -- a position-dependent datum step
# breaks the first (BD72 -> WGS84 misses by metres) and Arctic latitudes break
# the second (EPSG:3978) -- so add a CRS here only with a test beside it.
CONIC_ALLOWLIST = frozenset(
    {
        5070,  # NAD83 / Conus Albers
        6350,  # NAD83(2011) / Conus Albers
        3005,  # NAD83 / BC Albers
    }
)
# Interpolation error budget for the Y(rho) table. Well below a pixel at any zoom.
CONIC_TOLERANCE_METERS = 0.01
_CONIC_PROJ_NAMES = frozenset({"aea", "lcc"})
_CONIC_NODES = 1 << 16
_CONIC_SAMPLE_SIDE = 16
# A raster reaches past its CRS's area of use, so widen the table's latitude span.
_CONIC_LAT_PAD = 5.0
# Y diverges at the poles, so keep the padding short of them. It is a guard, not
# a tuning knob: Y'' grows as 1/rho^2 up there, and a CRS reaching that far needs
# far more nodes for the same tolerance, which is why EPSG:3978 is not allowed.
_CONIC_MAX_LAT = 89.5


@numba.njit(inline="always", cache=True)
def _project(dx, dy, slope, offset, rho_lo, inv_h, y_nodes):
    """One point. theta is exact; Y is a lookup in the uniform table."""
    scaled = (np.sqrt(dx * dx + dy * dy) - rho_lo) * inv_h
    last = y_nodes.size - 2
    k = int(scaled)
    outside = k < 0 or k > last
    if outside:
        k = min(max(k, 0), last)
    below = y_nodes[k]
    return (
        np.arctan2(dx, dy) * slope + offset,
        below + (scaled - k) * (y_nodes[k + 1] - below),
        outside,
    )


@numba.njit(parallel=True, cache=True, boundscheck=False)
def _grid_kernel(
    x, y, apex_x, apex_y, slope, offset, rho_lo, inv_h, y_nodes, out_x, out_y
):
    """Rectilinear source: the 1D axes are never broadcast."""
    missed = 0
    for i in numba.prange(x.size):  # ty: ignore[not-iterable]
        dx = x[i] - apex_x
        for j in range(y.size):
            out_x[i, j], out_y[i, j], outside = _project(
                dx, apex_y - y[j], slope, offset, rho_lo, inv_h, y_nodes
            )
            missed += outside
    return missed


@numba.njit(parallel=True, cache=True, boundscheck=False)
def _points_kernel(
    x, y, apex_x, apex_y, slope, offset, rho_lo, inv_h, y_nodes, out_x, out_y
):
    """Curvilinear source: x and y are matching arrays of points."""
    missed = 0
    for k in numba.prange(x.size):  # ty: ignore[not-iterable]
        out_x[k], out_y[k], outside = _project(
            x[k] - apex_x, apex_y - y[k], slope, offset, rho_lo, inv_h, y_nodes
        )
        missed += outside
    return missed


@dataclass(frozen=True)
class ConicToCylindrical:
    """``X = x_slope * theta + x_offset``; ``Y`` interpolated from ``y_nodes``.

    Both transforms return None if any point misses the table, so the caller can
    fall back to pyproj.
    """

    apex_x: float
    apex_y: float
    x_slope: float
    x_offset: float
    rho_lo: float
    inv_h: float
    y_nodes: np.ndarray

    def transform(self, x: np.ndarray, y: np.ndarray) -> _XY | None:
        """x and y are matching arrays of points, of any shape."""
        out_x = np.empty(x.shape, dtype=np.float64)
        out_y = np.empty(x.shape, dtype=np.float64)
        missed = self._run(
            _points_kernel, np.ravel(x), np.ravel(y), out_x.ravel(), out_y.ravel()
        )
        return None if missed else (out_x, out_y)

    def transform_grid(self, x: np.ndarray, y: np.ndarray) -> _XY | None:
        """x and y are the 1D axes of a rectilinear grid; output is (x.size, y.size)."""
        out_x = np.empty((x.size, y.size), dtype=np.float64)
        out_y = np.empty((x.size, y.size), dtype=np.float64)
        missed = self._run(_grid_kernel, x, y, out_x, out_y)
        return None if missed else (out_x, out_y)

    def _run(self, kernel, x, y, out_x, out_y):
        with NUMBA_THREADING_LOCK:
            return kernel(
                np.ascontiguousarray(x, dtype=np.float64),
                np.ascontiguousarray(y, dtype=np.float64),
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


def _cone_apex(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Intersect two meridians of a projected lat/lon grid.

    Meridians are straight lines through the apex, so two of them fix it. Shorter
    than Snyder's constants, and it needs no per-projection formula, no ellipsoid
    and no unit handling.
    """
    first, last = np.stack([x[:, 0], y[:, 0]], -1), np.stack([x[:, -1], y[:, -1]], -1)
    origin, along = first[0], first[-1] - first[0]
    other_origin, other_along = last[0], last[-1] - last[0]
    matrix = np.stack([along, -other_along], -1)
    steps = np.linalg.solve(matrix, other_origin - origin)
    apex = origin + steps[0] * along
    return float(apex[0]), float(apex[1])


@lru_cache
def _conic_signature(crs: CRS) -> tuple | None:
    """The parameters that define the cone, or None if the CRS is not conic.

    Matching on these rather than on the EPSG code lets a dataset carrying the
    same projection as a bare PROJ string or a CF grid mapping -- no EPSG code
    attached -- still reach the fastpath.
    """
    with warnings.catch_warnings():
        # to_dict() warns that a PROJ string loses information; we only read the
        # conic parameters, which it keeps.
        warnings.simplefilter("ignore", UserWarning)
        params = crs.to_dict()
    if params.get("proj") not in _CONIC_PROJ_NAMES:
        return None
    lat_1 = params.get("lat_1", 0.0)
    angles = (lat_1, params.get("lat_2", lat_1), params.get("lat_0", 0.0))
    offsets = (params.get("lon_0", 0.0), params.get("x_0", 0.0), params.get("y_0", 0.0))
    return (
        params["proj"],
        *(round(v, 9) for v in angles + offsets),
        params.get("units"),
        crs.datum.name if crs.datum is not None else None,
    )


@lru_cache
def _allowed_conics() -> dict[tuple, CRS]:
    """Allowlist signature -> the canonical CRS, which carries an area of use."""
    allowed = {}
    for code in CONIC_ALLOWLIST:
        crs = CRS.from_epsg(code)
        signature = _conic_signature(crs)
        if signature is not None:
            allowed[signature] = crs
    return allowed


@lru_cache
def conic_to_cylindrical(source_crs: CRS, target_crs: CRS) -> ConicToCylindrical | None:
    """Build the factored transform for an allowlisted conic source."""
    signature = _conic_signature(source_crs)
    if signature is None or signature not in _allowed_conics():
        return None
    canonical = _allowed_conics()[signature]

    to_source = transformer_from_crs(source_crs.geodetic_crs, source_crs)
    to_target = transformer_from_crs(source_crs, target_crs)
    # Sample the canonical CRS's area of use: an equivalent CRS built from a PROJ
    # string has none, and a global fallback would stretch the table pole to pole.
    lon, lat = _area_of_use_grid(canonical, _CONIC_SAMPLE_SIDE)
    grid_x, grid_y = to_source.transform(lon, lat)
    side = _CONIC_SAMPLE_SIDE
    apex_x, apex_y = _cone_apex(grid_x.reshape(side, side), grid_y.reshape(side, side))

    # rho depends on latitude alone, so the two padded latitude limits bound it.
    edge_lat = np.clip(
        [lat.min() - _CONIC_LAT_PAD, lat.max() + _CONIC_LAT_PAD],
        -_CONIC_MAX_LAT,
        _CONIC_MAX_LAT,
    )
    edge_x, edge_y = to_source.transform(np.full(2, lon[0]), edge_lat)
    rho_lo, rho_hi = sorted(np.hypot(edge_x - apex_x, apex_y - edge_y))
    y_nodes = _target_y(
        to_target, apex_x, apex_y, np.linspace(rho_lo, rho_hi, _CONIC_NODES)
    )

    # X is affine in theta; two points would do, but a fit costs nothing.
    theta = np.arctan2(grid_x - apex_x, apex_y - grid_y)
    x_slope, x_offset = np.polyfit(theta, to_target.transform(grid_x, grid_y)[0], 1)

    return ConicToCylindrical(
        apex_x,
        apex_y,
        float(x_slope),
        float(x_offset),
        rho_lo,
        (_CONIC_NODES - 1) / (rho_hi - rho_lo),
        y_nodes,
    )


def _target_y(
    to_target: pyproj.Transformer, apex_x: float, apex_y: float, rho: np.ndarray
) -> np.ndarray:
    """Y along the theta=0 ray, which is the central meridian."""
    return to_target.transform(np.full_like(rho, apex_x), apex_y - rho)[1]
