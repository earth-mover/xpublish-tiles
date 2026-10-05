"""HEALPix helpers: xyf <-> nested Morton kernels and bbox -> cell selection."""

import functools
import math

import numba
import numpy as np
import pandas as pd
from pyproj.aoi import BBox

_M1 = np.uint64(0x5555555555555555)
_M2 = np.uint64(0x3333333333333333)
_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_M8 = np.uint64(0x00FF00FF00FF00FF)
_M16 = np.uint64(0x0000FFFF0000FFFF)
_M32 = np.uint64(0x00000000FFFFFFFF)


@numba.njit(nogil=True, cache=True)
def _spread(v):
    v = v & _M32
    v = (v | (v << np.uint64(16))) & _M16
    v = (v | (v << np.uint64(8))) & _M8
    v = (v | (v << np.uint64(4))) & _M4
    v = (v | (v << np.uint64(2))) & _M2
    return (v | (v << np.uint64(1))) & _M1


@numba.njit(nogil=True, cache=True)
def _compact(v):
    v = v & _M1
    v = (v | (v >> np.uint64(1))) & _M2
    v = (v | (v >> np.uint64(2))) & _M4
    v = (v | (v >> np.uint64(4))) & _M8
    v = (v | (v >> np.uint64(8))) & _M16
    return (v | (v >> np.uint64(16))) & _M32


@numba.njit(nogil=True, cache=True)
def _fyx_to_nested(f, y, x, level, out):
    shift = np.uint64(2 * level)
    one = np.uint64(1)
    for i in range(out.size):
        out[i] = (f[i] << shift) | _spread(x[i]) | (_spread(y[i]) << one)


@numba.njit(nogil=True, cache=True)
def _nested_to_fyx(ids, level, f, y, x):
    shift = np.uint64(2 * level)
    mask = (np.uint64(1) << shift) - np.uint64(1)
    one = np.uint64(1)
    for i in range(ids.size):
        local = ids[i] & mask
        f[i] = ids[i] >> shift
        x[i] = _compact(local)
        y[i] = _compact(local >> one)


def fyx_to_nested(f, y, x, level: int) -> np.ndarray:
    """Nested ids for (face, y, x); x on even bits, y on odd bits (HEALPix xyf)."""
    f, y, x = (np.ascontiguousarray(a, dtype=np.uint64).ravel() for a in (f, y, x))
    out = np.empty(f.size, dtype=np.uint64)
    _fyx_to_nested(f, y, x, level, out)
    return out


def nested_to_fyx(ids, level: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.ascontiguousarray(ids, dtype=np.uint64).ravel()
    f, y, x = (np.empty(ids.size, dtype=np.uint64) for _ in range(3))
    _nested_to_fyx(ids, level, f, y, x)
    return f.astype(np.int64), y.astype(np.int64), x.astype(np.int64)


MERCATOR_MAX_LAT = 85.0511287798
POLAR_CLIP_LAT = 75.0


def mercator_polar_stretch(south: float, north: float) -> float:
    """Budget multiplier so the poleward row (clipped to POLAR_CLIP_LAT) keeps the mean density.

    Web Mercator pixel area on the sphere goes as cos²φ; its tile mean is
    (sin φn − sin φs) / (yn − ys).
    """
    s, n = (
        math.radians(min(max(v, -MERCATOR_MAX_LAT), MERCATOR_MAX_LAT))
        for v in (south, north)
    )
    if n - s < 1e-9:
        return 1.0
    mean_cos2 = (math.sin(n) - math.sin(s)) / (
        math.asinh(math.tan(n)) - math.asinh(math.tan(s))
    )
    ref = math.radians(min(max(abs(south), abs(north)), POLAR_CLIP_LAT))
    return max(1.0, mean_cos2 / math.cos(ref) ** 2)


def coarse_level(
    bbox: BBox, level: int, max_cells: int, *, mercator_stretch: bool = True
) -> int:
    """Finest level <= ``level`` whose estimated cell count in ``bbox`` fits ``max_cells``."""
    width = min(max(bbox.east - bbox.west, 0.0), 360.0)
    south, north = max(bbox.south, -90.0), min(bbox.north, 90.0)
    budget = max_cells * (
        mercator_polar_stretch(south, north) if mercator_stretch else 1.0
    )
    omega = math.radians(width) * (
        math.sin(math.radians(north)) - math.sin(math.radians(south))
    )
    ncells = omega / (4 * math.pi / (12 * 4**level))
    k = 0
    while k < level and ncells / 4**k > budget:
        k += 1
    return level - k


def bbox_cell_ids(bbox: BBox, depth: int) -> np.ndarray:
    """Nested ids at ``depth`` covering ``bbox``, dilated by one ring; sorted, unique."""
    from healpix_geo.nested import kth_neighbourhood, zone_coverage

    # ``zone_coverage`` takes degrees with lon_min ∈ [0, 360),
    # 0 < lon_max < 360 (exactly 360 panics), lat ∈ [-90, 90].
    # Also: cells whose boundary lies exactly on the zone edge are handled
    # inconsistently, so we nudge west/east outward by a small epsilon to
    # guarantee boundary cells are included.
    EPS = 1e-6
    MAX_LON = 360.0 - 1e-9
    west = (bbox.west - EPS) % 360.0
    east = (bbox.east + EPS) % 360.0
    if east == 0:
        east = MAX_LON
    south = max(bbox.south, -90.0)
    north = min(bbox.north, 90.0)
    if bbox.east - bbox.west >= 360:
        cell_ids, _, _ = zone_coverage((0.0, south, MAX_LON, north), depth, flat=True)
    elif west < east:
        cell_ids, _, _ = zone_coverage((west, south, east, north), depth, flat=True)
    else:
        # spans the anti-meridian
        # Normalize west/east into [0, 360) and split into two.
        left, _, _ = zone_coverage((west, south, MAX_LON, north), depth, flat=True)
        right, _, _ = zone_coverage((0.0, south, east, north), depth, flat=True)
        cell_ids = np.concatenate([left, right])

    # Dilate by one ring so neighbor cells covering tile-edge gaps are
    # included. This pads the selection in the same spirit as
    # ``apply_default_pad`` for 2D grids.
    neighbors = kth_neighbourhood(cell_ids.astype(np.uint64), depth, ring=1)
    neighbors = neighbors.ravel()
    # base-cell corners have 7 neighbours; the missing slot is -1
    return np.sort(pd.unique(neighbors[neighbors >= 0])).astype(cell_ids.dtype)


@functools.cache
def antimeridian_cells(depth: int) -> np.ndarray:
    """Cells touching lon 180° at ``depth`` (hairline zone straddling the seam)."""
    from healpix_geo.nested import zone_coverage

    eps = 1e-6
    cells, _, _ = zone_coverage((180.0 - eps, -90.0, 180.0 + eps, 90.0), depth, flat=True)
    cells.setflags(write=False)
    return cells
