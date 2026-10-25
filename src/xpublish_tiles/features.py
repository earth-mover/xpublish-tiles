"""Helpers for the ``Features`` grid: vector data cubes of polygon features.

Needs the ``[vector]`` extra; import this module lazily from core code.
"""

import numpy as np
import pyarrow as pa
import pyproj
from spatialpandas.geometry import MultiPolygonArray

import xarray as xr

try:
    import cf_xarray.geometry as cf_geometry
    import shapely
    from shapely import GeometryType
except ImportError as e:
    raise ImportError(
        "Vector data cubes require shapely; install `xpublish-tiles[vector]`."
    ) from e


def geometries_from_cf(ds: xr.Dataset, container: str) -> np.ndarray:
    return cf_geometry.cf_to_shapely(ds, container=container).values


def wrap_longitude(coords: np.ndarray) -> np.ndarray:
    coords = coords.copy()
    x = coords[:, 0]
    coords[:, 0] = np.where((x < -180) | (x > 180), ((x + 180) % 360) - 180, x)
    return coords


def shift_longitude_to_360(coords: np.ndarray) -> np.ndarray:
    coords = coords.copy()
    coords[:, 0] = np.where(coords[:, 0] < 0, coords[:, 0] + 360, coords[:, 0])
    return coords


def crossing_features(geometries: np.ndarray) -> np.ndarray:
    """A feature crosses when a ring has consecutive vertices > 180 degrees apart in longitude."""
    parts, part_geom = shapely.get_parts(geometries, return_index=True)
    rings, ring_part = shapely.get_rings(parts, return_index=True)
    coords, vertex_ring = shapely.get_coordinates(rings, return_index=True)
    jump = (vertex_ring[1:] == vertex_ring[:-1]) & (np.abs(np.diff(coords[:, 0])) > 180)
    crossing = np.zeros(geometries.shape, dtype=bool)
    crossing[part_geom[ring_part[vertex_ring[1:][jump]]]] = True
    return crossing


def split_at_antimeridian(geometries: np.ndarray) -> np.ndarray:
    """Split each feature that crosses 180° into parts on both sides, all in [-180, 180]."""
    crossing = crossing_features(geometries)
    if not crossing.any():
        return geometries
    # In [0, 360) a crossing ring is continuous; cut it at 180 and move the east part back.
    shifted = shapely.transform(geometries[crossing], shift_longitude_to_360)
    west = shapely.clip_by_rect(shifted, 0, -90, 180, 90)
    east = shapely.transform(
        shapely.clip_by_rect(shifted, 180, -90, 360, 90),
        lambda c: c - np.array([360.0, 0.0]),
    )
    out = geometries.copy()
    out[crossing] = shapely.union(west, east)
    return out


def features_to_multipolygons(
    geoms: np.ndarray,
    transformer: pyproj.Transformer,
) -> MultiPolygonArray:
    """Transform source-CRS geometries to a MultiPolygonArray."""
    # new writable array: to_ragged_array fails on read-only mixed-type input
    geoms = np.array(geoms, dtype=object, copy=True)

    geom_type, coords, offsets = shapely.to_ragged_array(geoms)
    if geom_type == GeometryType.POLYGON:
        offsets = (*offsets, np.arange(len(geoms) + 1))
    elif geom_type != GeometryType.MULTIPOLYGON:
        raise ValueError(f"Expected polygon geometries; got {geom_type!r}.")
    ring_offsets, part_offsets, geom_offsets = offsets

    x, y = transformer.transform(coords[:, 0], coords[:, 1])
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    flat = np.empty(2 * x.size, dtype=np.float64)
    flat[0::2], flat[1::2] = x, y
    inner = pa.Array.from_buffers(pa.float64(), flat.size, [None, pa.py_buffer(flat)], 0)
    rings_arr = pa.ListArray.from_arrays(
        pa.array(2 * np.asarray(ring_offsets, dtype=np.int32)), inner
    )
    parts_arr = pa.ListArray.from_arrays(
        pa.array(np.asarray(part_offsets, dtype=np.int32)), rings_arr
    )
    multis = pa.ListArray.from_arrays(
        pa.array(np.asarray(geom_offsets, dtype=np.int32)), parts_arr
    )
    return MultiPolygonArray(multis)
