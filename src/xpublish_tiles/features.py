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


def unwrap_longitude(geometries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unwrap each ring's longitudes (``np.unwrap``, period 360), and flag the
    features that changed: a ring edge longer than 180° counts as a crossing."""
    coords, vertex_geom = shapely.get_coordinates(geometries, return_index=True)
    lengths = shapely.get_num_coordinates(
        shapely.get_rings(shapely.get_parts(geometries))
    )
    starts = np.cumsum(lengths) - lengths
    x = coords[:, 0]
    offset = np.round((np.unwrap(x, period=360) - x) / 360) * 360
    # np.unwrap runs across ring boundaries; restart each ring at its first vertex.
    offset -= np.repeat(offset[starts], lengths)
    crossing = np.zeros(geometries.shape, dtype=bool)
    crossing[vertex_geom[offset != 0]] = True
    if not crossing.any():
        return geometries, crossing
    coords[:, 0] = x + offset
    return shapely.set_coordinates(geometries.copy(), coords), crossing


def split_at_antimeridian(geometries: np.ndarray) -> np.ndarray:
    """Split each feature that crosses 180° into parts on both sides, all in [-180, 180]."""
    unwrapped, crossing = unwrap_longitude(geometries)
    if not crossing.any():
        return geometries
    g = unwrapped[crossing]
    pieces = [
        shapely.transform(
            shapely.clip_by_rect(g, west, -90, west + 360, 90),
            lambda c, shift=shift: c + np.array([shift, 0.0]),
        )
        for west, shift in ((-540, 360), (-180, 0), (180, -360))
    ]
    out = geometries.copy()
    out[crossing] = shapely.union_all(np.stack(pieces), axis=0)
    return out


# We choose to split geometries at the antimeridian and store them as multipolygons.
# Since these geometries are (cached) coordinates, and the selection stuff will handle it
# easily, we do it upfront instead of fixing the wraps at render-time.
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
