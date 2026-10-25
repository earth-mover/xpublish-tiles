"""Helpers for the ``Features`` grid: vector data cubes of polygon features."""

from typing import TYPE_CHECKING, Any

import numpy as np
import pyproj

from xpublish_tiles.lib import unwrap_seam_vertices

if TYPE_CHECKING:
    from spatialpandas.geometry import MultiPolygonArray


def import_cf_geometry() -> tuple[Any, Any]:
    try:
        import cf_xarray.geometry as cf_geometry
        import shapely
    except ImportError as e:
        raise ImportError(
            "CF geometry containers require shapely; install `xpublish-tiles[vector]`."
        ) from e
    return shapely, cf_geometry


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
    import shapely

    parts, part_geom = shapely.get_parts(geometries, return_index=True)
    rings, ring_part = shapely.get_rings(parts, return_index=True)
    coords, vertex_ring = shapely.get_coordinates(rings, return_index=True)
    jump = (vertex_ring[1:] == vertex_ring[:-1]) & (np.abs(np.diff(coords[:, 0])) > 180)
    crossing = np.zeros(geometries.shape, dtype=bool)
    crossing[part_geom[ring_part[vertex_ring[1:][jump]]]] = True
    return crossing


def features_to_multipolygons(
    geoms: np.ndarray,
    crossing: np.ndarray,
    transformer: pyproj.Transformer,
) -> tuple["MultiPolygonArray", np.ndarray]:
    """Transform source-CRS geometries to a MultiPolygonArray and their
    source positions; crossing geometries are unwrapped and copied by
    ∓width, as in fix_healpix_discontinuity."""
    import pyarrow as pa
    import shapely
    from shapely import GeometryType
    from spatialpandas.geometry import MultiPolygonArray

    idx = np.flatnonzero(crossing)
    k, n = idx.size, len(geoms)
    # new writable array: to_ragged_array fails on read-only mixed-type input
    geoms = np.concatenate([geoms, geoms[idx], geoms[idx]])
    src = np.concatenate([np.arange(n), idx, idx])

    geom_type, coords, offsets = shapely.to_ragged_array(geoms)
    if geom_type == GeometryType.POLYGON:
        offsets = (*offsets, np.arange(len(geoms) + 1))
    elif geom_type != GeometryType.MULTIPOLYGON:
        raise ValueError(f"Expected polygon geometries; got {geom_type!r}.")
    ring_offsets, part_offsets, geom_offsets = offsets

    x, y = transformer.transform(coords[:, 0], coords[:, 1])
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if k:
        left, _, right, _ = transformer.transform_bounds(-180, -90, 180, 90)
        width = abs(right - left)
        part_geom = np.repeat(np.arange(len(geoms)), np.diff(geom_offsets))
        ring_geom = np.repeat(part_geom, np.diff(part_offsets))
        vert_geom = np.repeat(ring_geom, np.diff(ring_offsets))
        seam = np.concatenate([crossing, np.ones(2 * k, dtype=bool)])
        unwrap_seam_vertices(x, np.flatnonzero(seam[vert_geom]), width=width)
        x[(vert_geom >= n) & (vert_geom < n + k)] -= width
        x[vert_geom >= n + k] += width

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
    return MultiPolygonArray(multis), src
