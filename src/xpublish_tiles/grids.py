import itertools
import re
import threading
import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, Self, cast

import cachetools
import cf_xarray  # noqa: F401
import numbagg
import numpy as np
import pandas as pd
import rasterix
import triangle
from numba_celltree import CellTree2d
from pyproj import CRS
from pyproj.aoi import BBox

import xarray as xr
from xarray.core.indexing import IndexSelResult
from xpublish_tiles.config import config
from xpublish_tiles.lib import (
    Fill,
    VariableNotFoundError,
    _prevent_slice_overlap,
    crs_repr,
    is_4326_like,
    unwrap,
)
from xpublish_tiles.logger import get_context_logger, log_duration
from xpublish_tiles.utils import NUMBA_THREADING_LOCK, time_debug

GRID_DETECTION_LOCK = threading.Lock()

DEFAULT_CRS = CRS.from_epsg(4326)
MAX_COORD_VAR_NBYTES = 1 * 1024 * 1024 * 1024


@dataclass
class UgridIndexer:
    """Dataclass for UGRID indexing results from CellTreeIndex.sel."""

    vertices: np.ndarray
    connectivity: np.ndarray
    # face indices that intersect the anti-meridian
    antimeridian_vertices: dict[str, np.ndarray]


@dataclass
class GridMetadata:
    """Grid metadata with coordinate names, CRS, and grid class."""

    X: str
    Y: str
    crs: CRS
    grid_cls: type["GridSystem"]

    def __repr__(self) -> str:
        return (
            f"GridMetadata(X={self.X!r}, Y={self.Y!r}, "
            f"crs={crs_repr(self.crs)}, grid_cls={self.grid_cls.__name__})"
        )


@dataclass
class GridMappingInfo:
    """Information about a grid mapping and its coordinates."""

    grid_mapping: xr.DataArray | None
    crs: CRS | None
    coordinates: tuple[str, ...] | None

    def __repr__(self) -> str:
        gm_repr = (
            f"<DataArray: {self.grid_mapping.name}>"
            if self.grid_mapping is not None
            else "None"
        )
        return (
            f"GridMappingInfo(grid_mapping={gm_repr}, "
            f"crs={crs_repr(self.crs)}, coordinates={self.coordinates!r})"
        )


# Regex patterns for coordinate detection
X_COORD_PATTERN = re.compile(
    r"^(x|i|xi|nlon|rlon|ni)[a-z0-9_]*$|^x?(nav_lon|lon|glam)[a-z0-9_]*$",
    re.IGNORECASE,
)
Y_COORD_PATTERN = re.compile(
    r"^(y|j|eta|nlat|rlat|nj)[a-z0-9_]*$|^y?(nav_lat|lat|gphi)[a-z0-9_]*$",
    re.IGNORECASE,
)

_GRID_CACHE = cachetools.LRUCache(maxsize=config["grid_cache_max_size"])


def _last_true_along_axis(mask: np.ndarray, axis: int, default: int) -> int:
    """Find the last True index along axis, or default if no True values."""
    reduced = cast(np.ndarray, mask.any(axis=1 - axis))
    if not reduced.any():
        return default
    return int(reduced.size - 1 - np.argmax(reduced[::-1]))


def _first_true_along_axis(mask: np.ndarray, axis: int, default: int) -> int:
    """Find the first True index along axis, or default if no True values."""
    reduced = mask.any(axis=1 - axis)
    if not reduced.any():
        return default
    return int(np.argmax(reduced))


def _grab_edges(
    left: np.ndarray,
    right: np.ndarray,
    *,
    slicer: slice,
    axis: int,
    size: int,
    increasing: bool,
) -> list:
    # bottom edge is inclusive; similar to IntervalIndex used in Rectilinear grids
    assert slicer.start <= slicer.stop, slicer
    if increasing:
        ys = [
            _last_true_along_axis(left <= slicer.stop, axis, 0),
            _first_true_along_axis(right > slicer.stop, axis, size),
            _last_true_along_axis(left <= slicer.start, axis, 0),
            _first_true_along_axis(right > slicer.start, axis, size),
        ]
    else:
        ys = [
            _first_true_along_axis(left < slicer.stop, axis, size),
            _last_true_along_axis(right >= slicer.stop, axis, 0),
            _first_true_along_axis(left < slicer.start, axis, size),
            _last_true_along_axis(right >= slicer.start, axis, 0),
        ]
    return ys


def _get_xy_pad(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_pad = numbagg.nanmax(np.abs(np.diff(x)))
    y_pad = numbagg.nanmax(np.abs(np.diff(y)))
    return x_pad, y_pad


def _convert_longitude_slice(
    lon_slice: slice, *, left_break: float, right_break: float
) -> tuple[slice, ...]:
    """
    Convert longitude slice to match the coordinate system of the dataset.

    Handles conversion between arbitrary longitude coordinate systems with different
    offsets (e.g., -180→180, 0→360, or rolled systems like 90→450).
    May return multiple slices when selection crosses dataset boundaries
    defined by `left_break` and `right_break`.

    Parameters
    ----------
    lon_slice : slice
        Input longitude slice (potentially in different coordinate system)
    left_break : float
        Left boundary of the coordinate system (e.g., -180, 0)
    right_break : float
        Right boundary of the coordinate system (e.g., 180, 360)

    Returns
    -------
    tuple[slice, ...]
        Converted slice(s) that match dataset's coordinate system.
        Returns tuple of slices when selection crosses boundaries or spans multiple periods.
    """
    if lon_slice.start is None or lon_slice.stop is None:
        raise ValueError("start and stop should not be None")

    assert lon_slice.step is None, lon_slice.step

    # https://github.com/developmentseed/morecantile/issues/175
    # the precision in morecantile tile bounds isn't perfect,
    # a good way to test is `tms.bounds(Tile(0,0,0))` which should
    # match the spec exactly: https://docs.ogc.org/is/17-083r4/17-083r4.html#toc48
    # Example: tests/test_pipeline.py::test_pipeline_tiles[-90->90,0->360-wgs84_prime_meridian(2/2/1)]

    # Keep original values to preserve floating point precision when computing
    # wrapped bounds. Doing start + 360 - 360 can lose precision (e.g.,
    # 179.99999999999997 + 360 = 540.0, then 540.0 - 360 = 180.0, not 179.999...)
    original_start, original_stop = lon_slice.start, lon_slice.stop
    assert original_start <= original_stop, lon_slice

    # Normalize start and stop to be within or near the grid's coordinate range
    # by shifting them by multiples of 360
    # This is required to deal with cases where unwrapping changed
    # the longitude coordinates to be continuous
    coord_center = (left_break + right_break) / 2
    query_center = (original_start + original_stop) / 2

    # Calculate shift to align query center with coordinate center
    shift = round((coord_center - query_center) / 360) * 360
    start = original_start + shift
    stop = original_stop + shift

    # Now handle boundary crossing cases
    # Grid: [L=======R] where L=left_break, R=right_break
    if stop <= left_break:
        # Both below left boundary, shift up
        # [start,stop]  [L=======R]  →  [L===[start,stop]===R]
        net_shift = shift + 360
        return (slice(original_start + net_shift, original_stop + net_shift),)

    elif start >= right_break:
        # Both above right boundary, shift down
        # [L=======R]  [start,stop]  →  [L===[start,stop]===R]
        net_shift = shift - 360
        return (slice(original_start + net_shift, original_stop + net_shift),)

    elif start < left_break and stop <= right_break:
        # Crosses left boundary, split into two slices
        # [start,L=====stop,R]  →  [L,stop] + [start+360,R]
        slices = []
        # Use original + net_shift to preserve precision
        wrapped_start = original_start + (shift + 360)
        if wrapped_start <= right_break:
            slices.append(slice(wrapped_start, right_break))
        if left_break <= stop:
            slices.append(slice(left_break, stop))
        return tuple(slices) if slices else (slice(left_break, right_break),)

    elif start >= left_break and stop > right_break:
        # Crosses right boundary, split into two slices
        # [L,start=====R,stop]  →  [start,R] + [L,stop-360]
        slices = []
        if start <= right_break:
            slices.append(slice(start, right_break))
        # Use original + net_shift to preserve precision (compute shift-360 first!)
        wrapped_stop = original_stop + (shift - 360)
        if left_break <= wrapped_stop:
            slices.append(slice(left_break, wrapped_stop))
        return tuple(slices) if slices else (slice(left_break, right_break),)

    else:
        # Both within valid range
        # [L===[start,stop]===R]
        return (slice(start, stop),)


def _padded_diff(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute differences along an axis and pad to match original array shape.
    This replaces np.gradient for computing spacing between points.

    Similar to np.gradient but uses forward differences instead of centered differences.
    The result represents the spacing to the next point, with the last value repeated.

    Parameters
    ----------
    arr : np.ndarray
        Input array
    axis : int
        Axis along which to compute differences. Default is -1 (last axis).

    Returns
    -------
    np.ndarray
        Array of same shape as input with padded differences.
        Uses forward differences with the last value repeated.
    """
    # Compute forward differences
    diff = np.diff(arr, axis=axis)

    # Create padding width specification
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, 1)

    # Pad by repeating the last slice along the axis
    return np.pad(diff, pad_width, mode="edge")


def _compute_interval_bounds(centers: np.ndarray) -> np.ndarray:
    """
    Compute interval bounds from cell centers, handling non-uniform spacing.
    Handles both increasing and decreasing coordinate arrays.

    Parameters
    ----------
    centers : np.ndarray
        Array of cell center coordinates

    Returns
    -------
    np.ndarray
        Array of bounds with length len(centers) + 1
    """
    size = centers.size
    if size < 2:
        raise ValueError("lat/lon vector with size < 2!")

    # Compute bounds as midpoints between centers
    bounds = np.empty(len(centers) + 1)

    # Interior bounds: midpoints between adjacent centers
    bounds[1:-1] = (centers[:-1] + centers[1:]) / 2

    # First and last bounds: extrapolate using the spacing
    bounds[0] = centers[0] - (centers[1] - centers[0]) / 2
    bounds[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2

    # Round to eliminate floating point cancellation errors
    # e.g., (-1.65 + 1.65)/2 produces 1.4e-14 instead of 0.0
    # 12 decimal places preserves sub-millimeter precision for geographic coords
    bounds = np.round(bounds, decimals=12)

    return bounds


class CellTreeIndex(xr.Index):
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        fill_value: Any,
        X: str,
        Y: str,
        lon_spans_globe: bool,
    ):
        self.X = X
        self.Y = Y
        if lon_spans_globe:
            # We want to reflect the vertices across the anti-meridian, and then run the triangulation.
            # For simplicity, we just append two copies of the convex hull the longitudes modified differently.
            # This could be wasteful when re-triangulating for large grids but for now, this is easy and works.
            # If we don't do pad before triangulating, we run in to the issue where the right edge vertices
            # are not connected to the left edge vertices, and the padding doesn't affect the rendering.
            # An alternative approach would be to construct the CellTreeIndex here, then figure out the faces that intersect
            # the line at 180, -180; calculate the neighbours of those faces; extract those vertices and pad those.
            # However this does not work if the boundary faces without padding don't intersect those bounding longitudes
            # e.g. consider if vertices are between -178.4 and +178.4, how do we connect the boundary without the convex hull approach?
            boundary = pd.unique(triangle.convex_hull(vertices).ravel())
            nverts = vertices.shape[0]
            pos_verts = vertices[boundary, ...]
            neg_verts = pos_verts.copy()

            pos_verts[:, 0] += 360
            neg_verts[:, 0] -= 360

            vertices = np.concatenate([vertices, pos_verts, neg_verts], axis=0)
            with log_duration("re-triangulating", "▲"):
                faces = triangle.delaunay(vertices)
            # need to reindex the data to match the padding
            self.reindexer = np.concatenate([np.arange(nverts), boundary, boundary])

        else:
            self.reindexer = None

        with log_duration("Creating CellTree", "⊠"):
            self.tree = CellTree2d(vertices, faces, fill_value=fill_value)

        if lon_spans_globe:
            with log_duration("Handling periodic boundaries", "⊠"):
                # lets find the vertices closest to the -180 & 180 boundaries and cache them.
                # At indexing time, we'll return the indexes for vertices at the boundary
                # so we can fix the coordinate discontinuity later in `pipeline`
                idx, face_indices, _ = self.tree.intersect_edges(
                    np.array([[[180, -90], [180, 90]], [[-180, -90], [-180, 90]]])
                )
                (breakpt,) = np.nonzero(np.diff(idx))
                assert breakpt.size == 1
                breakpt = breakpt[0] + 1
                verts = faces[face_indices, ...]
                self.antimeridian_vertices = {
                    "pos": pd.unique(verts[:breakpt].ravel()),
                    "neg": pd.unique(verts[breakpt:].ravel()),
                }
        else:
            self.antimeridian_vertices = {"pos": np.array([]), "neg": np.array([])}

    def sel(self, labels, method=None, tolerance=None) -> IndexSelResult:
        xidxr = labels.get(self.X)
        yidxr = labels.get(self.Y)

        assert isinstance(xidxr, slice)
        assert isinstance(yidxr, slice)

        with NUMBA_THREADING_LOCK:
            _, face_indices = self.tree.locate_boxes(
                np.array([[xidxr.start, xidxr.stop, yidxr.start, yidxr.stop]])
            )
        inverse, vertex_indices = pd.factorize(
            self.tree.faces[face_indices].ravel(), sort=True
        )
        inverse = inverse.reshape(face_indices.size, self.tree.faces.shape[-1])

        # Check if selected faces intersect the anti-meridian
        # Figure out which of the anti-meridian faces we've ended up selecting, and save those
        antimeridian_vertices = {}
        for key, anti_verts in self.antimeridian_vertices.items():
            present = np.intersect1d(vertex_indices, anti_verts, assume_unique=True)
            # indices are indexes in to `vertex_indices` i.e. the subset
            # note that vertex_indices are sorted
            antimeridian_vertices[key] = np.searchsorted(vertex_indices, present)

        # account for any padding needed for triangulations we are executing
        if self.reindexer is not None:
            vertex_indices = self.reindexer[vertex_indices]

        # Debugging plot to check indexing if needed
        # import matplotlib.pyplot as plt
        # from matplotlib.tri import Triangulation
        # tri = Triangulation(
        #     x=self.tree.vertices[vertex_indices, 0],
        #     y=self.tree.vertices[vertex_indices, 1],
        #     triangles=inverse,
        # )
        # plt.triplot(tri)
        # plt.plot(
        #     [xidxr.start, xidxr.stop, xidxr.stop, xidxr.start, xidxr.start],
        #     [yidxr.start, yidxr.start, yidxr.stop, yidxr.stop, yidxr.start],
        #     color='k',
        # )
        # plt.show()

        return IndexSelResult(
            dim_indexers={
                "ugrid": UgridIndexer(
                    vertices=vertex_indices,
                    connectivity=inverse,
                    antimeridian_vertices=antimeridian_vertices,
                )
            }
        )


def _detect_tripolar_fold_row(X: np.ndarray, *, yaxis: int) -> int | None:
    """Detect tripolar grid fold row by checking if the last row is a flipped version of the second-to-last.

    In a tripolar grid, the northernmost row of data is a flipped version of the row below it.
    This creates a "seam" at the North Pole where the grid folds back on itself.

    Parameters
    ----------
    X : np.ndarray
        2D longitude coordinate array
    yaxis : int
        Axis index for Y dimension

    Returns
    -------
    int | None
        Index of the fold row (the row that is flipped), or None if not a tripolar grid
    """
    n_rows = X.shape[yaxis]
    if n_rows < 2:
        return None

    last_row = np.take(X, -1, axis=yaxis)
    second_last_row = np.take(X, -2, axis=yaxis)
    second_last_flipped = np.flip(second_last_row)

    # Check if last row matches flipped second-to-last row.
    # Use a tolerance that accounts for floating-point precision in grid coordinates.
    # For float32 coordinates (common in HYCOM), precision errors can reach ~0.0001,
    # so use a slightly larger tolerance of 0.001 (~1 km at equator).
    if np.allclose(last_row, second_last_flipped, atol=0.001, rtol=0):
        return n_rows - 1

    return None


class CurvilinearCellIndex(xr.Index):
    left_break: float
    right_break: float
    Xdim: str
    Ydim: str
    X: xr.DataArray
    Y: xr.DataArray
    xaxis: int
    yaxis: int
    y_is_increasing: bool
    left: np.ndarray
    right: np.ndarray
    bottom: np.ndarray
    top: np.ndarray
    _dXmin: float
    _dYmin: float
    tripolar_fold_row: int | None

    def __init__(
        self,
        *,
        X: xr.DataArray,
        Y: xr.DataArray,
        Xdim: str,
        Ydim: str,
        tripolar_fold_row: int | None = None,
    ):
        if Y.nbytes > MAX_COORD_VAR_NBYTES or X.nbytes > MAX_COORD_VAR_NBYTES:
            raise ValueError(
                f"Coordinate variables {X.name!r} and {Y.name!r} are too big to load in to memory!"
            )
        self.X, self.Y = X.reset_coords(drop=True), Y.reset_coords(drop=True)
        self.Xdim, self.Ydim = Xdim, Ydim
        self.tripolar_fold_row = tripolar_fold_row

        # derived quantities
        X, Y = self.X.data, self.Y.data
        xaxis = self.X.get_axis_num(self.Xdim)
        yaxis = self.Y.get_axis_num(self.Ydim)

        # Store coordinate bounds for longitude slice conversion.
        # For geographic CRS, X coordinates are already unwrapped by Curvilinear.from_dataset,
        # so left_break/right_break represent the continuous range (e.g., 74° to 434°).
        # _convert_longitude_slice uses these to map tile requests to grid indices.
        # Use first/last columns along x-axis for the breaks.
        # For tripolar grids, exclude the fold row since it has distorted coordinates.
        first_col = np.take(X, 0, axis=xaxis)
        last_col = np.take(X, -1, axis=xaxis)
        if tripolar_fold_row is not None:
            first_col = first_col[:tripolar_fold_row]
            last_col = last_col[:tripolar_fold_row]
        self.left_break = float(np.min(first_col))
        self.right_break = float(np.max(last_col))

        # Compute cell bounds from coordinate diffs. For geographic CRS, X coordinates
        # are already unwrapped, so diffs are continuous (no 360° jumps at antimeridian).
        dX, dY = _padded_diff(X, axis=xaxis), _padded_diff(Y, axis=yaxis)
        self.left, self.right = X - dX / 2, X + dX / 2
        self.bottom, self.top = Y - dY / 2, Y + dY / 2
        # Determine if Y is increasing by checking the first row.
        # Don't use .all() because tripolar grids have dY=0 in the fold region
        # which would cause the swap to trigger incorrectly.
        first_row_bottom = np.take(self.bottom, 0, axis=yaxis)
        first_row_top = np.take(self.top, 0, axis=yaxis)
        self.y_is_increasing = (first_row_bottom < first_row_top).all()
        if not self.y_is_increasing:
            self.top, self.bottom = self.bottom, self.top
        self.xaxis, self.yaxis = xaxis, yaxis

        # Calculate minimum spacing using in-place ops (dX/dY no longer needed)
        np.abs(dX, out=dX)
        dX[dX == 0] = np.inf
        result = numbagg.nanmin(dX)
        self._dXmin = None if np.isinf(result) else float(result)

        np.abs(dY, out=dY)
        dY[dY == 0] = np.inf
        result = numbagg.nanmin(dY)
        self._dYmin = None if np.isinf(result) else float(result)

    def get_min_spacing(self) -> tuple[float, float]:
        """Get minimum spacing in X and Y directions.

        Returns
        -------
        tuple[float | None, float | None]
            (dXmin, dYmin) - minimum spacing in X and Y directions
        """
        return self._dXmin, self._dYmin

    def sel(self, labels, method=None, tolerance=None) -> IndexSelResult:
        X, Y = self.X.data, self.Y.data
        xaxis, yaxis = self.xaxis, self.yaxis
        bottom, top, left, right = self.bottom, self.top, self.left, self.right
        Xlen = X.shape[xaxis]
        # For tripolar grids, exclude the fold row from consideration
        Ylen = self.tripolar_fold_row or Y.shape[yaxis]

        assert len(labels) == 1
        bbox = next(iter(labels.values()))
        assert isinstance(bbox, BBox)

        slices = _convert_longitude_slice(
            slice(bbox.west, bbox.east),
            left_break=self.left_break,
            right_break=self.right_break,
        )

        ys = _grab_edges(
            bottom,
            top,
            slicer=slice(bbox.south, bbox.north),
            axis=yaxis,
            size=Ylen,
            increasing=self.y_is_increasing,
        )

        # add 1 to account for slice upper end being exclusive
        y_start, y_stop = min(ys), max(ys) + 1
        # Clamp Y slice to exclude tripolar fold row, if applicable
        y_stop = min(y_stop, Ylen)

        all_indexers: list[slice] = []
        for sl in slices:
            xs = _grab_edges(
                left, right, slicer=sl, axis=xaxis, size=Xlen, increasing=True
            )
            # add 1 to account for slice upper end being exclusive
            indexer = slice(min(xs), max(xs) + 1)
            start, stop, _ = indexer.indices(X.shape[xaxis])
            all_indexers.append(slice(start, stop))

        # Prevent overlap between adjacent slices
        all_indexers = _prevent_slice_overlap(all_indexers)

        slicers = {
            self.Xdim: all_indexers,
            self.Ydim: slice(y_start, y_stop),
        }
        return IndexSelResult(slicers)

    def equals(self, other: Self) -> bool:
        return (
            np.allclose(self.X.data, other.X.data)
            and np.allclose(self.Y.data, other.Y.data)
            and self.Xdim == other.Xdim
            and self.Ydim == other.Ydim
        )


class LongitudeCellIndex(xr.indexes.PandasIndex):
    _dXmin: float | None

    def __init__(self, interval_index: pd.IntervalIndex, dim: str):
        """
        Initialize LongitudeCellIndex with an IntervalIndex.

        Parameters
        ----------
        interval_index : pd.IntervalIndex
            The IntervalIndex representing cell bounds
        dim : str
            The dimension name
        """
        assert interval_index.closed == "left"
        super().__init__(interval_index, dim)
        self.index = interval_index
        self._xrindex = xr.indexes.PandasIndex(interval_index, dim)
        self._is_global = self._determine_global_coverage()

        # Store the actual coordinate bounds (left_break, right_break)
        left_bounds = self.index.left.values
        right_bounds = self.index.right.values
        self.left_break = float(left_bounds[0])
        self.right_break = float(right_bounds[-1])

        # Calculate and store minimum cell width
        widths = right_bounds - left_bounds
        self._dXmin = float(np.min(widths)) if len(widths) > 0 else None

    @property
    def cell_bounds(self) -> np.ndarray:
        """Get the cell bounds as an array."""
        return np.array([self.index.left.values, self.index.right.values]).T

    @property
    def cell_centers(self) -> np.ndarray:
        """Get the cell centers as an array."""
        return self.index.mid.values

    @property
    def is_global(self) -> bool:
        """Check if this longitude index covers the full globe."""
        return self._is_global

    def get_min_spacing(self) -> float | None:
        """Get minimum cell width (right - left).

        Returns
        -------
        float | None
            Minimum cell width
        """
        return self._dXmin

    def _determine_global_coverage(self) -> bool:
        """
        Determine if the longitude coverage is global.

        Returns True if the longitude spans nearly 360 degrees, indicating
        global coverage.
        """
        left_bounds = self.index.left.values
        right_bounds = self.index.right.values

        # Get the full span from leftmost left bound to rightmost right bound
        min_lon = left_bounds.min()
        max_lon = right_bounds.max()
        lon_span = max_lon - min_lon

        # Consider global if span is at least 359 degrees (allowing small tolerance)
        return lon_span >= 359.0

    def sel(self, labels, method=None, tolerance=None) -> IndexSelResult:
        """
        Select values from the longitude index with coordinate system conversion.

        Handles selection for both -180→180 and 0→360 longitude coordinate systems.
        Automatically converts coordinates when needed to match the dataset's convention.

        Parameters
        ----------
        labels : scalar or array-like
            Labels to select (can be scalar, slice, or array-like)
        method : str, optional
            Selection method (e.g., 'nearest', 'ffill', 'bfill')
        tolerance : optional
            Tolerance for inexact matches

        Returns
        -------
        Selection result with potentially multiple indexers for boundary crossing
        """
        # Handle slice objects specially for longitude coordinate conversion
        _, value = next(iter(labels.items()))
        if not isinstance(value, slice):
            raise NotImplementedError

        converted_slices = _convert_longitude_slice(
            value, left_break=self.left_break, right_break=self.right_break
        )

        # If we got multiple slices (for boundary crossing), create multiple indexers
        # Handle multiple slices by selecting each and creating multiple indexers
        all_indexers: list[slice] = []
        for slice_part in converted_slices:
            sel_dict = {self.dim: slice_part}
            result = self._xrindex.sel(sel_dict, method=method, tolerance=tolerance)
            indexer = next(iter(result.dim_indexers.values()))
            start, stop, _ = indexer.indices(len(self))
            all_indexers.append(slice(start, stop))

        # Prevent overlap between adjacent slices
        all_indexers = _prevent_slice_overlap(all_indexers)
        return IndexSelResult({self.dim: all_indexers})

    def __len__(self) -> int:
        """Return the length of the longitude index."""
        return len(self.index)


def is_rotated_pole(crs: CRS) -> bool:
    return crs.to_cf().get("grid_mapping_name") == "rotated_latitude_longitude"


def _is_raster_index_global(raster_index, grid_bbox, crs) -> bool:
    """
    Determine if a RasterIndex represents global longitude coverage.

    Parameters
    ----------
    raster_index : RasterIndex
        The raster index to check
    grid_bbox : BBox
        The grid's bounding box
    crs : CRS
        The coordinate reference system

    Returns
    -------
    bool
        True if the index covers the full globe
    """
    if not crs.is_geographic:
        return False

    # Check if longitude span is nearly 360 degrees
    lon_span = grid_bbox.east - grid_bbox.west
    return lon_span >= 359.0


def _indexes_equal(a: xr.Index, b: xr.Index) -> bool:
    """Compare two xarray indexes for approximate equality.

    For PandasIndex with IntervalIndex, uses np.allclose for floating point comparison.
    For other index types, falls back to the index's equals() method.
    """
    if type(a) is not type(b):
        return False

    if isinstance(a, xr.indexes.PandasIndex) and isinstance(b, xr.indexes.PandasIndex):
        if a.dim != b.dim:
            return False
        if len(a.index) != len(b.index):
            return False
        if isinstance(a.index, pd.IntervalIndex) and isinstance(
            b.index, pd.IntervalIndex
        ):
            if a.index.closed != b.index.closed:
                return False
            return np.allclose(a.index.left.values, b.index.left.values) and np.allclose(
                a.index.right.values, b.index.right.values
            )

    return a.equals(b)


@dataclass(eq=False)
class GridSystem(ABC):
    """
    Marker class for Grid Systems.

    Subclasses contain all information necessary to define the horizontal mesh,
    bounds, and reference frame for that specific grid system.
    """

    # Note: The following attributes are expected to be provided by subclasses
    # as dataclass fields or properties. They are typed here for type checking purposes.
    crs: CRS
    bbox: BBox
    X: str
    Y: str
    Xdim: str
    Ydim: str

    # FIXME: do we really need these Index objects on the class?
    #   - reconsider when we do curvilinear and triangular grids
    #   - The ugliness is that booth would have to set the right indexes on the dataset.
    #   - So this is do-able, but there's some strong coupling between the
    #     plugin and the "orchestrator"
    indexes: tuple[xr.Index, ...]
    Z: str | None = None
    alternates: tuple[GridMetadata, ...] = field(default_factory=tuple)

    dXmin: float = 0
    dYmin: float = 0

    @classmethod
    @abstractmethod
    def from_dataset(cls, ds: xr.Dataset, crs: CRS, Xname: str, Yname: str) -> Self:
        pass

    @property
    @abstractmethod
    def dims(self) -> set[str]:
        """Return the set of dimension names for this grid system."""
        pass

    def assign_index(self, da: xr.DataArray) -> xr.DataArray:
        return da

    def equals(self, other: Self) -> bool:
        if not isinstance(self, type(other)):
            return False
        if len(self.indexes) != len(other.indexes):
            return False
        if self.Z != other.Z:
            return False
        if len(self.alternates) != len(other.alternates):
            return False
        if any(
            not _indexes_equal(a, b)
            for a, b in zip(self.indexes, other.indexes, strict=False)
        ):
            return False
        if any(a != b for a, b in zip(self.alternates, other.alternates, strict=False)):
            return False
        if self.dXmin is None or other.dXmin is None:
            if self.dXmin != other.dXmin:
                return False
        elif not np.isclose(self.dXmin, other.dXmin):
            return False
        if self.dYmin is None or other.dYmin is None:
            if self.dYmin != other.dYmin:
                return False
        elif not np.isclose(self.dYmin, other.dYmin):
            return False
        return True

    def __eq__(self, other) -> bool:
        """Override dataclass __eq__ to use our custom equals() method."""
        if not isinstance(other, GridSystem):
            return False
        return self.equals(other)

    @abstractmethod
    def sel(self, *, bbox: BBox) -> dict[str, list[slice | Fill | UgridIndexer]]:
        """Select a subset of the data array using a bounding box."""
        raise NotImplementedError("Subclasses must implement sel method")

    def pick_alternate_grid(self, crs: CRS, *, coarsen_factors) -> GridMetadata:
        """Pick an alternate grid system based on the target CRS.

        Parameters
        ----------
        crs : CRS
            Target CRS to match against alternates

        Returns
        -------
        GridMetadata
            Matching alternate grid metadata, or this grid's metadata if no suitable alternate found
        """
        # For large coarsening, it is just faster to
        # transform the coarsened native coordinates
        if crs == self.crs or not self.alternates:
            return self.to_metadata()

        logger = get_context_logger()
        if all(factor > 2 for factor in coarsen_factors.values()):
            logger.debug("🚫 large coarsening, skipping alternate coordinates")
            return self.to_metadata()

        # Check if any alternate grid has a matching CRS
        for alt in self.alternates:
            if alt.crs == crs:
                logger.debug(f"🔀 picking alternate grid system: {alt!r}")
                return alt

        # Check if any alternate grid is 4326-like
        for alt in self.alternates:
            if is_4326_like(alt.crs):
                logger.debug(f"🔀 picking alternate grid system: {alt!r}")
                return alt

        return self.to_metadata()

    def to_metadata(self) -> GridMetadata:
        """Convert this GridSystem to GridMetadata.

        Returns
        -------
        GridMetadata
            Metadata representation of this grid system
        """
        return GridMetadata(X=self.X, Y=self.Y, crs=self.crs, grid_cls=type(self))


class RectilinearSelMixin:
    """Mixin for generic rectilinear .sel"""

    def sel(
        self, *, bbox: BBox, y_is_increasing: bool, x_size: int, y_size: int
    ) -> dict[str, list[slice | Fill | UgridIndexer]]:
        """
        This method handles coordinate selection for rectilinear grids, automatically
        converting between different longitude conventions (0→360 vs -180→180).

        Parameters
        ----------
        bbox : BBox
            Bounding box for selection
        y_is_increasing : bool
            Whether Y coordinates are increasing
        x_size : int
            Size of X dimension
        y_size : int
            Size of Y dimension
        x_handle_wraparound : bool
            Whether to handle wraparound for X dimension
        """
        assert len(self.indexes) >= 1
        xindex, yindex = self.indexes[0], self.indexes[-1]

        # Handle Y dimension selection
        if y_is_increasing:
            yslice = yindex.sel({self.Y: slice(bbox.south, bbox.north)}).dim_indexers[
                self.Y
            ]
        else:
            yslice = yindex.sel({self.Y: slice(bbox.north, bbox.south)}).dim_indexers[
                self.Y
            ]

        # Handle X dimension selection
        xsel_result = xindex.sel({self.X: slice(bbox.west, bbox.east)})

        # Prepare slicers for padding (ensure lists of slices for consistency)
        # X dimension: LongitudeCellIndex can return multiple slices for antimeridian crossing
        x_raw = xsel_result.dim_indexers[self.X]

        # Handle single slice from PandasIndex
        x_indexers = x_raw if isinstance(x_raw, list) else [x_raw]

        # Y dimension: always a single slice from PandasIndex
        y_indexers = [yslice]

        slicers = {self.X: x_indexers, self.Y: y_indexers}

        return slicers


@dataclass(kw_only=True, eq=False)
class RasterAffine(RectilinearSelMixin, GridSystem):
    """2D horizontal grid defined by an affine transform."""

    crs: CRS
    bbox: BBox
    X: str
    Y: str
    Xdim: str = field(init=False)
    Ydim: str = field(init=False)
    indexes: tuple[rasterix.RasterIndex]
    Z: str | None = None
    lon_spans_globe: bool = field(init=False)
    dXmin: float = field(init=False)
    dYmin: float = field(init=False)

    def __post_init__(self) -> None:
        self.Xdim = self.X
        self.Ydim = self.Y
        # Determine if this raster spans the globe in longitude
        self.lon_spans_globe = self.crs.is_geographic and _is_raster_index_global(
            self.indexes[0], self.bbox, self.crs
        )
        # Calculate minimum grid spacing from affine transform
        (index,) = self.indexes
        affine = index.transform()
        self.dXmin = abs(affine.a)  # X pixel size
        self.dYmin = abs(affine.e)  # Y pixel size

    @classmethod
    def from_dataset(
        cls,
        ds: xr.Dataset,
        crs: CRS,
        Xname: str,
        Yname: str,
    ) -> "RasterAffine":
        """Create a RasterAffine grid from a dataset using rasterix."""
        ds = rasterix.assign_index(ds, x_dim=Xname, y_dim=Yname)
        index = ds.xindexes[Xname]
        # After assign_index, the index should be a RasterIndex
        raster_index = cast(rasterix.RasterIndex, index)
        return cls(
            crs=crs,
            X=Xname,
            Y=Yname,
            bbox=BBox(
                west=raster_index.bbox.left,
                east=raster_index.bbox.right,
                south=raster_index.bbox.bottom,
                north=raster_index.bbox.top,
            ),
            indexes=(raster_index,),
        )

    @property
    def dims(self) -> set[str]:
        """Return the set of dimension names for this grid system."""
        return {self.Xdim, self.Ydim}

    def assign_index(self, da: xr.DataArray) -> xr.DataArray:
        (index,) = self.indexes
        return da.assign_coords(xr.Coordinates.from_xindex(index))

    def sel(self, *, bbox: BBox) -> dict[str, list[slice | Fill]]:
        (index,) = self.indexes
        affine = index.transform()

        return super().sel(
            bbox=bbox,
            y_is_increasing=affine.e > 0,
            x_size=index._xy_shape[0],
            y_size=index._xy_shape[1],
        )

    def equals(self, other: Self) -> bool:
        if (self.crs == other.crs and self.bbox == other.bbox) or (
            self.X == other.X and self.Y == other.Y
        ):
            return super().equals(other)
        else:
            return False


@dataclass(kw_only=True, eq=False)
class Rectilinear(RectilinearSelMixin, GridSystem):
    """
    2D horizontal grid defined by two explicit 1D basis vectors.
    Assumes coordinates are cell centers.
    """

    crs: CRS
    bbox: BBox
    X: str
    Y: str
    Xdim: str = field(init=False)
    Ydim: str = field(init=False)
    indexes: tuple[xr.indexes.PandasIndex | LongitudeCellIndex, xr.indexes.PandasIndex]
    Z: str | None = None
    lon_spans_globe: bool = field(init=False)
    dXmin: float = field(init=False)
    dYmin: float = field(init=False)
    left_break: float = field(init=False)
    right_break: float = field(init=False)

    def __post_init__(self) -> None:
        self.Xdim = self.X
        self.Ydim = self.Y
        # Determine if this grid spans the globe in longitude
        if (
            self.indexes
            and self.crs.is_geographic
            and isinstance(self.indexes[0], LongitudeCellIndex)
        ):
            self.lon_spans_globe = self.indexes[0].is_global
        else:
            self.lon_spans_globe = False

        # Calculate minimum grid spacing and coordinate bounds from indexes
        if self.indexes:
            x_index = self.indexes[0]
            if isinstance(x_index, LongitudeCellIndex):
                self.dXmin = x_index.get_min_spacing()
                self.left_break = x_index.left_break
                self.right_break = x_index.right_break
            else:
                assert isinstance(x_index, xr.indexes.PandasIndex)
                left_bounds = x_index.index.left.values
                right_bounds = x_index.index.right.values
                widths = right_bounds - left_bounds
                self.dXmin = float(np.min(widths)) if len(widths) > 0 else None
                self.left_break = float(left_bounds[0])
                self.right_break = float(right_bounds[-1])

        if len(self.indexes) > 1:
            y_index = self.indexes[1]
            assert isinstance(y_index, xr.indexes.PandasIndex)
            widths = y_index.index.right.values - y_index.index.left.values
            self.dYmin = float(np.min(widths)) if len(widths) > 0 else None

    @property
    def dims(self) -> set[str]:
        """Return the set of dimension names for this grid system."""
        return {self.Xdim, self.Ydim}

    @classmethod
    def from_dataset(
        cls,
        ds: xr.Dataset,
        crs: CRS,
        Xname: str,
        Yname: str,
    ) -> "Rectilinear":
        """Create a Rectilinear grid from a dataset with cell-center adjusted bbox."""
        X = ds[Xname]
        Y = ds[Yname]

        x_data = X.data
        if crs.is_geographic:
            # Use np.unwrap to fix longitude discontinuities without centering
            # This ensures datasets with the meridian discontinuitiy in the "middle"
            # of the dataset maintain a continuous coordinate system
            x_data = unwrap(x_data, width=360)

        x_bounds = _compute_interval_bounds(x_data)
        x_intervals = pd.IntervalIndex.from_breaks(x_bounds, closed="left")
        if crs.is_geographic:
            x_index = LongitudeCellIndex(x_intervals, Xname)
        else:
            x_index = xr.indexes.PandasIndex(x_intervals, Xname)

        y_bounds = _compute_interval_bounds(Y.data)
        if Y.data[-1] > Y.data[0]:
            y_intervals = pd.IntervalIndex.from_breaks(y_bounds, closed="left")
        else:
            y_intervals = pd.IntervalIndex.from_breaks(y_bounds[::-1], closed="right")[
                ::-1
            ]
        y_index = xr.indexes.PandasIndex(y_intervals, Yname)

        west = np.round(float(x_bounds[0]), 3)
        east = np.round(float(x_bounds[-1]), 3)
        south = np.round(float(y_bounds[0]), 3)
        north = np.round(float(y_bounds[-1]), 3)
        south, north = min(south, north), max(south, north)

        if crs.is_geographic:
            # Handle global datasets - always normalize to -180, 180 for consistency
            x_span = east - west
            if x_span >= 359.0:  # Nearly global in longitude
                west, east = -180, 180
            south = max(-90, south)
            north = min(90, north)

        bbox = BBox(west=west, east=east, south=south, north=north)
        return cls(
            crs=crs,
            X=Xname,
            Y=Yname,
            bbox=bbox,
            indexes=(x_index, y_index),
        )

    def sel(self, *, bbox: BBox) -> dict[str, list[slice | Fill]]:
        """
        Select a subset of the data array using a bounding box.
        """
        x_index, y_index = self.indexes[0], self.indexes[-1]

        # For Rectilinear grids, X index is always LongitudeCellIndex (geographic)
        # or PandasIndex (non-geographic)
        if self.crs.is_geographic:
            # Geographic CRS should always have LongitudeCellIndex
            assert isinstance(x_index, LongitudeCellIndex), (
                f"Expected LongitudeCellIndex for geographic CRS, got {type(x_index)}"
            )
        else:
            # Non-geographic CRS should have regular PandasIndex
            assert isinstance(x_index, xr.indexes.PandasIndex), (
                f"Expected PandasIndex for non-geographic CRS, got {type(x_index)}"
            )

        # Both index types have len() method
        x_size = len(x_index.index)
        y_size = len(y_index.index)
        y_index_cast = cast(xr.indexes.PandasIndex, y_index)

        return super().sel(
            bbox=bbox,
            y_is_increasing=y_index_cast.index.is_monotonic_increasing,
            x_size=x_size,
            y_size=y_size,
        )

    def equals(self, other: Self) -> bool:
        if (self.crs == other.crs and self.bbox == other.bbox) or (
            self.X == other.X and self.Y == other.Y
        ):
            return super().equals(other)
        else:
            return False


_CURVILINEAR_BBOX_SENTINEL = BBox(west=0, east=0, south=0, north=0)


@dataclass(kw_only=True, eq=False)
class Curvilinear(GridSystem):
    """2D horizontal grid defined by two 2D arrays."""

    crs: CRS
    # This sentinel is purely so that I don't use BBox | None on the type
    # and so we can provide "expected" values in tests
    bbox: BBox = field(default_factory=lambda: _CURVILINEAR_BBOX_SENTINEL)
    X: str
    Y: str
    Xdim: str
    Ydim: str
    indexes: tuple[xr.Index, ...]
    Z: str | None = None
    lon_spans_globe: bool = field(init=False)
    dXmin: float = field(init=False)
    dYmin: float = field(init=False)

    def __post_init__(self) -> None:
        index = next(iter(self.indexes))
        assert isinstance(index, CurvilinearCellIndex)

        # Compute bbox from index if not provided (sentinel)
        if self.bbox is _CURVILINEAR_BBOX_SENTINEL:
            west = numbagg.nanmin(index.left)
            east = numbagg.nanmax(index.right)
            south = numbagg.nanmin(index.bottom)
            north = numbagg.nanmax(index.top)

            if self.crs.is_geographic:
                # This handling is necessary for subsets of tripolar grids;
                # where the anti-meridian discontinuity is not present in some, but not all, rows.
                lon_span = (
                    index.right.max(axis=index.xaxis) - index.left.min(axis=index.xaxis)
                ).max()
                self.lon_spans_globe = lon_span >= 350
                # Normalize bbox to -180->180 for global grids
                if self.lon_spans_globe:
                    west, east = -180, 180
            else:
                self.lon_spans_globe = False

            self.bbox = BBox(west=west, east=east, south=south, north=north)
        else:
            self.lon_spans_globe = (
                self.crs.is_geographic and (self.bbox.west - self.bbox.east) >= 350
            )

        # Calculate minimum grid spacing using index method
        self.dXmin, self.dYmin = index.get_min_spacing()

    @property
    def is_tripolar(self) -> bool:
        index = next(iter(self.indexes))
        return getattr(index, "tripolar_fold_row", None) is not None

    def _guess_dims(
        ds: xr.Dataset, *, X: xr.DataArray, Y: xr.DataArray
    ) -> tuple[str, str]:
        # Get the dimension names using cf.axes
        # For curvilinear grids, we need to find the dimensions that map to X and Y axes
        axes = ds.cf.axes

        # Find X and Y dimensions - these are the dimensions of the 2D coordinate arrays
        # that correspond to the logical X and Y axes
        Xdim_candidates = axes.get("X", [])
        Ydim_candidates = axes.get("Y", [])

        # Filter to only dimensions that are in the 2D coordinate arrays
        valid_dims = set(X.dims)
        Xdim = next((str(d) for d in Xdim_candidates if d in valid_dims), None)
        Ydim = next((str(d) for d in Ydim_candidates if d in valid_dims), None)

        # If we couldn't identify from cf.axes, try guess_coord_axis
        if not Xdim or not Ydim:
            # Try to guess coordinate axes
            ds = ds.cf.guess_coord_axis()
            axes = ds.cf.axes
            Xdim_candidates = axes.get("X", [])
            Ydim_candidates = axes.get("Y", [])

            # Filter to only dimensions that are in X.dims
            Xdim = next((str(d) for d in Xdim_candidates if d in valid_dims), None)
            Ydim = next((str(d) for d in Ydim_candidates if d in valid_dims), None)

            # Final fallback: try pattern matching on dimension names
            if not Xdim or not Ydim:
                for dim in X.dims:
                    dim_str = str(dim)
                    if X_COORD_PATTERN.match(dim_str) and not Xdim:
                        Xdim = dim_str
                    elif Y_COORD_PATTERN.match(dim_str) and not Ydim:
                        Ydim = dim_str

            # If we still can't identify, raise an error
            if not Xdim or not Ydim:
                raise RuntimeError(
                    f"Could not identify X and Y dimensions for curvilinear grid. "
                    f"Coordinate dimensions are {list(X.dims)}, but could not determine "
                    f"which corresponds to X and which to Y axes. "
                    f"Please ensure your dataset has proper CF axis attributes or add SGRID metadata."
                )
        return Xdim, Ydim

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, crs: CRS, Xname: str, Yname: str) -> Self:
        # Cast coordinates to float64 to avoid repeated conversions during transforms
        # Note that pyproj requires float64 C-contiguous arrays for in-place transforms.
        # Since Curvilinear grid transforms are quite expensive; we trade the memory cost
        # here to avoid repeated allocations when transforming.
        X, Y = ds[Xname].astype(np.float64), ds[Yname].astype(np.float64)
        Xdim, Ydim = Curvilinear._guess_dims(ds, X=X, Y=Y)
        yaxis = X.get_axis_num(Ydim)

        # Detect tripolar grid BEFORE unwrapping (the fold row detection relies on
        # the last row being a flipped version of the second-to-last, which only
        # holds true before unwrap modifies the coordinates)
        tripolar_fold_row = _detect_tripolar_fold_row(X.data, yaxis=yaxis)

        if crs.is_geographic:
            # Use np.unwrap to fix longitude discontinuities without centering
            # This ensures datasets with the meridian discontinuity in the "middle"
            # of the dataset maintain a continuous coordinate system
            X = X.copy(data=unwrap(X.data, width=360))

        index = CurvilinearCellIndex(
            X=X,
            Y=Y,
            Xdim=Xdim,
            Ydim=Ydim,
            tripolar_fold_row=tripolar_fold_row,
        )
        return cls(
            crs=crs,
            X=Xname,
            Y=Yname,
            Xdim=Xdim,
            Ydim=Ydim,
            indexes=(index,),
        )

    @property
    def dims(self) -> set[str]:
        """Return the set of dimension names for this grid system."""
        return {self.Xdim, self.Ydim}

    def equals(self, other: Self) -> bool:
        if (
            (self.crs == other.crs and self.bbox == other.bbox)
            and (self.X == other.X and self.Y == other.Y)
            and self.indexes[0].equals(other.indexes[0])
        ):
            return super().equals(other)
        else:
            return False

    def sel(self, *, bbox: BBox) -> dict[str, list[slice | Fill | UgridIndexer]]:
        """
        Select a subset of the data array using a bounding box.

        Uses masking to select out the bbox for curvilinear grids where coordinates
        are 2D arrays. Also normalizes longitude coordinates to -180→180 format.
        """
        # Uses masking to select out the bbox, following the discussion in
        # https://github.com/pydata/xarray/issues/10572
        index = next(iter(self.indexes))
        assert isinstance(index, CurvilinearCellIndex), (
            f"Expected CurvilinearCellIndex, got {type(index)}"
        )

        sel_result = index.sel({self.Xdim: bbox})

        # Get slicers for both dimensions (ensure they are lists of slices)
        # X dimension: CurvilinearCellIndex returns list[slice] for antimeridian crossing
        x_raw = sel_result.dim_indexers[self.Xdim]
        xslicers = x_raw if isinstance(x_raw, list) else list(x_raw)

        # Y dimension: CurvilinearCellIndex always returns a single slice
        y_raw = sel_result.dim_indexers[self.Ydim]
        yslicers = [y_raw]  # Always a single slice

        return {self.Xdim: xslicers, self.Ydim: yslicers}


@dataclass(init=False, kw_only=True, eq=False)
class Triangular(GridSystem):
    crs: CRS
    bbox: BBox

    X: str
    Y: str
    Z: str | None = None
    dim: str
    lon_spans_globe: bool
    indexes: tuple[xr.Index]

    # these make no sense
    dXmin: float = 0
    dYmin: float = 0

    def __init__(
        self,
        *,
        vertices: np.ndarray,
        faces: np.ndarray,
        crs: CRS,
        dim: str,
        Xname: str,
        Yname: str,
        fill_value: Any,
    ):
        self.crs = crs
        self.X, self.Y = Xname, Yname
        self.dim = dim
        xmin, xmax = vertices[:, 0].min(), vertices[:, 0].max()
        ymin, ymax = vertices[:, 1].min(), vertices[:, 1].max()
        # This "350" business is nonsense; we need a way to figure out if a grid has global coverage
        # but that's basically impossible if all you have are vertices.
        self.lon_spans_globe = crs.is_geographic and ((xmax - xmin) > 350)
        if self.lon_spans_globe:
            self.bbox = BBox(west=-180, east=180, south=ymin, north=ymax)
        else:
            self.bbox = BBox(west=xmin, east=xmax, south=ymin, north=ymax)
        self.indexes = (
            CellTreeIndex(
                vertices,
                faces,
                fill_value=fill_value,
                X=Xname,
                Y=Yname,
                lon_spans_globe=self.lon_spans_globe,
            ),
        )

    @property
    def Xdim(self) -> str:
        return self.dim

    @property
    def Ydim(self) -> str:
        return self.dim

    @property
    def dims(self) -> set[str]:
        return {self.dim}

    def sel(self, *, bbox: BBox) -> dict[str, list[slice | Fill | UgridIndexer]]:
        index = next(iter(self.indexes))
        result = index.sel(
            {self.X: slice(bbox.west, bbox.east), self.Y: slice(bbox.south, bbox.north)}
        )
        # Extract the UgridIndexer from the IndexSelResult
        ugrid_indexer = result.dim_indexers["ugrid"]
        assert isinstance(ugrid_indexer, UgridIndexer)
        return {self.dim: [ugrid_indexer]}

    @classmethod
    def from_dataset(
        cls,
        ds: xr.Dataset,
        crs: CRS,
        Xname: str,
        Yname: str,
    ) -> Self:
        # FIXME: detect UGRID here
        vertices = (
            ds.reset_coords()[[Xname, Yname]]
            .to_dataarray("variable")
            .transpose(..., "variable")
            .data
        )
        assert vertices.shape[-1] == 2, (
            f"Attempting to triangulate vertices with shape={vertices.shape}. Expected (n_points, 2)"
        )
        if crs.is_geographic:
            # TODO: consider normalizing these to the unit sphere like UXarray
            # normalize to -180<=grid.X<180
            vertices[:, 0] = ((vertices[:, 0] + 180) % 360) - 180

        (dim,) = ds[Xname].dims
        with log_duration("Triangulating", "🔺"):
            if numbagg.anynan(vertices):
                raise ValueError(
                    f"Triangulation failed. Variables {Xname!r} or {Yname!r} contain NaNs."
                )
            try:
                faces = triangle.delaunay(vertices)
            except Exception as e:
                raise ValueError(
                    f"Triangulation failed. This may indicate bad data in variables {Xname!r}, {Yname!r}."
                    f"Please check whether all values are the same. "
                    f"Original exception: {e!r}"
                ) from None

        return cls(
            vertices=vertices,
            faces=faces,
            crs=crs,
            Xname=Xname,
            Yname=Yname,
            dim=dim,
            fill_value=-1,
        )


# Type alias for 1D grid systems
GridSystem1D = Triangular
# Type alias for 2D grid systems that have X, Y, and crs attributes
GridSystem2D = RasterAffine | Rectilinear | Curvilinear


def _guess_grid_mappings_and_crs(
    ds: xr.Dataset,
) -> list[GridMappingInfo]:
    """
    Returns all grid mappings, CRS pairs, and coordinate pairs using new cf-xarray API.

    Returns
    -------
    list[GridMappingInfo]
        List of (grid_mapping variable, CRS, (coordinates...)) tuples
    """
    grid_mappings = ds.cf.grid_mappings
    if grid_mappings:
        result = []
        for grid_mapping_obj in grid_mappings:
            grid_mapping_var = grid_mapping_obj.array
            crs = grid_mapping_obj.crs
            # Get the coordinates specified in the grid mapping
            coords = grid_mapping_obj.coordinates or None
            coordinates = (
                coords if coords and len(coords) == 2 and coords != ([], []) else None
            )
            result.append(GridMappingInfo(grid_mapping_var, crs, coordinates))
        return result

    # Fall back to existing single grid mapping approach - construct default grid mapping
    grid_mapping_names: tuple[str, ...] = ()
    if "spatial_ref" in ds.variables:
        grid_mapping_names += ("spatial_ref",)
    elif "crs" in ds.variables:
        grid_mapping_names += ("crs",)

    if len(grid_mapping_names) == 0:
        keys = ds.cf.keys()
        if "latitude" in keys and "longitude" in keys:
            # Construct a default geographic grid mapping
            Xname, Yname = guess_coordinate_vars(ds, DEFAULT_CRS)
            coordinates = (
                tuple(itertools.chain(Xname, Yname)) if Xname and Yname else None
            )
            return [GridMappingInfo(None, DEFAULT_CRS, coordinates)]
        else:
            warnings.warn("No CRS detected", UserWarning, stacklevel=2)
            return [GridMappingInfo(None, None, None)]

    # Handle case where spatial_ref is present but not linked to by a grid_mapping attribute
    result = []
    for grid_mapping_var in grid_mapping_names:
        grid_mapping = ds[grid_mapping_var]
        crs = CRS.from_cf(grid_mapping.attrs)
        # For legacy approach, we don't have explicit coordinate info, so pass None
        result.append(GridMappingInfo(grid_mapping, crs, None))
    return result


def _guess_grid_mapping_and_crs(
    ds: xr.Dataset,
) -> tuple[xr.DataArray | None, CRS | None]:
    """
    Returns the first grid mapping and CRS (backwards compatibility).

    Returns
    -------
    grid_mapping variable
    CRS
    """
    all_mappings = _guess_grid_mappings_and_crs(ds)
    return (
        (all_mappings[0].grid_mapping, all_mappings[0].crs)
        if all_mappings
        else (None, None)
    )


def guess_coordinate_vars(
    ds: xr.Dataset, crs: CRS
) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
    if is_rotated_pole(crs):
        stdnames = ds.cf.standard_names
        Xname, Yname = (
            stdnames.get("grid_longitude", None),
            stdnames.get("grid_latitude", None),
        )
    elif crs.is_geographic:
        coords = ds.cf.coordinates
        Xname, Yname = coords.get("longitude", None), coords.get("latitude", None)
    else:
        axes = ds.cf.axes
        Xname, Yname = axes.get("X", None), axes.get("Y", None)
    return Xname, Yname


def _filter_coordinates(
    coords: tuple[str, ...] | None, skip_coordinates: set[str]
) -> tuple[str, ...] | None:
    """Filter out coordinates that should be skipped. Returns None if no coordinates remain."""
    if coords is None:
        return None
    filtered = tuple(coord for coord in coords if coord not in skip_coordinates)
    return filtered or None


def _guess_coordinates_for_mapping(
    ds: xr.Dataset,
    mapping: GridMappingInfo,
    skip_coordinates: set[str],
) -> tuple[str | None, str | None]:
    """
    Shared logic to guess X and Y coordinate variables for a grid mapping.
    """
    assert mapping.crs is not None, "CRS must not be None at this point"

    if mapping.coordinates:
        # Use explicit coordinate pair if provided
        Xname, Yname = guess_coordinate_vars(
            ds.reset_coords()[list(mapping.coordinates)].set_coords(
                list(mapping.coordinates)
            ),
            mapping.crs,
        )
    else:
        # No explicit coordinates, guess from full dataset
        Xname, Yname = guess_coordinate_vars(ds, mapping.crs)
        if Xname is None or Yname is None:
            # FIXME: Can we be a little more targeted in what we are guessing?
            ds = ds.cf.guess_coord_axis()
            Xname, Yname = guess_coordinate_vars(ds, mapping.crs)

    # Apply coordinate filtering, we don't want to have guessed coordinate vars
    # that belong to *other* grid mapping variables
    Xname = _filter_coordinates(Xname, skip_coordinates)
    Yname = _filter_coordinates(Yname, skip_coordinates)

    if Xname is None or Yname is None:
        return None, None

    if len(Xname) > 1 or (len(Yname) > 1 and len(ds.data_vars) == 1):
        first = next(iter(ds.data_vars.values()))
        Xname = [x for x in Xname if x in first.attrs.get("coordinates", [])]
        Yname = [y for y in Yname if y in first.attrs.get("coordinates", [])]

    if len(Xname) > 1 or len(Yname) > 1:
        raise RuntimeError(
            f"Multiple coordinate options found for grid mapping: {Xname=!r}, {Yname=!r}."
        )

    return Xname[0], Yname[0]


def _detect_grid_metadata(
    ds: xr.Dataset,
    mapping: GridMappingInfo,
    skip_coordinates: set[str],
) -> GridMetadata | None:
    """
    Create a GridMetadata for a specific CRS mapping.
    """
    assert mapping.crs is not None, "CRS must not be None"

    Xname, Yname = _guess_coordinates_for_mapping(ds, mapping, skip_coordinates)

    if Xname is None or Yname is None:
        # Handle the fallback case where coordinates can't be determined normally
        if mapping.grid_mapping is None:
            raise RuntimeError(
                "Creating raster affine grid system failed. "
                "No explicit coordinate variables were detected and "
                "no grid_mapping variable was detected."
            )

        if "GeoTransform" not in mapping.grid_mapping.attrs:
            # Return None to indicate no GeoTransform available
            raise RuntimeError(
                "Creating raster affine grid system failed. "
                "No explicit coordinate variables were detected and "
                "no GeoTransform attribute is present on "
                f"grid mapping variable: {mapping.grid_mapping!r}"
            )

        # Use regex patterns to find coordinate dimensions
        x_dim, y_dim = None, None
        for dim in ds.dims:
            dim = cast(str, dim)
            if x_dim is None and X_COORD_PATTERN.match(dim):
                x_dim = dim
            if y_dim is None and Y_COORD_PATTERN.match(dim):
                y_dim = dim
        if not (x_dim and y_dim):
            raise RuntimeError(
                "Creating raster affine grid system failed. "
                "No explicit coordinate variables were detected and "
                "no x or y dimensions could be inferred. "
            )
        Xname, Yname = x_dim, y_dim
        grid_cls = RasterAffine
    else:
        # Determine the appropriate grid class based on coordinate structure
        X = ds[Xname]
        Y = ds[Yname]

        if X.ndim == 1 and Y.ndim == 1:
            if is_rotated_pole(mapping.crs):
                raise NotImplementedError("Rotated pole grids are not supported yet.")
            grid_cls = Triangular if X.dims == Y.dims else Rectilinear
        elif X.ndim == 2 and Y.ndim == 2:
            grid_cls = Curvilinear

    return GridMetadata(X=Xname, Y=Yname, crs=mapping.crs, grid_cls=grid_cls)


@time_debug
def _guess_grid_for_dataset(ds: xr.Dataset) -> GridSystem:
    """
    Does some grid_mapping & CRS auto-guessing with support for multiple grid mappings.

    Raises RuntimeError to indicate that we might try again.
    """
    all_mappings = _guess_grid_mappings_and_crs(ds)
    if not all_mappings or all_mappings[0].crs is None:
        raise RuntimeError("CRS/grid system not detected")

    # Create primary grid system from first mapping
    primary_mapping = all_mappings[0]
    # make sure we don't detect coordinates referred to by OTHER grid_mapping variables
    skip_coordinates = set(
        itertools.chain(*(mapping.coordinates or [] for mapping in all_mappings[1:]))
    )
    primary_grid_metadata = _detect_grid_metadata(ds, primary_mapping, skip_coordinates)
    if primary_grid_metadata is None:
        raise RuntimeError("CRS/grid system not detected")
    primary_grid = primary_grid_metadata.grid_cls.from_dataset(
        ds, primary_grid_metadata.crs, primary_grid_metadata.X, primary_grid_metadata.Y
    )

    # Create alternate grid systems from remaining mappings
    alternates = []
    for mapping in all_mappings[1:]:
        try:
            alternate_grid = _detect_grid_metadata(ds, mapping, set())
            if alternate_grid is not None:
                alternates.append(alternate_grid)
            else:
                raise RuntimeError("Could not detect grid metadata")
        except RuntimeError as e:
            # Skip grid systems that can't be created but warn about it
            grid_mapping_name = (
                mapping.grid_mapping.name
                if mapping.grid_mapping is not None
                else "unknown"
            )
            warnings.warn(
                f"Could not create alternate grid for grid mapping '{grid_mapping_name}': {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

    # Update primary grid with alternates
    # Since dataclass is frozen=False for base class, we can modify alternates directly
    primary_grid.alternates = tuple(alternates)

    return primary_grid


def _guess_z_dimension(da: xr.DataArray) -> str | None:
    # make sure Z is a dimension we can select on
    # We have to do this here to deal with the try-except above.
    # In the except clause, we might detect multiple Z.
    possible = set(da.cf.coordinates.get("vertical", {})) | set(da.cf.axes.get("Z", {}))
    for z in sorted(possible):
        if z in da.dims:
            return z
    return None


def guess_grid_system(ds: xr.Dataset, name: Hashable) -> GridSystem:
    """
    Guess the grid system for a dataset.

    Uses caching with ds.attrs['_xpublish_id'] as cache key if present.
    If no _xpublish_id, skips caching to avoid cross-contamination.
    """
    xpublish_id = ds.attrs.get("_xpublish_id")
    cache_key = (xpublish_id, name) if xpublish_id is not None else None

    if cache_key is not None and cache_key in _GRID_CACHE:
        return _GRID_CACHE[cache_key]

    with GRID_DETECTION_LOCK:
        # Double-check in case another thread populated cache while we waited
        if cache_key is not None and cache_key in _GRID_CACHE:
            return _GRID_CACHE[cache_key]

        try:
            grid = _guess_grid_for_dataset(ds.cf[[name]])
        except RuntimeError:
            try:
                grid = _guess_grid_for_dataset(ds)
            except RuntimeError:
                ds = ds.cf.guess_coord_axis()
                grid = _guess_grid_for_dataset(ds)
        except KeyError:
            raise VariableNotFoundError(
                f"Variable {name!r} not found in dataset."
            ) from None

        grid.Z = _guess_z_dimension(ds.cf[name])

        if cache_key is not None:
            _GRID_CACHE[cache_key] = grid

        return grid
