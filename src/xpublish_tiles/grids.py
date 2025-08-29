import itertools
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Self, cast

import cachetools
import cf_xarray  # noqa: F401
import numbagg
import numpy as np
import pandas as pd
import rasterix
from pyproj import CRS
from pyproj.aoi import BBox

import xarray as xr
from xarray.core.indexing import IndexSelResult
from xpublish_tiles.utils import time_debug

DEFAULT_CRS = CRS.from_epsg(4326)
DEFAULT_PAD = 1


@dataclass(frozen=True)
class PadDimension:
    """Helper class to encapsulate padding parameters for a dimension."""

    name: str
    size: int
    pad: int = DEFAULT_PAD
    wraparound: bool = False
    prevent_overlap: bool = False


# Regex patterns for coordinate detection
X_COORD_PATTERN = re.compile(
    r"^(x|i|xi|nlon|rlon|ni)[a-z0-9_]*$|^x?(nav_lon|lon|glam)[a-z0-9_]*$"
)
Y_COORD_PATTERN = re.compile(
    r"^(y|j|eta|nlat|rlat|nj)[a-z0-9_]*$|^y?(nav_lat|lat|gphi)[a-z0-9_]*$"
)

# TTL cache for grid systems (5 minute TTL, max 128 entries)
_GRID_CACHE = cachetools.TTLCache(maxsize=128, ttl=300)


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
    assert slicer.start <= slicer.stop
    if increasing:
        ys = [
            np.append(np.nonzero(left <= slicer.stop)[axis], 0).max(),
            np.append(np.nonzero(right > slicer.stop)[axis], size).min(),
            np.append(np.nonzero(left <= slicer.start)[axis], 0).max(),
            np.append(np.nonzero(right > slicer.start)[axis], size).min(),
        ]
    else:
        ys = [
            np.append(np.nonzero(left < slicer.stop)[axis], size).min(),
            np.append(np.nonzero(right >= slicer.stop)[axis], 0).max(),
            np.append(np.nonzero(left < slicer.start)[axis], size).min(),
            np.append(np.nonzero(right >= slicer.start)[axis], 0).max(),
        ]
    return ys


def _get_xy_pad(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_pad = numbagg.nanmax(np.abs(np.diff(x)))
    y_pad = numbagg.nanmax(np.abs(np.diff(y)))
    return x_pad, y_pad


def _convert_longitude_slice(lon_slice: slice, *, uses_0_360) -> tuple[slice, ...]:
    """
        Convert longitude slice to match the coordinate system of the dataset.

    Handles conversion between -180→180 and 0→360 coordinate systems.
    May return multiple slices when selection crosses longitude boundaries.

    Parameters
    ----------
    lon_slice : slice
        Input longitude slice (potentially in different coordinate system)

    Returns
    -------
    slice or tuple[slice, ...]
        Converted slice(s) that match dataset's coordinate system.
        Returns tuple of slices when selection crosses boundaries.
    """
    if lon_slice.start is None or lon_slice.stop is None:
        raise ValueError("start and stop should not be None")

    assert lon_slice.step is None
    # start, stop = lon_slice.start, lon_slice.stop

    # https://github.com/developmentseed/morecantile/issues/175
    # the precision in morecantile tile bounds isn't perfect,
    # a good way to test is `tms.bounds(Tile(0,0,0))` which should
    # match the spec exactly: https://docs.ogc.org/is/17-083r4/17-083r4.html#toc48
    # Example: tests/test_pipeline.py::test_pipeline_tiles[-90->90,0->360-wgs84_prime_meridian(2/2/1)]
    start, stop = lon_slice.start, lon_slice.stop

    # Determine breakpoints based on coordinate system
    left_break = 0 if uses_0_360 else -180
    right_break = 360 if uses_0_360 else 180

    # Handle different boundary crossing cases
    if start < left_break and stop < left_break:
        # Both below left boundary
        # e.g., -370 to -350 or -190 to -170
        return (slice(start + 360, stop + 360),)

    elif start >= right_break and stop > right_break:
        # Both above right boundary
        # e.g., 370 to 390 or 190 to 210
        return (slice(start - 360, stop - 360),)

    elif start < left_break and ((left_break == stop) or (stop < right_break)):
        # Crosses left boundary from below
        # e.g., -185 to 1 or -10 to 10
        # For 0→360: slice(-10, 10) becomes slice(350, 360) + slice(0, 10)
        # remember this is left-inclusive intervals
        return (slice(start + 360, right_break), slice(left_break, stop))

    elif start >= left_break and stop > right_break:
        # Crosses right boundary from within
        # e.g., 170 to 190 or 350 to 370
        # For 0→360: slice(350, 370) becomes slice(350, 360) + slice(0, 10)
        # For -180→180: slice(170, 190) becomes slice(170, -170)
        return (slice(start, right_break), slice(left_break, stop - 360))

    elif start >= right_break:
        # Only start is above right boundary
        # e.g., 370 to 10 or 190 to 10
        return (slice(start - 360, stop),)

    else:
        # Both within valid range
        return (slice(start, stop),)


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

    # Calculate differences between adjacent centers
    halfwidth = np.gradient(centers) / 2

    # Initialize bounds array
    bounds = np.empty(len(centers) + 1)
    bounds[:-1] = centers - halfwidth
    bounds[-1] = centers[-1] + halfwidth[-1]
    return bounds


class CurvilinearCellIndex(xr.Index):
    uses_0_360: bool
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

    def __init__(self, *, X: xr.DataArray, Y: xr.DataArray, Xdim: str, Ydim: str):
        self.X, self.Y = X.reset_coords(drop=True), Y.reset_coords(drop=True)
        self.uses_0_360 = (X.data > 180).any()
        self.Xdim, self.Ydim = Xdim, Ydim

        # derived quantities
        X, Y = self.X.data, self.Y.data
        xaxis = self.X.get_axis_num(self.Xdim)
        yaxis = self.Y.get_axis_num(self.Ydim)
        dX, dY = np.gradient(X, axis=xaxis), np.gradient(Y, axis=yaxis)
        self.left, self.right = X - dX / 2, X + dX / 2
        self.bottom, self.top = Y - dY / 2, Y + dY / 2
        self.y_is_increasing = True
        if not (self.bottom < self.top).all():
            self.y_is_increasing = False
            self.top, self.bottom = self.bottom, self.top
        self.xaxis, self.yaxis = xaxis, yaxis

    def sel(self, labels, method=None, tolerance=None) -> IndexSelResult:
        X, Y = self.X.data, self.Y.data
        xaxis, yaxis = self.xaxis, self.yaxis
        bottom, top, left, right = self.bottom, self.top, self.left, self.right
        Xlen, Ylen = X.shape[xaxis], Y.shape[yaxis]

        assert len(labels) == 1
        bbox = next(iter(labels.values()))
        assert isinstance(bbox, BBox)

        slices = _convert_longitude_slice(
            slice(bbox.west, bbox.east), uses_0_360=self.uses_0_360
        )

        ys = _grab_edges(
            bottom,
            top,
            slicer=slice(bbox.south, bbox.north),
            axis=yaxis,
            size=Ylen,
            increasing=self.y_is_increasing,
        )
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
            # add 1 to account for slice upper end being exclusive
            self.Ydim: slice(min(ys), max(ys) + 1),
        }
        return IndexSelResult(slicers)

    def equals(self, other: Self) -> bool:
        return (
            self.X.equals(other.X)
            and self.Y.equals(other.Y)
            and self.Xdim == other.Xdim
            and self.Ydim == other.Ydim
        )


class LongitudeCellIndex(xr.indexes.PandasIndex):
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

        # Determine if dataset uses 0→360 or -180→180 coordinate system
        coord_centers = self.cell_centers
        self._uses_0_360 = coord_centers.min() >= 0 and coord_centers.max() > 180

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

    @property
    def uses_0_360(self) -> bool:
        """Check if this longitude index uses 0→360 coordinate system (vs -180→180)."""
        return self._uses_0_360

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
        key, value = next(iter(labels.items()))
        if not isinstance(value, slice):
            raise NotImplementedError

        converted_slices = _convert_longitude_slice(value, uses_0_360=self.uses_0_360)

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
        return IndexSelResult({self.dim: tuple(all_indexers)})

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


def _prevent_slice_overlap(indexers: list[slice]) -> list[slice]:
    """
    Prevent overlapping slices by adjusting stop positions.

    This mimics the original logic: if a slice's stop position would overlap
    with a previously added slice's start, adjust the stop to prevent overlap.
    This is used for anti-meridian longitude selections where slices may be
    processed in an order that could cause overlaps.
    """
    if len(indexers) <= 1:
        return indexers

    result = []
    for indexer in indexers:
        start, stop, step = indexer.start, indexer.stop, indexer.step

        # Apply the same logic as the original inline code:
        # if len(all_indexers) > 0 and (stop >= all_indexers[-1].start):
        #     stop = all_indexers[-1].start
        if len(result) > 0 and stop >= result[-1].start:
            stop = result[-1].start

        result.append(slice(start, stop, step))

    return result


def pad_slicers(
    slicers: dict[str, slice | tuple[slice, ...]], *, dimensions: list[PadDimension]
) -> dict[str, list[slice]]:
    """
    Apply padding to slicers for specified dimensions.

    Parameters
    ----------
    slicers : dict[str, slice | tuple[slice, ...]]
        Dictionary mapping dimension names to slices
    dimensions : list[PadDimension]
        List of dimension padding information

    Returns
    -------
    dict[str, list[slice]]
        Dictionary mapping dimension names to lists of padded slices
    """
    result = {}

    # Handle each specified dimension
    for dim in dimensions:
        if dim.name not in slicers:
            continue

        dim_slicers = slicers[dim.name]
        idxrs = (dim_slicers,) if isinstance(dim_slicers, slice) else dim_slicers
        indexers: list[slice] = [slice(*idxr.indices(dim.size)) for idxr in idxrs]  # type: ignore[arg-type, var-annotated]

        # Prevent overlap if requested (before padding)
        if dim.prevent_overlap:
            indexers = _prevent_slice_overlap(indexers)

        # Apply padding
        first, last = indexers[0], indexers[-1]
        if len(indexers) == 1:
            indexers = [
                slice(max(0, first.start - dim.pad), min(dim.size, first.stop + dim.pad))
            ]
        else:
            indexers = [
                slice(max(0, first.start - dim.pad), first.stop),
                *indexers[1:-1],
                slice(last.start, min(dim.size, last.stop + dim.pad)),
            ]

        # Apply wraparound if enabled for this dimension
        if dim.wraparound:
            if indexers[0].start == 0:
                # Starts at beginning, add wraparound from end
                indexers = [slice(-dim.pad, None), *indexers]
            if indexers[-1].stop >= dim.size - 1:
                # Ends at end, add wraparound from beginning
                indexers = indexers + [slice(0, dim.pad)]

        result[dim.name] = indexers

    # Pass through any other dimensions unchanged
    for key, value in slicers.items():
        if key not in result:
            if isinstance(value, slice):
                result[key] = [value]
            elif isinstance(value, tuple):
                result[key] = list(value)
            else:
                # This shouldn't happen given our type signature, but handle gracefully
                result[key] = [value]  # type: ignore[list-item]

    return result


@dataclass(eq=False)
class GridSystem(ABC):
    """
    Marker class for Grid Systems.

    Subclasses contain all information necessary to define the horizontal mesh,
    bounds, and reference frame for that specific grid system.
    """

    # FIXME: do we really need these Index objects on the class?
    #   - reconsider when we do curvilinear and triangular grids
    #   - The ugliness is that booth would have to set the right indexes on the dataset.
    #   - So this is do-able, but there's some strong coupling between the
    #     plugin and the "orchestrator"
    indexes: tuple[xr.Index, ...]
    Z: str | None = None

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
        if any(
            not a.equals(b) for a, b in zip(self.indexes, other.indexes, strict=False)
        ):
            return False
        return True

    def __eq__(self, other) -> bool:
        """Override dataclass __eq__ to use our custom equals() method."""
        if not isinstance(other, GridSystem):
            return False
        return self.equals(other)

    def sel(self, da: xr.DataArray, *, bbox: BBox) -> dict[str, list[slice]]:
        """Select a subset of the data array using a bounding box."""
        raise NotImplementedError("Subclasses must implement sel method")


class RectilinearSelMixin:
    """Mixin for generic rectilinear .sel"""

    def sel(
        self,
        *,
        da: xr.DataArray,
        bbox: BBox,
        y_is_increasing: bool,
        x_size: int,
        y_size: int,
        x_handle_wraparound: bool,
    ) -> dict[str, list[slice]]:
        """
        This method handles coordinate selection for rectilinear grids, automatically
        converting between different longitude conventions (0→360 vs -180→180).

        Parameters
        ----------
        da : xr.DataArray
            Data array to select from
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

        # Prepare slicers for padding
        slicers = {self.X: xsel_result.dim_indexers[self.X], self.Y: yslice}

        # Apply padding with PadDimension helpers
        xdim = PadDimension(
            name=self.X, size=x_size, pad=DEFAULT_PAD, wraparound=x_handle_wraparound
        )
        ydim = PadDimension(name=self.Y, size=y_size, pad=DEFAULT_PAD, wraparound=False)

        return pad_slicers(slicers, dimensions=[xdim, ydim])


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

    def __post_init__(self) -> None:
        self.Xdim = self.X
        self.Ydim = self.Y

    @property
    def dims(self) -> set[str]:
        """Return the set of dimension names for this grid system."""
        return {self.Xdim, self.Ydim}

    def assign_index(self, da: xr.DataArray) -> xr.DataArray:
        (index,) = self.indexes
        return da.assign_coords(xr.Coordinates.from_xindex(index))

    def sel(self, da: xr.DataArray, *, bbox: BBox) -> dict[str, list[slice]]:
        (index,) = self.indexes
        affine = index.transform()

        # Determine if this is a global raster
        x_handle_wraparound = self.crs.is_geographic and _is_raster_index_global(
            index, self.bbox, self.crs
        )

        return super().sel(
            da=da,
            bbox=bbox,
            y_is_increasing=affine.e > 0,
            x_size=index._xy_shape[0],
            y_size=index._xy_shape[1],
            x_handle_wraparound=x_handle_wraparound,
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

    def __post_init__(self) -> None:
        self.Xdim = self.X
        self.Ydim = self.Y

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

        x_bounds = _compute_interval_bounds(X.data)
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
            # Handle global datasets
            x_span = east - west
            if x_span >= 359.0:  # Nearly global in longitude
                if west < -179.0:
                    west, east = -180, 180
                elif east > 181:
                    west, east = 0, 360
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

    def sel(self, da: xr.DataArray, *, bbox: BBox) -> dict[str, list[slice]]:
        """
        Select a subset of the data array using a bounding box.
        """
        assert self.X in da.xindexes and self.Y in da.xindexes
        assert isinstance(da.xindexes[self.Y], xr.indexes.PandasIndex)

        x_index, y_index = self.indexes[0], self.indexes[-1]

        # For Rectilinear grids, X index is always LongitudeCellIndex (geographic)
        # or PandasIndex (non-geographic)
        if self.crs.is_geographic:
            # Geographic CRS should always have LongitudeCellIndex
            assert isinstance(
                x_index, LongitudeCellIndex
            ), f"Expected LongitudeCellIndex for geographic CRS, got {type(x_index)}"
            x_handle_wraparound = x_index.is_global
        else:
            # Non-geographic CRS should have regular PandasIndex
            assert isinstance(
                x_index, xr.indexes.PandasIndex
            ), f"Expected PandasIndex for non-geographic CRS, got {type(x_index)}"
            x_handle_wraparound = False  # No wraparound for projected coordinates

        # Both index types have len() method
        x_size = len(x_index.index)
        y_size = len(y_index.index)
        y_index_cast = cast(xr.indexes.PandasIndex, da.xindexes[self.Y])

        return super().sel(
            da=da,
            bbox=bbox,
            y_is_increasing=y_index_cast.index.is_monotonic_increasing,
            x_size=x_size,
            y_size=y_size,
            x_handle_wraparound=x_handle_wraparound,
        )

    def equals(self, other: Self) -> bool:
        if (self.crs == other.crs and self.bbox == other.bbox) or (
            self.X == other.X and self.Y == other.Y
        ):
            return super().equals(other)
        else:
            return False


@dataclass(kw_only=True, eq=False)
class Curvilinear(GridSystem):
    """2D horizontal grid defined by two 2D arrays."""

    crs: CRS
    bbox: BBox
    X: str
    Y: str
    Xdim: str
    Ydim: str
    indexes: tuple[xr.Index, ...]
    Z: str | None = None

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
        X, Y = ds[Xname], ds[Yname]
        Xdim, Ydim = Curvilinear._guess_dims(ds, X=X, Y=Y)
        index = CurvilinearCellIndex(X=X, Y=Y, Xdim=Xdim, Ydim=Ydim)
        bbox = BBox(
            west=numbagg.nanmin(index.left),
            east=numbagg.nanmax(index.right),
            south=numbagg.nanmin(index.bottom),
            north=numbagg.nanmax(index.top),
        )
        return cls(
            crs=crs,
            X=Xname,
            Y=Yname,
            Xdim=Xdim,
            Ydim=Ydim,
            bbox=bbox,
            indexes=(index,),
        )

    @property
    def dims(self) -> set[str]:
        """Return the set of dimension names for this grid system."""
        return {self.Xdim, self.Ydim}

    def equals(self, other: Self) -> bool:
        if (self.crs == other.crs and self.bbox == other.bbox) or (
            self.X == other.X and self.Y == other.Y
        ):
            return super().equals(other)
        else:
            return False

    def sel(self, da: xr.DataArray, *, bbox: BBox) -> dict[str, list[slice]]:
        """
        Select a subset of the data array using a bounding box.

        Uses masking to select out the bbox for curvilinear grids where coordinates
        are 2D arrays. Also normalizes longitude coordinates to -180→180 format.
        """
        # Uses masking to select out the bbox, following the discussion in
        # https://github.com/pydata/xarray/issues/10572
        index = next(iter(self.indexes))
        assert isinstance(
            index, CurvilinearCellIndex
        ), f"Expected CurvilinearCellIndex, got {type(index)}"

        X = index.X
        handle_wraparound = self.crs.is_geographic and (
            (numbagg.nanmax(X.data) - numbagg.nanmin(X.data)) >= 350
        )
        sel_result = index.sel({self.Xdim: bbox})

        # Get slicers for both dimensions
        xslicers = sel_result.dim_indexers[self.Xdim]
        yslicers = sel_result.dim_indexers[self.Ydim]

        # Get sizes for both dimensions
        xsize = index.X.sizes[self.Xdim]
        ysize = index.Y.sizes[self.Ydim]

        # Apply padding with PadDimension helpers
        xdim = PadDimension(
            name=self.Xdim, size=xsize, pad=DEFAULT_PAD, wraparound=handle_wraparound
        )
        ydim = PadDimension(name=self.Ydim, size=ysize, pad=DEFAULT_PAD, wraparound=False)

        return pad_slicers(
            {self.Xdim: xslicers, self.Ydim: yslicers},
            dimensions=[xdim, ydim],
        )


@dataclass(kw_only=True, eq=False)
class DGGS(GridSystem):
    cells: str
    indexes: tuple[xr.Index, ...]
    Z: str | None = None

    @property
    def dims(self) -> set[str]:
        """Return the set of dimension names for this grid system."""
        return {self.cells}

    def sel(self, da: xr.DataArray, *, bbox: BBox) -> xr.DataArray:
        """Select a subset of the data array using a bounding box."""
        raise NotImplementedError("sel not implemented for DGGS grids")

    def equals(self, other: Self) -> bool:
        if self.cells == other.cells:
            return super().equals(other)
        else:
            return False


def _guess_grid_mapping_and_crs(
    ds: xr.Dataset,
) -> tuple[xr.DataArray | None, CRS | None]:
    """
    Returns
    ------
    grid_mapping variable
    CRS
    """
    grid_mapping_names = tuple(itertools.chain(*ds.cf.grid_mapping_names.values()))
    if not grid_mapping_names:
        if "spatial_ref" in ds.variables:
            grid_mapping_names += ("spatial_ref",)
        elif "crs" in ds.variables:
            grid_mapping_names += ("crs",)
    if len(grid_mapping_names) == 0:
        keys = ds.cf.keys()
        if "latitude" in keys and "longitude" in keys:
            return None, DEFAULT_CRS
        else:
            warnings.warn("No CRS detected", UserWarning, stacklevel=2)
            return None, None
    if len(grid_mapping_names) > 1:
        raise ValueError(f"Multiple grid mappings found: {grid_mapping_names!r}!")
    (grid_mapping_var,) = grid_mapping_names
    grid_mapping = ds[grid_mapping_var]
    return grid_mapping, CRS.from_cf(grid_mapping.attrs)


def guess_coordinate_vars(ds: xr.Dataset, crs: CRS) -> tuple[str, str]:
    if is_rotated_pole(crs):
        stdnames = ds.cf.standard_names
        Xname, Yname = (
            stdnames.get("grid_longitude", ()),
            stdnames.get("grid_latitude", None),
        )
    elif crs.is_geographic:
        coords = ds.cf.coordinates
        Xname, Yname = coords.get("longitude", None), coords.get("latitude", None)
    else:
        axes = ds.cf.axes
        Xname, Yname = axes.get("X", None), axes.get("Y", None)
    return Xname, Yname


@time_debug
def _guess_grid_for_dataset(ds: xr.Dataset) -> GridSystem:
    """
    Does some grid_mapping & CRS auto-guessing.

    Raises RuntimeError to indicate that we might try again.
    """
    grid_mapping, crs = _guess_grid_mapping_and_crs(ds)
    if crs is not None:
        # This means we are not DGGS for sure.
        # TODO: we aren't handling the triangular case very explicitly yet.
        Xname, Yname = guess_coordinate_vars(ds, crs)
        if Xname is None or Yname is None:
            # FIXME: let's be a little more targeted in what we are guessing
            ds = ds.cf.guess_coord_axis()
            Xname, Yname = guess_coordinate_vars(ds, crs)

        # TODO: we might use rasterix for when there are explicit coords too?
        if Xname is None or Yname is None:
            if grid_mapping is None:
                raise RuntimeError("Grid system could not be inferred.")
            else:
                # Use regex patterns to find coordinate dimensions
                x_dim = None
                y_dim = None
                for dim in ds.dims:
                    if x_dim is None and X_COORD_PATTERN.match(dim):
                        x_dim = dim
                    if y_dim is None and Y_COORD_PATTERN.match(dim):
                        y_dim = dim

                if x_dim and y_dim:
                    ds = rasterix.assign_index(ds, x_dim=x_dim, y_dim=y_dim)
                    index = ds.xindexes[x_dim]
                    return RasterAffine(
                        crs=crs,
                        X=x_dim,
                        Y=y_dim,
                        bbox=BBox(
                            west=index.bbox.left,
                            east=index.bbox.right,
                            south=index.bbox.bottom,
                            north=index.bbox.top,
                        ),
                        indexes=(index,),
                    )
                raise RuntimeError(
                    f"Creating raster affine grid system failed. Detected {grid_mapping=!r}."
                )

        if Xname is not None and len(Xname) > 1:
            if len(ds.data_vars) == 1:
                da = next(iter(ds.data_vars.values()))
                if coords_attr := da.attrs.get("coordinates", ""):
                    Xname = tuple(x for x in Xname if x in coords_attr.split(" "))
        if Yname is not None and len(Yname) > 1:
            if len(ds.data_vars) == 1:
                da = next(iter(ds.data_vars.values()))
                if coords_attr := da.attrs.get("coordinates", ""):
                    Yname = tuple(y for y in Yname if y in coords_attr.split(" "))

        (Xname,) = Xname
        (Yname,) = Yname
        X = ds[Xname]
        Y = ds[Yname]

        if X.ndim == 1 and Y.ndim == 1:
            if is_rotated_pole(crs):
                raise NotImplementedError("Rotated pole grids are not supported yet.")
            return Rectilinear.from_dataset(ds, crs, Xname, Yname)
        elif X.ndim == 2 and Y.ndim == 2:
            return Curvilinear.from_dataset(ds, crs, Xname, Yname)
        else:
            raise RuntimeError(
                f"Unknown grid system: X={Xname!r}, ndim={X.ndim}; Y={Yname!r}, ndim={Y.ndim}"
            )
    else:
        raise RuntimeError("CRS/grid system not detected")


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
    # Only use cache if _xpublish_id is present
    if (xpublish_id := ds.attrs.get("_xpublish_id")) is not None:
        if (cache_key := (xpublish_id, name)) in _GRID_CACHE:
            return _GRID_CACHE[cache_key]

    try:
        grid = _guess_grid_for_dataset(ds.cf[[name]])
    except RuntimeError:
        try:
            grid = _guess_grid_for_dataset(ds)
        except RuntimeError:
            ds = ds.cf.guess_coord_axis()
            grid = _guess_grid_for_dataset(ds)

    grid.Z = _guess_z_dimension(ds.cf[name])

    if xpublish_id is not None:
        _GRID_CACHE[cache_key] = grid

    return grid
