import itertools
import re
import warnings
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Self, cast

import cachetools
import numbagg
import numpy as np
import pandas as pd
import rasterix
from pyproj import CRS
from pyproj.aoi import BBox

import xarray as xr
from xarray.core.indexing import IndexSelResult

DEFAULT_CRS = CRS.from_epsg(4326)

# Regex patterns for coordinate detection
X_COORD_PATTERN = re.compile(r"^(x|i|nlon|rlon|ni|x?(nav_lon|lon|glam)[a-z0-9]*)$")
Y_COORD_PATTERN = re.compile(r"^(y|j|nlat|rlat|nj|y?(nav_lat|lat|gphi)[a-z0-9]*)$")

# TTL cache for grid systems (5 minute TTL, max 128 entries)
_GRID_CACHE = cachetools.TTLCache(maxsize=128, ttl=300)


def _get_xy_pad(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_pad = numbagg.nanmax(np.abs(np.diff(x)))
    y_pad = numbagg.nanmax(np.abs(np.diff(y)))
    return x_pad, y_pad


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
    halfwidth = np.diff(centers) / 2

    # Initialize bounds array
    bounds = np.empty(len(centers) + 1)
    bounds[0] = centers[0] - halfwidth[0]
    bounds[1:-1] = centers[:-1] + halfwidth
    bounds[-1] = centers[-1] + halfwidth[-1]
    return bounds


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

    def sel(self, labels, method=None, tolerance=None):
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
        if isinstance(value, slice):
            converted_slices = self._convert_longitude_slice(value)

            # If we got multiple slices (for boundary crossing), create multiple indexers
            if isinstance(converted_slices, tuple):
                # Handle multiple slices by selecting each and creating multiple indexers
                all_indexers = []
                for slice_part in converted_slices:
                    sel_dict = {self.dim: slice_part}
                    result = self._xrindex.sel(
                        sel_dict, method=method, tolerance=tolerance
                    )
                    indexer = next(iter(result.dim_indexers.values()))
                    start, stop, step = indexer.indices(len(self))
                    if len(all_indexers) > 0 and (stop >= all_indexers[-1].start):
                        stop = all_indexers[-1].start
                    all_indexers.append(slice(start, stop, step))
                return IndexSelResult({self.dim: tuple(all_indexers)})
            else:
                value = converted_slices
        return self._xrindex.sel({key: value}, method=method, tolerance=tolerance)

    def _convert_longitude_slice(self, lon_slice: slice) -> slice | tuple[slice, ...]:
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

        start, stop = lon_slice.start, lon_slice.stop
        if self.uses_0_360:
            # Check if we need to split across the 0/360 boundary
            if start < 0 <= stop:
                # Selection crosses from negative to positive
                # e.g., slice(-180, 10) becomes slice(180, 360) + slice(0, 10)
                if stop != 0:
                    return (slice(start + 360, 360), slice(0, stop))
                else:
                    return (slice(start + 360, 360),)

            elif start < 0 and stop < 0:
                # Both negative, convert to 0→360
                return slice(start + 360, stop + 360)
            return lon_slice

        else:
            # Check if we need to split across the -180/180 boundary
            if start > 180 >= stop or (start > 180 and stop <= start):
                # Selection might cross boundary, but this is less common
                # For now, just convert values > 180
                if start > 180:
                    start = start - 360
                if stop > 180:
                    stop = stop - 360
                return slice(start, stop, lon_slice.step)
            elif start > 180:
                start -= 360
            if stop > 180:
                stop -= 360
            return slice(start, stop, lon_slice.step)

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


def _handle_longitude_selection(
    lon_index: xr.Index, bbox: BBox, grid_bbox: BBox, crs: CRS, dimname: Hashable
) -> list[slice]:
    """
    Handle longitude coordinate selection with support for different conventions.

    Uses the longitude index to determine selection strategy. For global RasterIndex or
    LongitudeCellIndex, applies complex anti-meridian and boundary logic. Otherwise,
    uses simple .sel() method to determine slices.

    Parameters
    ----------
    lon_index : xr.Index
        The longitude/X coordinate index (first element from grid.indexes)
    bbox : BBox
        Bounding box for selection (may cross anti-meridian after coordinate transformation)
    grid_bbox : BBox, optional
        The grid's overall bounding box (needed for RasterIndex global detection)
    crs : CRS, optional
        The coordinate reference system (needed for RasterIndex global detection)

    Returns
    -------
    tuple[slice, ...]
        Tuple of slices for coordinate selection. Usually one slice, but two slices
        when bbox crosses boundaries or anti-meridian.
    """
    if isinstance(lon_index, rasterix.RasterIndex):
        handle_wraparound = crs.is_geographic and _is_raster_index_global(
            lon_index, grid_bbox, crs
        )
        size = lon_index._xy_shape[0]
    elif isinstance(lon_index, LongitudeCellIndex):
        handle_wraparound = (
            crs.is_geographic
            and lon_index.cell_centers[-1] == 360 + lon_index.cell_centers[0]
        )
        size = len(lon_index.index)
    else:
        assert isinstance(lon_index, xr.indexes.PandasIndex)
        handle_wraparound = (
            crs.is_geographic and lon_index.index[-1] == 360 + lon_index.index[0]
        )
        size = len(lon_index.index)

    sel_result = lon_index.sel({dimname: slice(bbox.west, bbox.east)})
    indexer = next(iter(sel_result.dim_indexers.values()))
    indexers = (indexer,) if isinstance(indexer, slice) else indexer
    indexers: list[slice] = [slice(*idxr.indices(size)) for idxr in indexers]
    first, last = indexers[0], indexers[-1]
    if len(indexers) == 1:
        indexers = [slice(max(0, first.start - 1), first.stop + 1)]
    else:
        indexers = [
            slice(max(0, first.start - 1), first.stop),
            *indexers[1:-1],
            slice(last.start, last.stop + 1, 1),
        ]
    if handle_wraparound:
        if indexers[0].start == 0:
            # Starts at beginning, add wraparound from end
            indexers = [slice(-1, None), *indexers]
        if indexers[-1].stop >= size - 1:
            # Ends at end, add wraparound from beginning
            indexers = indexers + [slice(0, 1)]
    return indexers


@dataclass(eq=False)
class GridSystem:
    """
    Marker class for Grid Systems.

    Subclasses contain all information necessary to define the horizontal mesh,
    bounds, and reference frame for that specific grid system.
    """

    dims: set[str]
    # FIXME: do we really need these Index objects on the class?
    #   - reconsider when we do curvilinear and triangular grids
    #   - The ugliness is that booth would have to set the right indexes on the dataset.
    #   - So this is do-able, but there's some strong coupling between the
    #     plugin and the "orchestrator"
    indexes: tuple[xr.Index, ...]
    Z: str | None = None

    def equals(self, other: Self) -> bool:
        if not isinstance(self, type(other)):
            return False
        if self.dims != other.dims:
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

    def sel(self, da: xr.DataArray, *, bbox: BBox) -> xr.DataArray:
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
    ) -> xr.DataArray:
        """
        This method handles coordinate selection for rectilinear grids, automatically
        converting between different longitude conventions (0→360 vs -180→180).
        """
        assert len(self.indexes) >= 1
        xindex, yindex = self.indexes[0], self.indexes[-1]
        # yes, this works for increasing & decreasing y
        if y_is_increasing:
            yslice = yindex.sel({self.Y: slice(bbox.south, bbox.north)}).dim_indexers[
                self.Y
            ]
        else:
            yslice = yindex.sel({self.Y: slice(bbox.north, bbox.south)}).dim_indexers[
                self.Y
            ]

        yslice = slice(max(0, yslice.start - 1), yslice.stop + 1, 1)

        # Notes:
        # 1. I am relying on these slices being in the correct order
        # 2. There is no point to transforming the coordinate values so that they are contiguous
        #    pyproj will reintroduce the discontintuity at the anti-meridian after transforming
        #    coordinate systems
        lon_slices = _handle_longitude_selection(
            xindex, bbox, self.bbox, self.crs, self.X
        )
        results = [
            da.isel({self.X: lon_slice, self.Y: yslice}) for lon_slice in lon_slices
        ]
        if len(results) > 1:
            result = xr.concat(results, dim=self.X)
        else:
            (result,) = results
        return result


@dataclass(kw_only=True, eq=False)
class RasterAffine(RectilinearSelMixin, GridSystem):
    """2D horizontal grid defined by an affine transform."""

    crs: CRS
    bbox: BBox
    X: str
    Y: str
    dims: set[str] = field(init=False)
    indexes: tuple[rasterix.RasterIndex]
    Z: str | None = None

    def __post_init__(self) -> None:
        self.dims = {self.X, self.Y}

    def sel(self, da: xr.DataArray, *, bbox: BBox) -> xr.DataArray:
        (index,) = self.indexes
        affine = index.transform()
        da = da.assign_coords(xr.Coordinates.from_xindex(index))
        return super().sel(
            da=da,
            bbox=bbox,
            y_is_increasing=affine.e > 0,
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
    dims: set[str] = field(init=False)
    indexes: tuple[xr.indexes.PandasIndex, xr.indexes.PandasIndex]
    Z: str | None = None

    def __post_init__(self) -> None:
        self.dims = {self.X, self.Y}

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
                if east > 180:
                    west, east = 0, 360
                elif west < 0:
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

    def sel(self, da: xr.DataArray, *, bbox: BBox) -> xr.DataArray:
        """
        Select a subset of the data array using a bounding box.
        """
        assert self.X in da.xindexes and self.Y in da.xindexes
        assert isinstance(da.xindexes[self.Y], xr.indexes.PandasIndex)
        y_index = cast(xr.indexes.PandasIndex, da.xindexes[self.Y])
        return super().sel(
            da=da,
            bbox=bbox,
            y_is_increasing=y_index.index.is_monotonic_increasing,
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
    dims: set[str]
    indexes: tuple[xr.Index, ...]
    Z: str | None = None

    def equals(self, other: Self) -> bool:
        if (self.crs == other.crs and self.bbox == other.bbox) or (
            self.X == other.X and self.Y == other.Y
        ):
            return super().equals(other)
        else:
            return False

    def sel(self, da: xr.DataArray, *, bbox: BBox) -> xr.DataArray:
        """
        Select a subset of the data array using a bounding box.

        Uses masking to select out the bbox for curvilinear grids where coordinates
        are 2D arrays. Also normalizes longitude coordinates to -180→180 format.
        """
        # Assert that bbox doesn't cross anti-meridian (Web Mercator tiles never do)
        assert (
            bbox.west <= bbox.east
        ), f"BBox crosses anti-meridian: west={bbox.west} > east={bbox.east}"

        # Uses masking to select out the bbox, following the discussion in
        # https://github.com/pydata/xarray/issues/10572
        X = da[self.X].data
        Y = da[self.Y].data

        xinds, yinds = np.nonzero(
            (X >= bbox.west) & (X <= bbox.east) & (Y >= bbox.south) & (Y <= bbox.north)
        )
        slicers = {
            self.X: slice(xinds.min(), xinds.max() + 1),
            self.Y: slice(yinds.min(), yinds.max() + 1),
        }

        result = da.isel(slicers)
        return result

    # def sel_ndpoint(self, da: xr.DataArray, *, bbox: BBox) -> xr.DataArray:
    #     # https://github.com/pydata/xarray/issues/10572
    #     assert len(self.indexes) == 1
    #     (index,) = self.indexes
    #     assert isinstance(index, xr.indexes.NDPointIndex)

    #     slicers = {
    #         self.X: slice(bbox.west, bbox.east),
    #         self.Y: slice(bbox.south, bbox.north),
    #     }
    #     index = da.xindexes[self.X]
    #     edges = tuple((slicer.start, slicer.stop) for slicer in slicers.values())
    #     vectorized_sel = {
    #         name: xr.DataArray(dims=("pts",), data=data)
    #         for name, data in zip(
    #             slicers.keys(),
    #             map(np.asarray, zip(*itertools.product(*edges), strict=False)),
    #             strict=False,
    #         )
    #     }
    #     idxrs = index.sel(vectorized_sel, method="nearest").dim_indexers
    #     new_slicers = {
    #         name: slice(array.min().item(), array.max().item())
    #         for name, array in idxrs.items()
    #     }
    #     return da.isel(new_slicers)


@dataclass(kw_only=True, eq=False)
class DGGS(GridSystem):
    cells: str
    dims: set[str]
    indexes: tuple[xr.Index, ...]
    Z: str | None = None

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

        # FIXME: nice error here
        (Xname,) = Xname
        (Yname,) = Yname
        X = ds[Xname]
        Y = ds[Yname]

        if X.ndim == 1 and Y.ndim == 1:
            if is_rotated_pole(crs):
                raise NotImplementedError("Rotated pole grids are not supported yet.")
            return Rectilinear.from_dataset(ds, crs, Xname, Yname)
        elif X.ndim == 2 and Y.ndim == 2:
            # TODO: bbox is wrong here
            bbox = BBox(
                west=numbagg.nanmin(X.data).item(),
                east=numbagg.nanmax(X.data).item(),
                south=numbagg.nanmin(Y.data).item(),
                north=numbagg.nanmax(Y.data).item(),
            )
            dims = set(X.dims) | set(Y.dims)
            # See discussion in https://github.com/pydata/xarray/issues/10572
            return Curvilinear(
                crs=crs,
                X=Xname,
                Y=Yname,
                dims=cast(set[str], dims),
                bbox=bbox,
                indexes=tuple(),
            )

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


def guess_grid_system(ds: xr.Dataset, name: str) -> GridSystem:
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
