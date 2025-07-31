import itertools
import warnings
from dataclasses import dataclass
from typing import cast

import numpy as np
import pyproj
import pyproj.aoi

import xarray as xr

DEFAULT_CRS = pyproj.CRS.from_epsg(4326)


def pad_bbox(
    bbox: pyproj.aoi.BBox, da: xr.DataArray, x_dim: str, y_dim: str
) -> pyproj.aoi.BBox:
    """
    Extend bbox slightly to account for discrete coordinate sampling.
    This prevents transparency gaps at tile edges due to coordinate resolution.

    The function ensures that the padded bbox does not cross the anti-meridian
    by checking if padding would cause west > east.
    """
    x = da[x_dim].data
    y = da[y_dim].data

    # Extend bbox by maximum coordinate spacing on each side
    # This is needed for high zoom tiles smaller than coordinate spacing
    x_pad = np.abs(np.diff(x)).max()
    y_pad = np.abs(np.diff(y)).max()

    padded_west = float(bbox.west - x_pad)
    padded_east = float(bbox.east + x_pad)

    # Check if padding would cause anti-meridian crossing
    # This happens when the padded west > padded east
    if padded_west > padded_east:
        # Don't pad in the x direction to avoid crossing
        padded_west = float(bbox.west)
        padded_east = float(bbox.east)

    return pyproj.aoi.BBox(
        west=padded_west,
        east=padded_east,
        south=float(bbox.south - y_pad),
        north=float(bbox.north + y_pad),
    )


def is_rotated_pole(crs: pyproj.CRS) -> bool:
    return crs.to_cf().get("grid_mapping_name") == "rotated_latitude_longitude"


def _handle_longitude_selection(
    lon_coord: xr.DataArray, bbox: pyproj.aoi.BBox, is_geographic: bool
) -> tuple[tuple[slice, ...], xr.DataArray | None]:
    """
    Handle longitude coordinate selection with support for different conventions.

    This function handles coordinate convention conversion between bbox (-180→180) and data
    (which may be 0→360), as well as anti-meridian crossing bboxes that can occur after
    coordinate transformation from Web Mercator to geographic coordinates.

    Parameters
    ----------
    lon_coord : xr.DataArray
        The longitude/X coordinate array
    bbox : pyproj.aoi.BBox
        Bounding box for selection (may cross anti-meridian after coordinate transformation)
    is_geographic : bool
        Whether the CRS is geographic (True) or projected (False)

    Returns
    -------
    tuple[slice, ...]
        Tuple of slices for coordinate selection. Usually one slice, but two slices
        when bbox crosses the anti-meridian or 360°/0° boundary.
    """
    if not is_geographic:
        # For projected coordinates, treat X as regular coordinate
        return (slice(bbox.west, bbox.east),)

    # For geographic coordinates, handle longitude wrapping
    lon_values = lon_coord.data
    lon_min, lon_max = lon_values.min().item(), lon_values.max().item()

    # Determine if data uses 0→360 or -180→180 convention
    uses_0_360 = lon_min >= 0 and lon_max > 180

    # Handle anti-meridian crossing bboxes (west > east)
    if bbox.west > bbox.east:
        if uses_0_360:
            # Data is 0→360, bbox crosses anti-meridian
            # Convert to 0→360 convention
            # Region 1: from west+360 to 360, Region 2: from 0 to east+360
            west_360 = bbox.west + 360 if bbox.west < 0 else bbox.west
            east_360 = bbox.east + 360
            return (slice(west_360, 360.0), slice(0.0, east_360))
        else:
            # Data is -180→180, bbox crosses anti-meridian
            # Region 1: from west to 180, Region 2: from -180 to east
            return (slice(bbox.west, 180.0), slice(-180.0, bbox.east))

    # No anti-meridian crossing
    bbox_west = bbox.west
    bbox_east = bbox.east

    if uses_0_360 and bbox.west < 0:
        # Data is 0→360, bbox is typically -180→180
        # Convert negative longitudes to 0→360 range
        bbox_west_360 = bbox.west + 360
        bbox_east_360 = bbox.east + 360 if bbox.east < 0 else bbox.east

        if bbox_west_360 > bbox_east_360:
            # Bbox crosses 360°/0° boundary - need to select two ranges
            # Return two slices: [bbox_west_360, 360] and [0, bbox_east_360]
            return (slice(bbox_west_360, 360), slice(0, bbox_east_360))
        else:
            # Normal case - single range in 0→360 convention
            return (slice(bbox_west_360, bbox_east_360),)
    else:
        # Use original bbox coordinates (data is -180→180 or no negative bbox values)
        return (slice(bbox_west, bbox_east),)


class GridSystem:
    """
    Marker class for Grid Systems.

    Subclasses contain all information necessary to define the horizontal mesh,
    bounds, and reference frame for that specific grid system.
    """

    indexes: tuple[xr.Index, ...]

    def sel(self, da: xr.DataArray, *, bbox: pyproj.aoi.BBox) -> xr.DataArray:
        """Select a subset of the data array using a bounding box."""
        raise NotImplementedError("Subclasses must implement sel method")


@dataclass(kw_only=True)
class RasterAffine(GridSystem):
    """2D horizontal grid defined by an affine transform."""

    crs: pyproj.CRS
    bbox: pyproj.aoi.BBox
    indexes: tuple[xr.Index, ...]

    def pad_bbox(self, bbox: pyproj.aoi.BBox, da: xr.DataArray) -> pyproj.aoi.BBox:
        """Extend bbox slightly to account for discrete coordinate sampling."""
        raise NotImplementedError("pad_bbox not implemented for RasterAffine grids")


@dataclass(kw_only=True)
class Rectilinear(GridSystem):
    """2D horizontal grid defined by two explicit 1D basis vectors."""

    crs: pyproj.CRS
    bbox: pyproj.aoi.BBox
    X: str
    Y: str
    indexes: tuple[xr.Index, ...]

    def sel(self, da: xr.DataArray, *, bbox: pyproj.aoi.BBox) -> xr.DataArray:
        """
        Select a subset of the data array using a bounding box.

        This method handles coordinate selection for rectilinear grids, automatically
        converting between different longitude conventions (0→360 vs -180→180) and
        ensuring consistent output coordinates in -180→180 format.

        Web Mercator tiles are guaranteed never to cross the anti-meridian, which
        simplifies the longitude handling significantly compared to arbitrary bounding boxes.
        """
        assert self.X in da.xindexes and self.Y in da.xindexes
        assert isinstance(da.xindexes[self.X], xr.indexes.PandasIndex)
        assert isinstance(da.xindexes[self.Y], xr.indexes.PandasIndex)

        slicers = {}
        y_index = cast(xr.indexes.PandasIndex, da.xindexes[self.Y])
        if y_index.index.is_monotonic_increasing:
            slicers[self.Y] = slice(bbox.south, bbox.north)
        else:
            slicers[self.Y] = slice(bbox.north, bbox.south)

        if self.crs.is_geographic:
            lon_slices = _handle_longitude_selection(
                da[self.X], bbox, self.crs.is_geographic
            )

            if len(lon_slices) == 1:
                # Single slice - normal case
                slicers[self.X] = lon_slices[0]
                result = da.sel(slicers)
            else:
                # Multiple slices - bbox crosses 360°/0° boundary
                results = []
                for lon_slice in lon_slices:
                    subset_slicers = slicers.copy()
                    subset_slicers[self.X] = lon_slice
                    results.append(da.sel(subset_slicers))
                # Concatenate along longitude dimension
                result = xr.concat(results, dim=self.X)

        else:
            # Non-geographic coordinates
            slicers[self.X] = slice(bbox.west, bbox.east)
            result = da.sel(slicers)

        # Apply smart longitude conversion only when needed for coordinate system compatibility
        # if self.crs.is_geographic:
        #     result = ensure_continuous_longitudes(result, self.X, bbox)

        return result

    def pad_bbox(self, bbox: pyproj.aoi.BBox, da: xr.DataArray) -> pyproj.aoi.BBox:
        """Extend bbox slightly to account for discrete coordinate sampling."""
        return pad_bbox(bbox, da, self.X, self.Y)


@dataclass(kw_only=True)
class Curvilinear(GridSystem):
    """2D horizontal grid defined by two 2D arrays."""

    crs: pyproj.CRS
    bbox: pyproj.aoi.BBox
    X: str
    Y: str
    indexes: tuple[xr.Index, ...]

    def sel(self, da: xr.DataArray, *, bbox: pyproj.aoi.BBox) -> xr.DataArray:
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

    def pad_bbox(self, bbox: pyproj.aoi.BBox, da: xr.DataArray) -> pyproj.aoi.BBox:
        """Extend bbox slightly to account for discrete coordinate sampling."""
        return pad_bbox(bbox, da, self.X, self.Y)

    # def sel_ndpoint(self, da: xr.DataArray, *, bbox: pyproj.aoi.BBox) -> xr.DataArray:
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


@dataclass(kw_only=True)
class DGGS(GridSystem):
    cells: str
    indexes: tuple[xr.Index, ...]

    def sel(self, da: xr.DataArray, *, bbox: pyproj.aoi.BBox) -> xr.DataArray:
        """Select a subset of the data array using a bounding box."""
        raise NotImplementedError("sel not implemented for DGGS grids")

    def pad_bbox(self, bbox: pyproj.aoi.BBox, da: xr.DataArray) -> pyproj.aoi.BBox:
        """Extend bbox slightly to account for discrete coordinate sampling."""
        raise NotImplementedError("pad_bbox not implemented for DGGS grids")


def _guess_grid_mapping_and_crs(
    ds: xr.Dataset,
) -> tuple[xr.DataArray | None, pyproj.CRS | None]:
    """
    Returns
    ------
    grid_mapping variable
    pyproj.CRS
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
    return grid_mapping, pyproj.CRS.from_cf(grid_mapping.attrs)


# FIXME: cache here, we'll need some xpublish/booth specific attrs on ds
def _guess_grid_for_dataset(ds: xr.Dataset) -> GridSystem:
    grid_mapping, crs = _guess_grid_mapping_and_crs(ds)
    if crs is not None:
        # This means we are not DGGS for sure.
        # FIXME: we aren't handling the triangular case very explicitly yet.
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

        if Xname is None or Yname is None:
            if grid_mapping is None:
                # FIXME: Add test
                raise ValueError("Grid system could not be inferred.")
            else:
                # we have spatial_ref with GeoTransform hopefully.
                # infer bbox from GeoTransform
                # infer some dimension names
                # assign indexes = (rasterix.RasterIndex,)
                raise NotImplementedError(
                    "Support for raster affine grid system not implemented yet."
                )

        # FIXME: nice error here
        (Xname,) = Xname
        (Yname,) = Yname
        X = ds[Xname]
        Y = ds[Yname]

        # intentionally reduce with Xarray to use numbagg
        bbox = pyproj.aoi.BBox(
            west=X.min().item(),
            east=X.max().item(),
            south=Y.min().item(),
            north=Y.max().item(),
        )
        if X.ndim == 1 and Y.ndim == 1:
            return Rectilinear(
                crs=crs,
                X=Xname,
                Y=Yname,
                bbox=bbox,
                indexes=(ds.xindexes[Xname], ds.xindexes[Yname]),
            )
        elif X.ndim == 2 and Y.ndim == 2:
            # See discussion in https://github.com/pydata/xarray/issues/10572
            return Curvilinear(crs=crs, X=Xname, Y=Yname, bbox=bbox, indexes=tuple())

        else:
            raise RuntimeError(
                f"Unknown grid system: X={Xname!r}, ndim={X.ndim}; Y={Yname!r}, ndim={Y.ndim}"
            )
    else:
        raise NotImplementedError("CRS/grid system not detected")


def guess_grid_system(ds: xr.Dataset, name: str) -> GridSystem:
    try:
        return _guess_grid_for_dataset(ds.cf[[name]])
    except RuntimeError:
        return _guess_grid_for_dataset(ds)
