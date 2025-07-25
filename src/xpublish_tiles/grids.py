import itertools
import warnings
from dataclasses import dataclass
from typing import cast

import numpy as np
import pyproj
import pyproj.aoi

import xarray as xr

DEFAULT_CRS = pyproj.CRS.from_epsg(4326)


def is_rotated_pole(crs: pyproj.CRS) -> bool:
    return crs.to_cf().get("grid_mapping_name") == "rotated_latitude_longitude"


def _normalize_longitudes_to_180(result: xr.DataArray, x_coord_name: str) -> xr.DataArray:
    """
    Normalize longitude coordinates to -180→180 convention for consistent output.

    This function handles the conversion from 0→360 coordinates to -180→180,
    while preserving spatial continuity for coordinates that span the 180°/360° boundary.

    Parameters
    ----------
    result : xr.DataArray
        Data array with longitude coordinates to normalize
    x_coord_name : str
        Name of the longitude coordinate dimension

    Returns
    -------
    xr.DataArray
        Data array with normalized longitude coordinates in -180→180 format
    """
    if x_coord_name not in result.coords:
        return result

    lon_coord = result.coords[x_coord_name]
    lon_values = lon_coord.data

    # Check if coordinates span the 180°/360° boundary before conversion
    original_min, original_max = lon_values.min(), lon_values.max()
    spans_180_boundary = (
        original_min < 180 and original_max >= 180 and (original_max - original_min) < 180
    )

    # Convert 0→360 to -180→180
    converted_values = ((lon_values + 180) % 360) - 180

    # If original data spanned 180° boundary and conversion created apparent anti-meridian crossing,
    # adjust the converted values to maintain spatial continuity
    if spans_180_boundary:
        converted_min, converted_max = (
            converted_values.min(),
            converted_values.max(),
        )
        if (
            converted_max - converted_min > 180
        ):  # Apparent anti-meridian crossing after conversion
            # Convert negative values back to positive to maintain continuity
            converted_values = np.where(
                converted_values < 0, converted_values + 360, converted_values
            )

    return result.assign_coords(
        {x_coord_name: (lon_coord.dims, converted_values, lon_coord.attrs)}
    )


def _handle_longitude_selection(
    lon_coord: xr.DataArray, bbox: pyproj.aoi.BBox, is_geographic: bool
) -> slice:
    """
    Handle longitude coordinate selection with support for different conventions.

    Since Web Mercator tiles never cross the anti-meridian, this function only needs
    to handle coordinate convention conversion between bbox (-180→180) and data
    (which may be 0→360).

    Parameters
    ----------
    lon_coord : xr.DataArray
        The longitude/X coordinate array
    bbox : pyproj.aoi.BBox
        Bounding box for selection (guaranteed not to cross anti-meridian)
    is_geographic : bool
        Whether the CRS is geographic (True) or projected (False)

    Returns
    -------
    slice
        Single slice for coordinate selection
    """
    if not is_geographic:
        # For projected coordinates, treat X as regular coordinate
        return slice(bbox.west, bbox.east)

    # For geographic coordinates, handle longitude wrapping
    lon_values = lon_coord.data
    lon_min, lon_max = lon_values.min().item(), lon_values.max().item()

    # Determine if data uses 0→360 or -180→180 convention
    uses_0_360 = lon_min >= 0 and lon_max > 180

    # Convert bbox to match data convention
    bbox_west = bbox.west
    bbox_east = bbox.east

    if uses_0_360:
        # Data is 0→360, bbox is typically -180→180
        # Convert negative longitudes to 0→360 range
        if bbox_west < 0:
            bbox_west += 360
        if bbox_east < 0:
            bbox_east += 360

    return slice(bbox_west, bbox_east)


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
    """2D horizontal grid defined by an affine transfo."""

    crs: pyproj.CRS
    bbox: pyproj.aoi.BBox
    indexes: tuple[xr.Index, ...]


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

        # Assert that bbox doesn't cross anti-meridian (Web Mercator tiles never do)
        assert (
            bbox.west <= bbox.east
        ), f"BBox crosses anti-meridian: west={bbox.west} > east={bbox.east}"

        slicers = {}
        y_index = cast(xr.indexes.PandasIndex, da.xindexes[self.Y])
        if y_index.index.is_monotonic_increasing:
            slicers[self.Y] = slice(bbox.south, bbox.north)
        else:
            slicers[self.Y] = slice(bbox.north, bbox.south)

        # Handle longitude (X coordinate) using helper function
        lon_slice = _handle_longitude_selection(da[self.X], bbox, self.crs.is_geographic)
        slicers[self.X] = lon_slice
        result = da.sel(slicers)

        # Convert longitude coordinates to -180→180 convention for consistent output
        if self.crs.is_geographic:
            result = _normalize_longitudes_to_180(result, self.X)

        return result


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

        # Convert longitude coordinates to -180→180 convention for consistent output
        if self.crs.is_geographic:
            result = _normalize_longitudes_to_180(result, self.X)

        return result

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
