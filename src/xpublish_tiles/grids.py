import itertools
import warnings
from dataclasses import dataclass
from typing import cast

import numpy as np
import rasterix
from pyproj import CRS
from pyproj.aoi import BBox

import xarray as xr

DEFAULT_CRS = CRS.from_epsg(4326)


def _get_xy_pad(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # FIXME: use numbagg
    x_pad = np.abs(np.diff(x)).max()
    y_pad = np.abs(np.diff(y)).max()
    return x_pad, y_pad


def pad_bbox(bbox: BBox, da: xr.DataArray, *, x_pad: float, y_pad: float) -> BBox:
    """
    Extend bbox slightly to account for discrete coordinate sampling.
    This prevents transparency gaps at tile edges due to coordinate resolution.

    The function ensures that the padded bbox does not cross the anti-meridian
    by checking if padding would cause west > east.
    """
    padded_west = float(bbox.west - x_pad)
    padded_east = float(bbox.east + x_pad)

    # Check if padding would cause anti-meridian crossing
    # This happens when the padded west > padded east
    if padded_west > padded_east:
        # Don't pad in the x direction to avoid crossing
        padded_west = float(bbox.west)
        padded_east = float(bbox.east)

    return BBox(
        west=padded_west,
        east=padded_east,
        south=float(bbox.south - y_pad),
        north=float(bbox.north + y_pad),
    )


def is_rotated_pole(crs: CRS) -> bool:
    return crs.to_cf().get("grid_mapping_name") == "rotated_latitude_longitude"


def _handle_longitude_selection(lon_coord: xr.DataArray, bbox: BBox) -> tuple[slice, ...]:
    """
    Handle longitude coordinate selection with support for different conventions.

    This function handles coordinate convention conversion between bbox (-180→180) and data
    (which may be 0→360), as well as anti-meridian crossing bboxes that can occur after
    coordinate transformation from Web Mercator to geographic coordinates.

    Parameters
    ----------
    lon_coord : xr.DataArray
        The longitude/X coordinate array
    bbox : BBox
        Bounding box for selection (may cross anti-meridian after coordinate transformation)

    Returns
    -------
    tuple[slice, ...]
        Tuple of slices for coordinate selection. Usually one slice, but two slices
        when bbox crosses the anti-meridian or 360°/0° boundary.
    """
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
            slice1 = slice(bbox_west_360, 360)
            slice2 = slice(0, bbox_east_360)
            return (slice1, slice2)
        else:
            # Single range in 0→360 convention - but need to convert coordinates for tiles at -180° boundary
            single_slice = slice(bbox_west_360, bbox_east_360)
            return (single_slice,)
    else:
        # Use original bbox coordinates (data is -180→180 or no negative bbox values)
        return (slice(bbox_west, bbox_east),)


class GridSystem:
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

    def sel(self, da: xr.DataArray, *, bbox: BBox) -> xr.DataArray:
        """Select a subset of the data array using a bounding box."""
        raise NotImplementedError("Subclasses must implement sel method")

    def pad_bbox(self, bbox: BBox, da: xr.DataArray) -> BBox:
        """Extend bbox slightly to account for discrete coordinate sampling."""
        raise NotImplementedError("Subclasses must implement pad_bbox method")


class RectilinearSelMixin:
    """Mixin for generic rectilinear .sel"""

    def sel(
        self,
        *,
        da: xr.DataArray,
        bbox: BBox,
        xindex: xr.Index,
        yindex: xr.Index,
        y_is_increasing: bool,
    ) -> xr.DataArray:
        if y_is_increasing:
            yslice = slice(bbox.south, bbox.north)
        else:
            yslice = slice(bbox.north, bbox.south)

        slicers = yindex.sel({self.Y: yslice}).dim_indexers
        if self.crs.is_geographic:
            lon_slices = _handle_longitude_selection(da[self.X], bbox)
            if len(lon_slices) == 1:
                # Single slice - normal case
                slicers |= xindex.sel({self.X: lon_slices[0]}).dim_indexers
                result = da.isel(slicers)
            else:
                # Multiple slices - bbox crosses 360°/0° boundary
                results = []
                for lon_slice in lon_slices:
                    subset_slicers = slicers.copy()
                    subset_slicers.update(xindex.sel({self.X: lon_slice}).dim_indexers)
                    results.append(da.isel(subset_slicers))
                # Concatenate along longitude dimension
                result = xr.concat(results, dim=self.X)

        else:
            # Non-geographic coordinates
            slicers |= xindex.sel({self.X: slice(bbox.west, bbox.east)}).dim_indexers
            result = da.isel(slicers)
        return result


@dataclass(kw_only=True)
class RasterAffine(RectilinearSelMixin, GridSystem):
    """2D horizontal grid defined by an affine transform."""

    crs: CRS
    bbox: BBox
    X: str
    Y: str
    indexes: tuple[rasterix.RasterIndex]

    def pad_bbox(self, bbox: BBox, da: xr.DataArray) -> BBox:
        """Extend bbox slightly to account for discrete coordinate sampling."""
        (index,) = self.indexes
        affine = index.transform()
        return pad_bbox(bbox, da, x_pad=affine.a, y_pad=affine.e)

    def sel(self, da: xr.DataArray, *, bbox: BBox) -> xr.DataArray:
        (index,) = self.indexes
        affine = index.transform()
        return super().sel(
            da=da,
            bbox=bbox,
            xindex=index,
            yindex=index,
            y_is_increasing=affine.e > 0,
        )


@dataclass(kw_only=True)
class Rectilinear(RectilinearSelMixin, GridSystem):
    """2D horizontal grid defined by two explicit 1D basis vectors."""

    crs: CRS
    bbox: BBox
    X: str
    Y: str
    indexes: tuple[xr.indexes.PandasIndex, xr.indexes.PandasIndex]

    def sel(self, da: xr.DataArray, *, bbox: BBox) -> xr.DataArray:
        """
        Select a subset of the data array using a bounding box.

        This method handles coordinate selection for rectilinear grids, automatically
        converting between different longitude conventions (0→360 vs -180→180) and
        ensuring consistent output coordinates in -180→180 format.

        Web Mercator tiles are guaranteed never to cross the anti-meridian, which
        simplifies the longitude handling significantly compared to arbitrary bounding boxes.
        """
        assert self.X in da.xindexes and self.Y in da.xindexes
        assert isinstance(da.xindexes[self.Y], xr.indexes.PandasIndex)
        y_index = cast(xr.indexes.PandasIndex, da.xindexes[self.Y])
        return super().sel(
            da=da,
            bbox=bbox,
            xindex=self.indexes[0],  # FIXME: dict instead?
            yindex=self.indexes[1],  # FIXME: dict instead?
            y_is_increasing=y_index.index.is_monotonic_increasing,
        )

    def pad_bbox(self, bbox: BBox, da: xr.DataArray) -> BBox:
        """
        Extend bbox by maximum coordinate spacing on each side
        This is needed for high zoom tiles smaller than coordinate spacing
        """
        x_pad, y_pad = _get_xy_pad(da[self.X].data, da[self.Y].data)
        return pad_bbox(bbox, da, x_pad=x_pad, y_pad=y_pad)


@dataclass(kw_only=True)
class Curvilinear(GridSystem):
    """2D horizontal grid defined by two 2D arrays."""

    crs: CRS
    bbox: BBox
    X: str
    Y: str
    indexes: tuple[xr.Index, ...]

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

    def pad_bbox(self, bbox: BBox, da: xr.DataArray) -> BBox:
        """Extend bbox slightly to account for discrete coordinate sampling."""
        x_pad, y_pad = _get_xy_pad(da[self.X].data, da[self.Y].data)
        return pad_bbox(bbox, da, x_pad=x_pad, y_pad=y_pad)

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


@dataclass(kw_only=True)
class DGGS(GridSystem):
    cells: str
    indexes: tuple[xr.Index, ...]

    def sel(self, da: xr.DataArray, *, bbox: BBox) -> xr.DataArray:
        """Select a subset of the data array using a bounding box."""
        raise NotImplementedError("sel not implemented for DGGS grids")

    def pad_bbox(self, bbox: BBox, da: xr.DataArray) -> BBox:
        """Extend bbox slightly to account for discrete coordinate sampling."""
        raise NotImplementedError("pad_bbox not implemented for DGGS grids")


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
            # FIXME: figure out how to do this only once, ever
            # FIXME: let's be a little more targeted in what we are guessing
            ds = ds.cf.guess_coord_axis()
            Xname, Yname = guess_coordinate_vars(ds, crs)

        # TODO: we might use rasterix for when there are explicit coords too?
        if Xname is None or Yname is None:
            if grid_mapping is None:
                raise RuntimeError("Grid system could not be inferred.")
            else:
                # FIXME: use cf-xarray regexes here?
                for x, y in (("x", "y"), ("lon", "lat"), ("longitude", "latitude")):
                    if x in ds.dims and y in ds.dims:
                        ds = rasterix.assign_index(ds, x_dim=x, y_dim=y)
                        index = ds.xindexes[x]
                        return RasterAffine(
                            crs=crs,
                            X=x,
                            Y=y,
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

        # intentionally reduce with Xarray to use numbagg
        # FIXME: just use numbagg directly
        # FIXME: compile ahead of time by reducing a 1D and 2D array
        bbox = BBox(
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
                indexes=(
                    cast(xr.indexes.PandasIndex, ds.xindexes[Xname]),
                    cast(xr.indexes.PandasIndex, ds.xindexes[Yname]),
                ),
            )
        elif X.ndim == 2 and Y.ndim == 2:
            # See discussion in https://github.com/pydata/xarray/issues/10572
            return Curvilinear(crs=crs, X=Xname, Y=Yname, bbox=bbox, indexes=tuple())

        else:
            raise RuntimeError(
                f"Unknown grid system: X={Xname!r}, ndim={X.ndim}; Y={Yname!r}, ndim={Y.ndim}"
            )
    else:
        raise RuntimeError("CRS/grid system not detected")


# FIXME: cache here, we'll need some xpublish/booth specific attrs on ds
def guess_grid_system(ds: xr.Dataset, name: str) -> GridSystem:
    try:
        return _guess_grid_for_dataset(ds.cf[[name]])
    except RuntimeError:
        return _guess_grid_for_dataset(ds)
