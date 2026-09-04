import io
from numbers import Number
from typing import cast

import datashader as dsh
import datashader.reductions
import matplotlib.colors as mcolors
import numbagg
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import NearestNDInterpolator

import xarray as xr
from xpublish_tiles.grids import Curvilinear, Geostationary, GridSystem2D, Triangular
from xpublish_tiles.lib import (
    maybe_cast_data,
)
from xpublish_tiles.logger import get_context_logger, log_duration
from xpublish_tiles.render import DatashaderRenderer, register_renderer
from xpublish_tiles.render.kernels import footprint_mode, offdisk_quad_mask
from xpublish_tiles.types import (
    ContinuousData,
    DiscreteData,
    ImageFormat,
    RenderContext,
)
from xpublish_tiles.utils import NUMBA_THREADING_LOCK


def nearest_on_uniform_grid_scipy(da: xr.DataArray, Xdim: str, Ydim: str) -> xr.DataArray:
    """This is quite slow. 10s for a 2000x3000 array"""
    X, Y = da[Xdim], da[Ydim]
    dx = abs(X.diff(Xdim).median().data)
    dy = abs(Y.diff(Ydim).median().data)
    newX = np.arange(numbagg.nanmin(X.data), numbagg.nanmax(X.data) + dx, dx)
    newY = np.arange(numbagg.nanmin(Y.data), numbagg.nanmax(Y.data) + dy, dy)

    interpolator = NearestNDInterpolator(
        np.stack([X.data.ravel(), Y.data.ravel()], axis=-1),
        da.data.ravel(),
    )

    logger = get_context_logger()
    logger.debug(f"interpolating from {da.shape} to {newY.size}x{newX.size}")

    new = xr.DataArray(
        interpolator(*np.meshgrid(newX, newY)),
        dims=(Ydim, Xdim),
        name=da.name,
        # this dx, dy offset is weird but it gets raster to almost look like quadmesh
        # FIXME: I should need to offset this with `-dx` and `-dy`
        # but that leads to transparent pixels at high res
        # coords=dict(x=("x", newX - dx/2), y=("y", newY - dy/2)),
        coords=dict(x=("x", newX), y=("y", newY)),
    )
    return new


def _range_color_to_rgba(color: str) -> tuple[int, int, int, int]:
    if color == "transparent":
        return (0, 0, 0, 0)
    rgba = mcolors.to_rgba(color)
    return tuple(round(channel * 255) for channel in rgba)


def _apply_out_of_range_colors(
    image: Image.Image,
    mesh: xr.DataArray,
    colorscalerange: tuple[Number, Number] | None,
    abovemaxcolor: str | None,
    belowmincolor: str | None,
) -> Image.Image:
    if colorscalerange is None:
        return image

    apply_over = abovemaxcolor not in (None, "extend")
    apply_under = belowmincolor not in (None, "extend")
    if not apply_over and not apply_under:
        return image

    mesh_values = np.asarray(mesh)
    if mesh_values.size == 0:
        return image

    finite_mask = np.isfinite(mesh_values)
    under_mask = finite_mask & (mesh_values < colorscalerange[0]) if apply_under else None
    over_mask = finite_mask & (mesh_values > colorscalerange[1]) if apply_over else None

    if under_mask is None or not np.any(under_mask):
        under_mask = None
    if over_mask is None or not np.any(over_mask):
        over_mask = None
    if under_mask is None and over_mask is None:
        return image

    should_flip = False
    if mesh.dims:
        y_dim = mesh.dims[0]
        if y_dim in mesh.coords:
            y_vals = np.asarray(mesh.coords[y_dim])
            if y_vals.size >= 2 and y_vals[0] < y_vals[-1]:
                # Datashader's PIL output is top-down; flip if y increases upward.
                should_flip = True

    if should_flip:
        if under_mask is not None:
            under_mask = np.flipud(under_mask)
        if over_mask is not None:
            over_mask = np.flipud(over_mask)

    if image.mode != "RGBA":
        image = image.convert("RGBA")
    img_array = np.array(image)

    if under_mask is not None and belowmincolor is not None:
        img_array[under_mask] = _range_color_to_rgba(belowmincolor)
    if over_mask is not None and abovemaxcolor is not None:
        img_array[over_mask] = _range_color_to_rgba(abovemaxcolor)

    return Image.fromarray(img_array, mode="RGBA")


def nearest_on_uniform_grid_quadmesh(
    da: xr.DataArray, Xdim: str, Ydim: str
) -> xr.DataArray:
    """
    This is a trick; for upsampling, datashader will do nearest neighbor resampling.
    """
    X, Y = da[Xdim], da[Ydim]
    dx = abs(X.diff(Xdim).median().data)
    dy = abs(Y.diff(Ydim).median().data)
    xmin, xmax = numbagg.nanmin(X.data), numbagg.nanmax(X.data)
    ymin, ymax = numbagg.nanmin(Y.data), numbagg.nanmax(Y.data)
    newshape = (
        round(abs((xmax - xmin) / dx)) + 1,
        round(abs((ymax - ymin) / dy)) + 1,
    )
    cvs = dsh.Canvas(
        *newshape,
        x_range=(xmin - dx / 2, xmax + dx / 2),
        y_range=(ymin - dy / 2, ymax + dy / 2),
    )
    res = cvs.quadmesh(da, x=Xdim, y=Ydim, agg=dsh.reductions.first(cast(str, da.name)))
    return res


def _nearest_index_map(coord: np.ndarray, lo: float, hi: float, n: int) -> np.ndarray:
    """Source index nearest each of ``n`` uniform samples across ``[lo, hi]``.

    ``-1`` marks a sample that falls outside the grid's coverage, so it renders
    transparent instead of smearing the edge cell across the tile.
    """
    centers = lo + (hi - lo) * (np.arange(n) + 0.5) / n
    ascending = coord[-1] >= coord[0]
    axis = coord if ascending else coord[::-1]

    right = np.searchsorted(axis, centers).clip(1, axis.size - 1)
    left = right - 1
    nearest = np.where(centers - axis[left] <= axis[right] - centers, left, right)

    # coverage extends half a cell past the outermost centers
    half_lo = (axis[1] - axis[0]) / 2 if axis.size > 1 else 0.0
    half_hi = (axis[-1] - axis[-2]) / 2 if axis.size > 1 else 0.0
    outside = (centers < axis[0] - half_lo) | (centers > axis[-1] + half_hi)

    if not ascending:
        nearest = axis.size - 1 - nearest
    return np.where(outside, -1, nearest)


def _footprint_index_map(
    coord: np.ndarray, lo: float, hi: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Source index range ``[start, stop)`` whose cell centres fall in each sample.

    Centre-in-sample makes the samples a partition: every source cell in the
    tile belongs to exactly one output pixel, counted once.
    """
    edges = lo + (hi - lo) * np.arange(n + 1) / n
    ascending = coord[-1] >= coord[0]
    axis = coord if ascending else coord[::-1]
    bounds = np.searchsorted(axis, edges)
    if ascending:
        start, stop = bounds[:-1], bounds[1:]
    else:
        start, stop = axis.size - bounds[1:], axis.size - bounds[:-1]
    return start.astype(np.int64), stop.astype(np.int64)


def rasterize_categorical_rectilinear(
    da: xr.DataArray,
    grid: GridSystem2D,
    *,
    datatype: DiscreteData,
    bbox,
    width: int,
    height: int,
) -> xr.DataArray:
    """Rasterize categorical data on 1D X/Y coordinates by footprint mode.

    The counting table spans ``flag_values``; other codes render transparent
    anyway, so they are not counted.
    """
    values = np.asarray(datatype.values, dtype=np.float64)
    offset = np.int64(np.floor(values.min()))
    counts_span = int(np.floor(values.max())) - int(offset) + 1

    xcoord, ycoord = da[grid.X], da[grid.Y]
    (xdim,), (ydim,) = xcoord.dims, ycoord.dims
    source = np.asarray(da.transpose(ydim, xdim).data)
    x, y = np.asarray(xcoord.data), np.asarray(ycoord.data)
    xstart, xstop = _footprint_index_map(x, bbox.west, bbox.east, width)
    ystart, ystop = _footprint_index_map(y, bbox.south, bbox.north, height)
    xnear = _nearest_index_map(x, bbox.west, bbox.east, width).astype(np.int64)
    ynear = _nearest_index_map(y, bbox.south, bbox.north, height).astype(np.int64)

    out = np.empty((height, width), dtype=np.float64)
    footprint_mode(
        source, ystart, ystop, xstart, xstop, ynear, xnear, counts_span, offset, out
    )

    xc = bbox.west + (bbox.east - bbox.west) * (np.arange(width) + 0.5) / width
    yc = bbox.south + (bbox.north - bbox.south) * (np.arange(height) + 0.5) / height
    return xr.DataArray(
        out,
        dims=(ydim, xdim),
        coords={xdim: xc, ydim: yc},
        attrs={
            "res": (xc[1] - xc[0], yc[0] - yc[1]),
            "x_range": (bbox.west, bbox.east),
            "y_range": (bbox.south, bbox.north),
        },
    )


@register_renderer
class DatashaderRasterRenderer(DatashaderRenderer):
    def validate(self, context: dict[str, "RenderContext"]):
        assert len(context) == 1

    def render(
        self,
        *,
        contexts: dict[str, "RenderContext"],
        buffer: io.BytesIO,
        width: int,
        height: int,
        variant: str,
        colorscalerange: tuple[Number, Number] | None = None,
        format: ImageFormat = ImageFormat.PNG,
        context_logger=None,
        colormap: dict[str, str] | None = None,
        abovemaxcolor: str | None = None,
        belowmincolor: str | None = None,
    ):
        prepared = self._prepare_render(
            contexts,
            buffer=buffer,
            width=width,
            height=height,
            variant=variant,
            format=format,
            context_logger=context_logger,
        )
        if prepared is None:
            return
        context, cvs, variant = prepared
        logger = context_logger if context_logger is not None else get_context_logger()
        data = maybe_cast_data(context.da)

        if isinstance(context.grid, GridSystem2D):
            grid = context.grid
            if isinstance(context.datatype, DiscreteData):
                if isinstance(grid, Curvilinear):
                    # FIXME: we'll need to track Xdim, Ydim explicitly no dims: tuple[str]
                    raise NotImplementedError
                # Lock is only used when tbb is not available (e.g., macOS)
                with NUMBA_THREADING_LOCK:
                    if data[grid.X].ndim == 2 or data[grid.Y].ndim == 2:
                        # 2D coords: nearest-neighbour resample to a uniform grid
                        # so the pixel -> cell map factors into 1D index maps.
                        with log_duration(
                            f"nearest neighbour regridding (discrete) {data.shape}",
                            "⊞",
                            logger,
                        ):
                            data = nearest_on_uniform_grid_quadmesh(data, grid.X, grid.Y)
                    with log_duration(
                        f"render (discrete) {data.shape} mode", "🎨", logger
                    ):
                        mesh = rasterize_categorical_rectilinear(
                            data,
                            grid,
                            datatype=context.datatype,
                            bbox=context.bbox,
                            width=width,
                            height=height,
                        )
            else:
                data = maybe_cast_data(context.da)
                if isinstance(grid, Geostationary):
                    xcoord, ycoord = data[grid.X], data[grid.Y]
                    # Transformation of points near the edge of the disk is undefined
                    # and corrupts the rendering. Mask them out but only for
                    # Geostationary to avoid the perf hit.
                    bad = offdisk_quad_mask(
                        np.asarray(xcoord.data), np.asarray(ycoord.data)
                    )
                    data = data.where(xr.DataArray(~bad, dims=xcoord.dims))
                with log_duration(
                    f"render (continuous) {data.shape} quadmesh", "🎨", logger
                ):
                    # Lock is only used when tbb is not available (e.g., on macOS)
                    # AND if we use the rectilinear or raster code path.
                    # errstate: off-disk NaN coords (geostationary) cast to garbage
                    # ints inside datashader's quad scaling; those quads no-op via
                    # the NaN data mask above, so the warning is expected noise.
                    with NUMBA_THREADING_LOCK, np.errstate(invalid="ignore"):
                        mesh = cvs.quadmesh(
                            data.transpose(grid.Ydim, grid.Xdim), x=grid.X, y=grid.Y
                        )
        elif isinstance(context.grid, Triangular):
            with log_duration(f"render (continuous) {data.shape} trimesh", "🔺", logger):
                assert context.ugrid_indexer is not None
                if context.grid.dim in data.coords:
                    # dropping gets us a cheap RangeIndex in the DataFrame
                    # Only drop the dimension coordinate if it exists as a variable
                    data = data.drop_vars(context.grid.dim)
                df = data.to_dataframe()
                mesh = cvs.trimesh(
                    df[[context.grid.X, context.grid.Y, data.name]],
                    pd.DataFrame(
                        context.ugrid_indexer.connectivity, columns=["v0", "v1", "v2"]
                    ),
                )
        else:
            raise NotImplementedError(
                f"Grid type {type(context.grid)} not supported by DatashaderRasterRenderer"
            )

        im = self.shade_mesh(
            mesh,
            context.datatype,
            variant=variant,
            colorscalerange=colorscalerange,
            colormap=colormap,
            abovemaxcolor=abovemaxcolor,
            belowmincolor=belowmincolor,
        )
        if isinstance(context.datatype, ContinuousData):
            im = _apply_out_of_range_colors(
                im, mesh, colorscalerange, abovemaxcolor, belowmincolor
            )

        im.save(buffer, format=str(format))

    @staticmethod
    def style_id() -> str:
        return "raster"
