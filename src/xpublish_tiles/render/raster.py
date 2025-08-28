import contextlib
import importlib.util
import io
import threading
import time
from typing import TYPE_CHECKING, cast

import datashader as dsh  # type: ignore
import datashader.reductions  # type: ignore
import datashader.transfer_functions as tf  # type: ignore
import matplotlib as mpl  # type: ignore
import numbagg
import numpy as np
from PIL import Image
from scipy.interpolate import NearestNDInterpolator

import xarray as xr
from xpublish_tiles.grids import Curvilinear, RasterAffine, Rectilinear
from xpublish_tiles.logger import logger
from xpublish_tiles.render import Renderer, register_renderer, render_error_image
from xpublish_tiles.types import (
    ContinuousData,
    DiscreteData,
    ImageFormat,
    NullRenderContext,
    PopulatedRenderContext,
    RenderContext,
)
from xpublish_tiles.utils import time_debug

# Only use lock if tbb is not available
HAS_TBB = importlib.util.find_spec("tbb") is not None
LOCK = contextlib.nullcontext() if HAS_TBB else threading.Lock()


@time_debug
def nearest_on_uniform_grid_scipy(da: xr.DataArray, Xdim: str, Ydim: str) -> xr.DataArray:
    """This is quite slow. 10s for a 2000x3000 array"""
    X, Y = da[Xdim], da[Ydim]
    dx = abs(X.diff(Xdim).median().data)
    dy = abs(Y.diff(Ydim).median().data)
    newX = np.arange(numbagg.nanmin(X.data), numbagg.nanmax(X.data) + dx, dx)
    newY = np.arange(numbagg.nanmin(Y.data), numbagg.nanmax(Y.data) + dy, dy)
    tic = time.time()
    interpolator = NearestNDInterpolator(
        np.stack([X.data.ravel(), Y.data.ravel()], axis=-1),
        da.data.ravel(),
    )
    logger.debug(f"constructing interpolator: {time.time() - tic} seconds")
    logger.debug(f"interpolating from {da.shape} to {newY.size}x{newX.size}")
    tic = time.time()
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
    logger.debug(f"interpolating: {time.time() - tic} seconds")
    return new


@time_debug
def nearest_on_uniform_grid_quadmesh(
    da: xr.DataArray, Xdim: str, Ydim: str
) -> xr.DataArray:
    """
    This is a trick; for upsampling, datashader will do nearest neighbor resampling.
    """
    tic = time.time()
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
    print(
        f"Tiles : interpolating categorical from {da.shape} to {newshape}: {time.time() - tic} "
    )
    return res


@register_renderer
class DatashaderRasterRenderer(Renderer):
    def validate(self, context: dict[str, "RenderContext"]):
        assert len(context) == 1

    def maybe_cast_data(self, data) -> xr.DataArray:  # type: ignore[name-defined]
        dtype = data.dtype
        totype = str(dtype.str)
        # numba only supports float32 and float64. upcast everything else
        # https://numba.readthedocs.io/en/stable/reference/types.html#numbers
        if dtype.kind == "f" and dtype.itemsize < 4:
            totype = totype[:-1] + "4"
        return data.astype(totype, copy=False)

    @time_debug
    def render(
        self,
        *,
        contexts: dict[str, "RenderContext"],
        buffer: io.BytesIO,
        width: int,
        height: int,
        variant: str,
        colorscalerange: tuple[float, float] | None = None,
        format: ImageFormat = ImageFormat.PNG,
    ):
        # Handle "default" alias
        if variant == "default":
            variant = self.default_variant()

        self.validate(contexts)
        (context,) = contexts.values()
        if isinstance(context, NullRenderContext):
            im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            im.save(buffer, format=str(format))
            return

        if TYPE_CHECKING:
            assert isinstance(context, PopulatedRenderContext)

        bbox = context.bbox
        cvs = dsh.Canvas(
            plot_height=height,
            plot_width=width,
            x_range=(bbox.west, bbox.east),
            y_range=(bbox.south, bbox.north),
        )

        if isinstance(context.grid, RasterAffine | Rectilinear | Curvilinear):
            # Use the actual coordinate names from the grid system
            grid = cast(RasterAffine | Rectilinear | Curvilinear, context.grid)
            if isinstance(context.datatype, DiscreteData):
                if isinstance(grid, Curvilinear):
                    # FIXME: we'll need to track Xdim, Ydim explicitly no dims: tuple[str]
                    raise NotImplementedError
                # datashader only supports rectilinear input for the mode aggregation;
                # Our input coordinates are most commonly "curvilinear", so
                # we nearest-neighbour resample to a rectilinear grid, and the use
                # the mode aggregation.
                # https://github.com/holoviz/datashader/issues/1435
                # Lock is only used when tbb is not available (e.g., on macOS)
                with LOCK:
                    data = self.maybe_cast_data(context.da)
                    data = nearest_on_uniform_grid_quadmesh(data, grid.X, grid.Y)
                    mesh = cvs.raster(
                        data,
                        interpolate="nearest",
                        agg=dsh.reductions.mode(cast(str, data.name)),
                    )
            else:
                data = self.maybe_cast_data(context.da)
                ydata = data[grid.Y]
                if (
                    ydata.ndim == 1
                    and
                    # decreasing coordinates are buggy
                    # https://github.com/holoviz/datashader/issues/1438
                    (ydata[1] - ydata[0] < 0)
                ):
                    data = data.isel({grid.Ydim: slice(None, None, -1)})
                if ydata.ndim == 1 and (
                    # for high zoom tiles are usually 3x3
                    # use curvilinear code path here
                    # https://github.com/holoviz/datashader/issues/1439
                    ydata.size < 10
                ):
                    data = data.assign_coords(
                        dict(
                            zip(
                                (grid.X, grid.Y),
                                xr.broadcast(data[grid.X], data[grid.Y]),
                                strict=False,
                            )
                        )
                    )
                # Lock is only used when tbb is not available (e.g., on macOS)
                # AND if we use the rectilinear or raster code path
                with LOCK if data[grid.Y].ndim == 1 else contextlib.nullcontext():
                    mesh = cvs.quadmesh(
                        data.transpose(grid.Ydim, grid.Xdim), x=grid.X, y=grid.Y
                    )
        else:
            raise NotImplementedError(
                f"Grid type {type(context.grid)} not supported by DatashaderRasterRenderer"
            )

        if isinstance(context.datatype, ContinuousData):
            if colorscalerange is None:
                valid_min = context.datatype.valid_min
                valid_max = context.datatype.valid_max
                if valid_min is not None and valid_max is not None:
                    colorscalerange = (valid_min, valid_max)
                else:
                    raise ValueError(
                        "`colorscalerange` must be specified when array does not have valid_min and valid_max attributes specified."
                    )
            with np.errstate(invalid="ignore"):
                shaded = tf.shade(
                    mesh,
                    cmap=mpl.colormaps.get_cmap(variant),
                    how="linear",
                    span=colorscalerange,
                )
        elif isinstance(context.datatype, DiscreteData):
            kwargs = {}
            if context.datatype.colors is not None:
                kwargs["color_key"] = dict(
                    zip(context.datatype.values, context.datatype.colors, strict=True)
                )
            else:
                kwargs["cmap"] = mpl.colormaps.get_cmap(variant)
                kwargs["span"] = (
                    min(context.datatype.values),
                    max(context.datatype.values),
                )
            with np.errstate(invalid="ignore"):
                shaded = tf.shade(mesh, how="linear", **kwargs)
        else:
            raise NotImplementedError(f"Unsupported datatype: {type(context.datatype)}")

        im = shaded.to_pil()
        im.save(buffer, format=str(format))

    def render_error(
        self,
        *,
        buffer: io.BytesIO,
        width: int,
        height: int,
        message: str,
        format: ImageFormat = ImageFormat.PNG,
        cmap: str = "",
        colorscalerange: tuple[float, float] | None = None,
        **kwargs,
    ):
        """Render an error tile with the given message."""
        error_buffer = render_error_image(
            message, width=width, height=height, format=format
        )
        buffer.write(error_buffer.getvalue())
        error_buffer.close()

    @staticmethod
    def style_id() -> str:
        return "raster"

    @staticmethod
    def supported_variants() -> list[str]:
        colormaps = list(mpl.colormaps)
        return [name for name in sorted(colormaps) if not name.endswith("_r")]

    @staticmethod
    def default_variant() -> str:
        return "viridis"

    @classmethod
    def describe_style(cls, variant: str) -> dict[str, str]:
        return {
            "id": f"{cls.style_id()}/{variant}",
            "title": f"Raster - {variant.title()}",
            "description": f"Raster rendering using {variant} colormap",
        }
