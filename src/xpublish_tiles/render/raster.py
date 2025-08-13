import contextlib
import importlib.util
import io
import logging
import threading
import time
from typing import TYPE_CHECKING, cast

import datashader as dsh  # type: ignore
import datashader.reductions  # type: ignore
import datashader.transfer_functions as tf  # type: ignore
import matplotlib as mpl  # type: ignore
import numbagg
import numpy as np
from scipy.interpolate import NearestNDInterpolator

import xarray as xr
from xpublish_tiles.grids import Curvilinear, RasterAffine, Rectilinear
from xpublish_tiles.render import Renderer, register_renderer
from xpublish_tiles.types import (
    ContinuousData,
    DiscreteData,
    ImageFormat,
    NullRenderContext,
    PopulatedRenderContext,
    RenderContext,
)

# Only use lock if tbb is not available
HAS_TBB = importlib.util.find_spec("tbb") is not None
LOCK = contextlib.nullcontext() if HAS_TBB else threading.Lock()
logger = logging.getLogger("xpublish-tiles")

# without controlling the seed, output for categorical data is non-deterministic
np.random.seed(1234)


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
    print(f"constructing interpolator: {time.time() - tic} seconds")
    print(f"interpolating from {da.shape} to {newY.size}x{newX.size}")
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
    print(f"interpolating: {time.time() - tic} seconds")
    return new


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
    # tic = time.time()
    cvs = dsh.Canvas(
        *newshape,
        x_range=(xmin - dx / 2, xmax + dx / 2),
        y_range=(ymin - dy / 2, ymax + dy / 2),
    )
    res = cvs.quadmesh(da, x="x", y="y", agg=dsh.reductions.first(cast(str, da.name)))
    # print(f"interpolating from {da.shape} to {newshape}: {time.time() - tic} seconds")
    return res


@register_renderer
class DatashaderRasterRenderer(Renderer):
    def validate(self, context: dict[str, "RenderContext"]):
        assert len(context) == 1

    def maybe_cast_data(self, data) -> xr.DataArray:  # type: ignore[name-defined]
        return data.astype(np.float64, copy=False)

    def render(
        self,
        *,
        contexts: dict[str, "RenderContext"],
        buffer: io.BytesIO,
        width: int,
        height: int,
        cmap: str,
        colorscalerange: tuple[float, float] | None = None,
        format: ImageFormat = ImageFormat.PNG,
    ):
        # Handle "default" alias
        if cmap == "default":
            cmap = self.default_variant()

        self.validate(contexts)
        (context,) = contexts.values()
        if isinstance(context, NullRenderContext):
            raise ValueError("no overlap with requested bbox.")
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
                    data = nearest_on_uniform_grid_quadmesh(context.da, grid.X, grid.Y)
                    mesh = cvs.raster(
                        data,
                        interpolate="nearest",
                        agg=dsh.reductions.mode(cast(str, data.name)),
                    )
            else:
                data = self.maybe_cast_data(context.da)
                mesh = cvs.quadmesh(data, x=grid.X, y=grid.Y)
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
            shaded = tf.shade(
                mesh,
                cmap=mpl.colormaps.get_cmap(cmap),
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
                kwargs["cmap"] = mpl.colormaps.get_cmap(cmap)
                kwargs["span"] = (
                    min(context.datatype.values),
                    max(context.datatype.values),
                )
            shaded = tf.shade(mesh, how="linear", **kwargs)
        else:
            raise NotImplementedError(f"Unsupported datatype: {type(context.datatype)}")

        im = shaded.to_pil()
        im.save(buffer, format=str(format))

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
