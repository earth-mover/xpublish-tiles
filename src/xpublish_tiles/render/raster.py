import io
from typing import TYPE_CHECKING, cast

import datashader as dsh  # type: ignore
import datashader.reductions  # type: ignore
import datashader.transfer_functions as tf  # type: ignore
import matplotlib as mpl  # type: ignore
import matplotlib.colors as mcolors
import numbagg
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import NearestNDInterpolator

import xarray as xr
from xpublish_tiles.grids import Curvilinear, GridSystem2D, Triangular
from xpublish_tiles.lib import (
    MissingParameterError,
    create_colormap_from_dict,
    create_listed_colormap_from_dict,
)
from xpublish_tiles.logger import get_context_logger, log_duration
from xpublish_tiles.render import Renderer, register_renderer, render_error_image
from xpublish_tiles.types import (
    ContinuousData,
    DiscreteData,
    ImageFormat,
    NullRenderContext,
    PopulatedRenderContext,
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
        context_logger=None,
        colormap: dict[str, str] | None = None,
    ):
        # Use the passed context logger or fallback to get_context_logger
        logger = context_logger if context_logger is not None else get_context_logger()
        # Handle "default" alias
        if variant == "default":
            variant = self.default_variant()

        self.validate(contexts)
        (context,) = contexts.values()
        if isinstance(context, NullRenderContext):
            logger.debug("☐ No data")
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
        data = self.maybe_cast_data(context.da)

        if isinstance(context.grid, GridSystem2D):
            # Use the actual coordinate names from the grid system
            grid = cast(GridSystem2D, context.grid)
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
                with NUMBA_THREADING_LOCK:
                    with log_duration(
                        f"nearest neighbour regridding (discrete) {data.shape}",
                        "⊞",
                        logger,
                    ):
                        data = nearest_on_uniform_grid_quadmesh(data, grid.X, grid.Y)
                    with log_duration(
                        f"render (discrete) {data.shape} raster", "🎨", logger
                    ):
                        mesh = cvs.raster(
                            data,
                            interpolate="nearest",
                            agg=dsh.reductions.mode(cast(str, data.name)),
                        )
            else:
                data = self.maybe_cast_data(context.da)
                with log_duration(
                    f"render (continuous) {data.shape} quadmesh", "🎨", logger
                ):
                    # Lock is only used when tbb is not available (e.g., on macOS)
                    # AND if we use the rectilinear or raster code path
                    with NUMBA_THREADING_LOCK:
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

        if isinstance(context.datatype, ContinuousData):
            if colorscalerange is None:
                valid_min = context.datatype.valid_min
                valid_max = context.datatype.valid_max
                if valid_min is not None and valid_max is not None:
                    colorscalerange = (valid_min, valid_max)
                else:
                    raise MissingParameterError(
                        "`colorscalerange` must be specified when array does not have valid_min and valid_max attributes specified."
                    )

            # Use custom colormap if provided, otherwise use variant
            if colormap is not None:
                cmap = create_colormap_from_dict(colormap)
            else:
                cmap = mpl.colormaps.get_cmap(variant)

            with np.errstate(invalid="ignore"):
                shaded = tf.shade(
                    mesh,
                    cmap=cmap,
                    how="linear",
                    span=colorscalerange,
                )
        elif isinstance(context.datatype, DiscreteData):
            kwargs = {}
            flag_values = context.datatype.values
            # Custom colormap overrides flag_colors
            if colormap is not None:
                kwargs["color_key"] = create_listed_colormap_from_dict(
                    colormap, flag_values
                )
            elif context.datatype.colors is not None:
                kwargs["color_key"] = dict(
                    zip(context.datatype.values, context.datatype.colors, strict=True)
                )
            else:
                minv = min(flag_values)
                maxv = max(flag_values)
                cmap = mpl.colormaps.get_cmap(variant)
                kwargs["color_key"] = {
                    v: mcolors.to_hex(cmap((v - minv) / maxv)) for v in flag_values
                }
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
        variants = [name for name in sorted(colormaps) if not name.endswith("_r")]
        variants.append("custom")
        return variants

    @staticmethod
    def default_variant() -> str:
        return "viridis"

    @classmethod
    def describe_style(cls, variant: str) -> dict[str, str]:
        if variant == "custom":
            return {
                "id": f"{cls.style_id()}/{variant}",
                "title": "Raster - Custom",
                "description": "Raster rendering with a custom colormap provided via the 'colormap' parameter",
            }
        return {
            "id": f"{cls.style_id()}/{variant}",
            "title": f"Raster - {variant.title()}",
            "description": f"Raster rendering using {variant} colormap",
        }
