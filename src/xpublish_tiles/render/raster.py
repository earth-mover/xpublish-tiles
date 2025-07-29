import io
import logging
from typing import TYPE_CHECKING, cast

import datashader as dsh  # type: ignore
import datashader.transfer_functions as tf  # type: ignore
import matplotlib as mpl  # type: ignore
import numpy as np

import xarray as xr
from xpublish_tiles.grids import Curvilinear, Rectilinear
from xpublish_tiles.render import Renderer
from xpublish_tiles.types import (
    DataType,
    ImageFormat,
    NullRenderContext,
    PopulatedRenderContext,
    RenderContext,
)

logger = logging.getLogger("xpublish-tiles")


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
        self.validate(contexts)
        (context,) = contexts.values()
        if isinstance(context, NullRenderContext):
            raise NotImplementedError("no overlap with requested bbox.")
        if TYPE_CHECKING:
            assert isinstance(context, PopulatedRenderContext)
        bbox = context.bbox
        data = self.maybe_cast_data(context.da)
        cvs = dsh.Canvas(
            plot_height=height,
            plot_width=width,
            x_range=(bbox.west, bbox.east),
            y_range=(bbox.south, bbox.north),
        )

        if isinstance(context.grid, Rectilinear | Curvilinear):
            # Use the actual coordinate names from the grid system
            grid = cast(Rectilinear | Curvilinear, context.grid)
            mesh = cvs.quadmesh(data, x=grid.X, y=grid.Y)
        else:
            raise NotImplementedError(
                f"Grid type {type(context.grid)} not supported by DatashaderRasterRenderer"
            )

        if context.datatype is DataType.CONTINUOUS:
            shaded = tf.shade(
                mesh,
                cmap=mpl.colormaps.get_cmap(cmap),
                how="linear",
                span=colorscalerange,
            )
        else:
            raise NotImplementedError("Categorical data not supported yet")

        im = shaded.to_pil()
        im.save(buffer, format=str(format))
