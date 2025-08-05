import io
import logging
from typing import TYPE_CHECKING, cast

import datashader as dsh  # type: ignore
import datashader.transfer_functions as tf  # type: ignore
import matplotlib as mpl  # type: ignore
import numpy as np

import xarray as xr
from xpublish_tiles.grids import Curvilinear, RasterAffine, Rectilinear
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
    @staticmethod
    def precompile() -> None:
        cvs = dsh.Canvas(
            plot_height=4,
            plot_width=4,
            x_range=(-2, 2),
            y_range=(-2, 2),
        )
        for dtype in (np.int64, np.float32, np.float64):
            array = np.array([[2, 2], [2, 2]], dtype=dtype)
            data = xr.DataArray(
                array,
                dims=("x", "y"),
                coords={
                    "x": np.array([0, 1], dtype=dtype),
                    "y": np.array([0, 1], dtype=dtype),
                },
                name="foo",
            )
            cvs.quadmesh(data, x="x", y="y")

            data = data.assign_coords(
                {
                    "lon": (("x", "y"), np.array([[0, 1], [1, 2]], dtype=dtype)),
                    "lat": (("x", "y"), np.array([[0, 1], [1, 2]], dtype=dtype)),
                }
            )
            cvs.quadmesh(data, x="lon", y="lat")

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
            raise ValueError("no overlap with requested bbox.")
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

        if (
            colorscalerange is None
            and "valid_min" in data.attrs
            and "valid_max" in data.attrs
        ):
            colorscalerange = (data.attrs.get("valid_min"), data.attrs.get("valid_max"))

        if isinstance(context.grid, RasterAffine | Rectilinear | Curvilinear):
            # Use the actual coordinate names from the grid system
            grid = cast(RasterAffine | Rectilinear | Curvilinear, context.grid)
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
