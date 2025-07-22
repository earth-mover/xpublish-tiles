import io
import logging
from numbers import Number
from typing import TYPE_CHECKING

import datashader as dsh  # type: ignore
import datashader.transfer_functions as tf  # type: ignore
import matplotlib as mpl  # type: ignore
import numpy as np

import xarray as xr
from xpublish_tiles.pipeline import DataType, GridType, ImageFormat
from xpublish_tiles.render import Renderer

if TYPE_CHECKING:
    from xpublish_tiles.pipeline import RenderContext

logger = logging.getLogger("xpublish-tiles")


class DatashaderRasterRenderer(Renderer):
    def validate(self, context: dict[str, "RenderContext"]):
        assert len(context) == 1

    def maybe_cast_data(self, data) -> xr.DataArray:  # type: ignore[name-defined]
        kind = data.dtype.kind
        itemsize = data.dtype.itemsize

        # numba only supports float32 and float64. Cast everything else
        if kind == "f" and itemsize not in [4, 8]:
            logger.warning(
                "DataArray dtype is %s, which is not a floating point type "
                "of size 32 or 64. This will result in a slow render.",
                data.dtype,
            )
        if itemsize < 4:
            logger.warning(
                "DataArray dtype is 16-bit. This must be converted to 32-bit before rendering."
            )
            data = data.astype(np.float32)
        elif itemsize < 8:
            # FIXME: this makes no sense.
            logger.warning(
                "DataArray dtype is 32-bit. This must be converted to 64-bit before rendering."
            )
            data = data.astype(np.float64)
        else:
            raise ValueError(
                "DataArray dtype is %s, which is not a floating point type "  # noqa: UP031
                "greater than 64-bit. This is not currently supported." % data.dtype
            )

        return data

    def render(
        self,
        *,
        contexts: dict[str, "RenderContext"],
        buffer: io.BytesIO,
        width: int,
        height: int,
        cmap: str,
        colorscalerange: tuple[Number, Number] | None = None,
        format: ImageFormat = ImageFormat.PNG,
    ):
        self.validate(contexts)
        (context,) = contexts.values()
        bbox = context.bbox
        data = self.maybe_cast_data(context.da)
        cvs = dsh.Canvas(
            plot_height=height,
            plot_width=width,
            x_range=(bbox.west, bbox.east),
            y_range=(bbox.south, bbox.north),
        )

        if context.grid in (
            GridType.REGULAR,
            GridType.RECTILINEAR,
            GridType.CURVILINEAR,
        ):
            mesh = cvs.quadmesh(data, x="x", y="y")
        else:
            raise NotImplementedError

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
