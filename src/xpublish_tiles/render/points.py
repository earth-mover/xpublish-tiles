import io
from numbers import Number
from typing import ClassVar

import datashader as dsh
import datashader.transfer_functions as tf
import spatialpandas
from PIL import Image

from xpublish_tiles.logger import get_context_logger, log_duration
from xpublish_tiles.render import DatashaderRenderer, register_renderer
from xpublish_tiles.render.raster import _apply_out_of_range_colors
from xpublish_tiles.types import (
    DEFAULT_MARKERSIZE,
    ContinuousData,
    ImageFormat,
    RenderContext,
)


@register_renderer
class PointsRenderer(DatashaderRenderer):
    """Renderer for point features: a filled circle ``DEFAULT_MARKERSIZE`` pixels across per point.

    Points in one pixel average; where markers overlap, the larger value wins.
    """

    supports_rgb: ClassVar[bool] = False

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
        edgecolor: str | None = None,
        edgewidth: int = 1,
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
        context, _, variant = prepared
        logger = context_logger if context_logger is not None else get_context_logger()

        if context.points is None:
            raise ValueError(
                "Points not found. Ensure transform_for_render was called with points style."
            )

        if len(context.points) == 0:
            logger.debug("No data")
            im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            im.save(buffer, format=str(format))
            return

        data = context.da
        # Draw on a canvas `pad` pixels wider on each side, so markers centred
        # just outside the tile still reach into it.
        pad = DEFAULT_MARKERSIZE // 2
        bbox = context.bbox
        dx = (bbox.east - bbox.west) / width
        dy = (bbox.north - bbox.south) / height
        cvs = dsh.Canvas(
            plot_width=width + 2 * pad,
            plot_height=height + 2 * pad,
            x_range=(bbox.west - pad * dx, bbox.east + pad * dx),
            y_range=(bbox.south - pad * dy, bbox.north + pad * dy),
        )
        with log_duration(f"render (points) {data.shape}", logger, key="render"):
            gdf = spatialpandas.GeoDataFrame(
                {"geometry": context.points, "data": data.values}
            )
            reduction = (
                dsh.mean("data")
                if isinstance(context.datatype, ContinuousData)
                else dsh.max("data")
            )
            agg = cvs.points(gdf, geometry="geometry", agg=reduction)
            agg = tf.spread(agg, px=pad, shape="circle", how="max")
            mesh = agg.isel(x=slice(pad, pad + width), y=slice(pad, pad + height))

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
                im,
                mesh,
                colorscalerange,
                abovemaxcolor,
                belowmincolor,
            )
        with log_duration(f"encode {format}", logger, key="encode"):
            im.save(buffer, format=str(format))

    @staticmethod
    def style_id() -> str:
        return "points"
