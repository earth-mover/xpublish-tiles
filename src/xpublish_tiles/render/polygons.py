import io
from numbers import Number

import datashader as dsh
import spatialpandas
from PIL import Image

from xpublish_tiles.lib import polygons_from_rings
from xpublish_tiles.logger import get_context_logger, log_duration
from xpublish_tiles.render import DatashaderRenderer, register_renderer
from xpublish_tiles.render.raster import _apply_out_of_range_colors
from xpublish_tiles.types import (
    ContinuousData,
    ImageFormat,
    RenderContext,
)


@register_renderer
class PolygonsRenderer(DatashaderRenderer):
    """Renderer for polygon-based grids like HealPix using datashader polygons."""

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

        if context.cell_rings is None:
            raise ValueError(
                "Cell rings not found. Ensure transform_for_render was called with polygons style."
            )

        if len(context.cell_rings) == 0:
            logger.debug("☐ No data")
            im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            im.save(buffer, format=str(format))
            return

        data = context.da

        with log_duration(f"render (polygons) {data.shape}", "⬡", logger):
            values = data.values
            gdf = spatialpandas.GeoDataFrame(
                {"geometry": polygons_from_rings(context.cell_rings), "data": values}
            )

            try:
                mesh = cvs.polygons(gdf, "geometry", agg=dsh.mean("data"))
            except ValueError as e:
                if "Geometry type combination is not supported" not in str(e):
                    raise
                logger.debug("☐ No data (polygons don't overlap tile bbox)")
                im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                im.save(buffer, format=str(format))
                return

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
        return "polygons"
