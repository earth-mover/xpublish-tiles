import io
from numbers import Number

import datashader as dsh
import datashader.transfer_functions as tf
import geopandas as gpd
import matplotlib as mpl
import numbagg
import numpy as np
from PIL import Image

from xpublish_tiles.grids import Triangular
from xpublish_tiles.lib import (
    MissingParameterError,
    apply_range_colors,
    create_colormap_from_dict,
)
from xpublish_tiles.logger import get_context_logger, log_duration
from xpublish_tiles.render import DatashaderRenderer, register_renderer
from xpublish_tiles.types import (
    ContinuousData,
    ImageFormat,
    NullRenderContext,
    PopulatedRenderContext,
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
        logger = context_logger if context_logger is not None else get_context_logger()

        if variant == "default":
            variant = self.default_variant()

        assert len(contexts) == 1
        (context,) = contexts.values()

        if isinstance(context, NullRenderContext):
            logger.debug("☐ No data")
            im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            im.save(buffer, format=str(format))
            return

        assert isinstance(context, PopulatedRenderContext)

        if context.cell_boundaries is None:
            raise ValueError(
                "Cell boundaries not found. Ensure transform_for_render was called with polygons style."
            )

        if len(context.cell_boundaries) == 0:
            logger.debug("☐ No data")
            im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            im.save(buffer, format=str(format))
            return

        bbox = context.bbox
        cvs = dsh.Canvas(
            plot_height=height,
            plot_width=width,
            x_range=(bbox.west, bbox.east),
            y_range=(bbox.south, bbox.north),
        )

        data = context.da

        with log_duration(f"render (polygons) {data.shape}", "⬡", logger):
            gdf = gpd.GeoDataFrame(geometry=context.cell_boundaries)
            if isinstance(context.grid, Triangular) and context.ugrid_indexer is not None:
                # Triangular grids: data is per-vertex, polygons are per-face.
                # Average the 3 vertex values to get a face-center value.
                face_vals = data.values[context.ugrid_indexer.connectivity]
                gdf["data"] = numbagg.nanmean(face_vals, axis=1)
            else:
                gdf["data"] = data.values

            try:
                mesh = cvs.polygons(gdf, "geometry", agg=dsh.mean("data"))
            except ValueError as e:
                if "Geometry type combination is not supported" not in str(e):
                    raise
                logger.debug("☐ No data (polygons don't overlap tile bbox)")
                im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                im.save(buffer, format=str(format))
                return

        if isinstance(context.datatype, ContinuousData):
            if colorscalerange is None:
                valid_min = context.datatype.valid_min
                valid_max = context.datatype.valid_max
                if valid_min is not None and valid_max is not None:
                    colorscalerange = (valid_min, valid_max)
                else:
                    raise MissingParameterError(
                        "`colorscalerange` must be specified when array does not have valid_min and valid_max attributes."
                    )

            if colormap is not None:
                cmap = create_colormap_from_dict(colormap)
            else:
                cmap = mpl.colormaps.get_cmap(variant)

            cmap = apply_range_colors(cmap, abovemaxcolor, belowmincolor)

            with np.errstate(invalid="ignore"):
                shaded = tf.shade(
                    mesh,
                    cmap=cmap,
                    how="linear",
                    span=colorscalerange,
                )
            im = shaded.to_pil()
        else:
            raise NotImplementedError(
                f"PolygonsRenderer only supports ContinuousData, got {type(context.datatype)}"
            )

        im.save(buffer, format=str(format))

    @staticmethod
    def style_id() -> str:
        return "polygons"
