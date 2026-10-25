import io
from numbers import Number

import datashader as dsh
import numpy as np
import spatialpandas
from PIL import Image

import xarray as xr
from xpublish_tiles.logger import get_context_logger, log_duration
from xpublish_tiles.render import DatashaderRenderer, register_renderer
from xpublish_tiles.render.raster import (
    _apply_out_of_range_colors,
    _range_color_to_rgba,
)
from xpublish_tiles.types import (
    ContinuousData,
    ImageFormat,
    RenderContext,
    RGBData,
)


def _dilate(mask: np.ndarray, width: int) -> np.ndarray:
    """Grow 1px lines to ``width`` px with a square kernel; even widths lean to +x/+y."""
    out = mask.copy()
    h, w = mask.shape
    span = range(-((width - 1) // 2), width // 2 + 1)
    for dy in span:
        for dx in span:
            out[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] |= mask[
                max(-dy, 0) : h + min(-dy, 0), max(-dx, 0) : w + min(-dx, 0)
            ]
    return out


def _render_edges(
    cvs: dsh.Canvas,
    geometry,
    data: xr.DataArray,
    *,
    edgecolor: str,
    edgewidth: int,
    band_dim: str | None,
) -> Image.Image | None:
    """RGBA image of the boundaries of cells with valid data, on the fill's canvas."""
    rgba = _range_color_to_rgba(edgecolor)
    if rgba[3] == 0:
        return None
    valid = np.isfinite(data.values)
    if band_dim is not None:
        valid = valid.all(axis=data.get_axis_num(band_dim))
    if not valid.any():
        return None
    agg = cvs.line(
        spatialpandas.GeoDataFrame({"geometry": geometry[valid].boundary}),
        geometry="geometry",
        agg=dsh.any(),
    )
    mask = agg.values
    if edgewidth > 1:
        mask = _dilate(mask, edgewidth)
    yvals = agg.coords[agg.dims[0]].values
    if yvals.size >= 2 and yvals[0] < yvals[-1]:
        # Aggregates are y-up; images are top-down.
        mask = mask[::-1]
    out = np.zeros((*mask.shape, 4), dtype=np.uint8)
    out[mask] = rgba
    return Image.fromarray(out, mode="RGBA")


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
        context, cvs, variant = prepared
        logger = context_logger if context_logger is not None else get_context_logger()

        if context.polygons is None:
            raise ValueError(
                "Polygons not found. Ensure transform_for_render was called with polygons style."
            )

        if len(context.polygons) == 0:
            logger.debug("No data")
            im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            im.save(buffer, format=str(format))
            return

        data = context.da

        with log_duration(f"render (polygons) {data.shape}", logger, key="render"):
            geometry = context.polygons
            try:
                if isinstance(context.datatype, RGBData):
                    # One pass over the geometry: `summary` aggregates all bands together.
                    band_dim = context.datatype.band_dim
                    bands = [f"band{k}" for k in range(data.sizes[band_dim])]
                    gdf = spatialpandas.GeoDataFrame(
                        {
                            "geometry": geometry,
                            **dict(
                                zip(
                                    bands,
                                    data.transpose(band_dim, ...).values,
                                    strict=True,
                                )
                            ),
                        }
                    )
                    agg = cvs.polygons(
                        gdf,
                        "geometry",
                        agg=dsh.summary(**{b: dsh.mean(b) for b in bands}),
                    )
                    mesh = agg[bands].to_dataarray(dim=band_dim)
                else:
                    gdf = spatialpandas.GeoDataFrame(
                        {"geometry": geometry, "data": data.values}
                    )
                    mesh = cvs.polygons(gdf, "geometry", agg=dsh.mean("data"))
            except ValueError as e:
                if "Geometry type combination is not supported" not in str(e):
                    raise
                logger.debug("No data (polygons don't overlap tile bbox)")
                im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                im.save(buffer, format=str(format))
                return

        if isinstance(context.datatype, RGBData):
            im = self.shade_rgb(
                mesh,
                context.datatype,
                colorscalerange=colorscalerange,
            )
        else:
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
        if edgecolor is not None:
            with log_duration("render (polygon edges)", logger, key="render_edges"):
                edges = _render_edges(
                    cvs,
                    geometry,
                    data,
                    edgecolor=edgecolor,
                    edgewidth=edgewidth,
                    band_dim=context.datatype.band_dim
                    if isinstance(context.datatype, RGBData)
                    else None,
                )
            if edges is not None:
                im = Image.alpha_composite(im.convert("RGBA"), edges)
        with log_duration(f"encode {format}", logger, key="encode"):
            im.save(buffer, format=str(format))

    @staticmethod
    def style_id() -> str:
        return "polygons"
