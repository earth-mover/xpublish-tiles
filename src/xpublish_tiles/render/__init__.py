import io
from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from numbers import Number
from typing import TYPE_CHECKING, Literal

import datashader as dsh
import datashader.transfer_functions as tf
import matplotlib as mpl
import matplotlib.colorbar
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw

import xarray as xr
from xpublish_tiles.lib import (
    MissingParameterError,
    apply_range_colors,
    create_colormap_from_dict,
    create_listed_colormap_from_dict,
)
from xpublish_tiles.logger import get_context_logger
from xpublish_tiles.types import (
    ContinuousData,
    DataType,
    DiscreteData,
    ImageFormat,
    NullRenderContext,
    PopulatedRenderContext,
)

if TYPE_CHECKING:
    from xpublish_tiles.types import RenderContext


def render_error_image(
    message: str, *, width: int, height: int, format: ImageFormat
) -> io.BytesIO:
    """Render an error message as an image tile."""
    buffer = io.BytesIO()
    img = Image.new("RGBA", (width, height), (255, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), message, fill=(255, 255, 255, 255))
    img.save(buffer, format=format)
    buffer.seek(0)
    return buffer


class RenderRegistry:
    """Registry for renderer classes."""

    _renderers: dict[str, type["Renderer"]] = {}
    _loaded: bool = False

    @classmethod
    def _load_entry_points(cls) -> None:
        """Load renderers from entry points."""
        if cls._loaded:
            return

        eps = entry_points(group="xpublish_tiles.renderers")
        for ep in eps:
            renderer_cls = ep.load()
            cls.register(renderer_cls)

        cls._loaded = True

    @classmethod
    def register(cls, renderer_cls: type["Renderer"]) -> None:
        """Register a renderer class."""
        style_id = renderer_cls.style_id()
        cls._renderers[style_id] = renderer_cls

    @classmethod
    def get(cls, style_id: str) -> type["Renderer"]:
        """Get a renderer class by style ID."""
        cls._load_entry_points()
        if style_id not in cls._renderers:
            raise ValueError(f"Unknown style: {style_id}")
        return cls._renderers[style_id]

    @classmethod
    def all(cls) -> dict[str, type["Renderer"]]:
        """Get all registered renderers."""
        cls._load_entry_points()
        return cls._renderers.copy()


def register_renderer(cls: type["Renderer"]) -> type["Renderer"]:
    """Decorator to register a renderer class."""
    RenderRegistry.register(cls)
    return cls


class Renderer(ABC):
    @abstractmethod
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
        pass

    @abstractmethod
    def render_error(
        self,
        *,
        buffer: io.BytesIO,
        width: int,
        height: int,
        message: str,
        format: ImageFormat = ImageFormat.PNG,
        cmap: str = "",
        colorscalerange: tuple[Number, Number] | None = None,
        **kwargs,
    ):
        """Render an error tile with the given message."""
        pass

    def render_legend(
        self,
        *,
        buffer: io.BytesIO,
        width: int,
        height: int,
        variant: str,
        datatype: DataType,
        colorscalerange: tuple[float, float] | None = None,
        colormap: dict[str, str] | None = None,
        abovemaxcolor: str | None = None,
        belowmincolor: str | None = None,
        vertical: bool = True,
        label: str | None = None,
        background_color: str | None = None,
        text_color: str | None = None,
        format: ImageFormat = ImageFormat.PNG,
    ):
        """Render a legend image describing this style/variant + datatype."""
        raise NotImplementedError(
            f"Legend rendering not supported for style {self.style_id()!r}"
        )

    @staticmethod
    def style_id() -> str:
        """Return the style identifier for this renderer."""
        raise NotImplementedError

    @staticmethod
    def supported_variants() -> list[str]:
        """Return supported variants for this renderer."""
        raise NotImplementedError

    @staticmethod
    def default_variant() -> str:
        """Return the default variant name."""
        raise NotImplementedError

    @classmethod
    def describe_style(cls, variant: str) -> dict[str, str]:
        """Return metadata for a style/variant combination."""
        return {
            "id": f"{cls.style_id()}/{variant}",
            "title": f"{cls.style_id().title()} - {variant.title()}",
            "description": f"{cls.style_id().title()} rendering using {variant}",
        }


class DatashaderRenderer(Renderer):
    """Base class for datashader-based renderers with common colormap handling."""

    def _prepare_render(
        self,
        contexts: dict[str, "RenderContext"],
        *,
        buffer: io.BytesIO,
        width: int,
        height: int,
        variant: str,
        format: ImageFormat = ImageFormat.PNG,
        context_logger=None,
    ) -> tuple | None:
        """Common preamble for datashader renderers.

        Returns ``(context, canvas, variant)`` or ``None`` if an empty tile was written.
        """

        logger = context_logger if context_logger is not None else get_context_logger()

        if variant == "default":
            variant = self.default_variant()

        (context,) = contexts.values()
        if isinstance(context, NullRenderContext):
            logger.debug("☐ No data")
            im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            im.save(buffer, format=str(format))
            return None

        assert isinstance(context, PopulatedRenderContext)
        bbox = context.bbox
        cvs = dsh.Canvas(
            plot_height=height,
            plot_width=width,
            x_range=(bbox.west, bbox.east),
            y_range=(bbox.south, bbox.north),
        )
        return context, cvs, variant

    @staticmethod
    def shade_mesh(
        mesh: "xr.DataArray",
        datatype: DataType,
        *,
        variant: str,
        colorscalerange: tuple[Number, Number] | None = None,
        colormap: dict[str, str] | None = None,
        abovemaxcolor: str | None = None,
        belowmincolor: str | None = None,
    ) -> Image.Image:
        """Shade a datashader mesh (raster aggregation) into a PIL image.

        Shared by raster and polygon renderers — everything after ``cvs.quadmesh`` /
        ``cvs.trimesh`` / ``cvs.polygons`` is identical.
        """
        if isinstance(datatype, ContinuousData):
            if colorscalerange is None:
                valid_min = datatype.valid_min
                valid_max = datatype.valid_max
                if valid_min is not None and valid_max is not None:
                    colorscalerange = (valid_min, valid_max)
                else:
                    raise MissingParameterError(
                        "`colorscalerange` must be specified when array does not have valid_min and valid_max attributes specified."
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
            return shaded.to_pil()

        elif isinstance(datatype, DiscreteData):
            flag_values = datatype.values
            kwargs: dict = {}
            if colormap is not None:
                kwargs["color_key"] = create_listed_colormap_from_dict(
                    colormap, flag_values
                )
            elif datatype.colors is not None:
                kwargs["color_key"] = dict(
                    zip(datatype.values, datatype.colors, strict=True)
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
            return shaded.to_pil()

        else:
            raise NotImplementedError(f"Unsupported datatype: {type(datatype)}")

    def render_error(
        self,
        *,
        buffer: io.BytesIO,
        width: int,
        height: int,
        message: str,
        format: ImageFormat = ImageFormat.PNG,
        cmap: str = "",
        colorscalerange: tuple[Number, Number] | None = None,
        **kwargs,
    ):
        error_buffer = render_error_image(
            message, width=width, height=height, format=format
        )
        buffer.write(error_buffer.getvalue())
        error_buffer.close()

    def render_legend(
        self,
        *,
        buffer: io.BytesIO,
        width: int,
        height: int,
        variant: str,
        datatype: DataType,
        colorscalerange: tuple[float, float] | None = None,
        colormap: dict[str, str] | None = None,
        abovemaxcolor: str | None = None,
        belowmincolor: str | None = None,
        vertical: bool = True,
        label: str | None = None,
        background_color: str | None = None,
        text_color: str | None = None,
        format: ImageFormat = ImageFormat.PNG,
    ):
        if variant == "default":
            variant = self.default_variant()

        orientation = "vertical" if vertical else "horizontal"
        dpi = 100
        # Default to transparent background; JPEG can't carry alpha so fall back
        # to white in that case.
        transparent: tuple[float, float, float, float] = (0, 0, 0, 0)
        facecolor: tuple[float, float, float, float] | str
        if background_color is None:
            facecolor = "white" if format == ImageFormat.JPEG else transparent
        elif background_color == "transparent":
            facecolor = transparent
        else:
            facecolor = background_color
        resolved_text_color: tuple[float, float, float, float] | str | None
        if text_color == "transparent":
            resolved_text_color = transparent
        else:
            resolved_text_color = text_color
        fig = Figure(
            figsize=(width / dpi, height / dpi),
            dpi=dpi,
            layout="constrained",
            facecolor=facecolor,
        )
        cax = fig.add_subplot()

        if isinstance(datatype, ContinuousData):
            if colorscalerange is None:
                if datatype.valid_min is not None and datatype.valid_max is not None:
                    colorscalerange = (datatype.valid_min, datatype.valid_max)
                else:
                    raise MissingParameterError(
                        "`colorscalerange` must be specified when array does not have valid_min and valid_max attributes specified."
                    )

            if colormap is not None:
                cmap = create_colormap_from_dict(colormap)
            else:
                cmap = mpl.colormaps.get_cmap(variant)
            cmap = apply_range_colors(cmap, abovemaxcolor, belowmincolor)

            extend_low = belowmincolor not in (None, "extend")
            extend_high = abovemaxcolor not in (None, "extend")
            if extend_low and extend_high:
                extend: Literal["neither", "min", "max", "both"] = "both"
            elif extend_low:
                extend = "min"
            elif extend_high:
                extend = "max"
            else:
                extend = "neither"

            norm = mcolors.Normalize(vmin=colorscalerange[0], vmax=colorscalerange[1])
            cb = mpl.colorbar.Colorbar(
                cax, cmap=cmap, norm=norm, orientation=orientation, extend=extend
            )
        elif isinstance(datatype, DiscreteData):
            flag_values = list(datatype.values)
            flag_meanings = list(datatype.meanings)
            if colormap is not None:
                color_map = create_listed_colormap_from_dict(colormap, flag_values)
                colors = [color_map[v] for v in flag_values]
            elif datatype.colors is not None:
                colors = list(datatype.colors)
            else:
                minv = min(flag_values)
                maxv = max(flag_values)
                base = mpl.colormaps.get_cmap(variant)
                colors = [
                    mcolors.to_hex(base((v - minv) / maxv if maxv else 0.0))
                    for v in flag_values
                ]

            cmap = mcolors.ListedColormap(colors)
            boundaries = list(range(len(flag_values) + 1))
            norm = mcolors.BoundaryNorm(boundaries, cmap.N)
            cb = mpl.colorbar.Colorbar(
                cax,
                cmap=cmap,
                norm=norm,
                orientation=orientation,
                boundaries=boundaries,
                ticks=[i + 0.5 for i in range(len(flag_values))],
            )
            cb.set_ticklabels(flag_meanings)
        else:
            raise NotImplementedError(f"Unsupported datatype: {type(datatype)}")

        if label:
            cb.set_label(label)

        if resolved_text_color is not None:
            cb.ax.tick_params(colors=resolved_text_color)
            for spine in cb.ax.spines.values():
                spine.set_edgecolor(resolved_text_color)
            if label:
                cb.ax.xaxis.label.set_color(resolved_text_color)
                cb.ax.yaxis.label.set_color(resolved_text_color)

        canvas = FigureCanvasAgg(fig)
        pil_format = "PNG" if format == ImageFormat.PNG else "JPEG"
        with io.BytesIO() as raw:
            canvas.print_png(raw)
            raw.seek(0)
            img = Image.open(raw)
            img.load()
        if pil_format == "JPEG" and img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buffer, format=pil_format)

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
        style_name = cls.style_id().title()
        if variant == "custom":
            return {
                "id": f"{cls.style_id()}/{variant}",
                "title": f"{style_name} - Custom",
                "description": f"{style_name} rendering with a custom colormap provided via the 'colormap' parameter",
            }
        return {
            "id": f"{cls.style_id()}/{variant}",
            "title": f"{style_name} - {variant.title()}",
            "description": f"{style_name} rendering using {variant} colormap",
        }
