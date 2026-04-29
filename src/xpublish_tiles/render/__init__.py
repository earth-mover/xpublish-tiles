import io
from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from numbers import Number
from typing import TYPE_CHECKING

import datashader as dsh
import datashader.transfer_functions as tf
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
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
    DiscreteData,
    ImageFormat,
    NullRenderContext,
    PopulatedRenderContext,
)

if TYPE_CHECKING:
    from xpublish_tiles.types import DataType, RenderContext


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

    @staticmethod
    def geometry_kind() -> str:
        """Geometry pipeline kind: 'raster' or 'polygons'.

        Drives branching in ``transform_for_render`` / ``subset_to_bbox``.
        Vector tiles share the polygon geometry pipeline with the polygons
        renderer; only the final encoder differs.
        """
        return "raster"

    @staticmethod
    def supported_formats() -> set[ImageFormat]:
        """Output formats this renderer can encode."""
        return {ImageFormat.PNG, ImageFormat.JPEG}

    @classmethod
    def media_type(cls, format: ImageFormat) -> str:
        """HTTP Content-Type for the given output format."""
        return {
            ImageFormat.PNG: "image/png",
            ImageFormat.JPEG: "image/jpeg",
            ImageFormat.MVT: "application/vnd.mapbox-vector-tile",
            ImageFormat.GEOJSON: "application/geo+json",
        }[format]

    @classmethod
    def response_headers(cls, format: ImageFormat) -> dict[str, str]:
        """Extra HTTP headers for the response (e.g. Content-Encoding)."""
        return {}

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
        datatype: "DataType",
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
