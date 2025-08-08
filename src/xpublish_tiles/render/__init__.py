import io
from numbers import Number
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xpublish_tiles.types import RenderContext


class RenderRegistry:
    """Registry for renderer classes."""

    _renderers: dict[str, type["Renderer"]] = {}
    _loaded: bool = False

    @classmethod
    def _load_entry_points(cls) -> None:
        """Load renderers from entry points."""
        if cls._loaded:
            return

        from importlib.metadata import entry_points

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


class Renderer:
    def render(
        self,
        *,
        contexts: dict[str, "RenderContext"],
        buffer: io.BytesIO,
        width: int,
        height: int,
        cmap: str,
        colorscalerange: tuple[Number, Number] | None = None,
    ):
        raise NotImplementedError

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
