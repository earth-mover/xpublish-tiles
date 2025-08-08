from typing import TYPE_CHECKING

from xpublish_tiles.render import Renderer, register_renderer

if TYPE_CHECKING:
    from xpublish_tiles.types import RenderContext


@register_renderer
class QuiverRenderer(Renderer):
    def validate(self, contexts: dict[str, "RenderContext"]) -> None:
        assert len(contexts) in [2, 3]
        # assert we can find u,v

    def render(self, *, data: dict[str, "RenderContext"]) -> None:
        # look at CF metadata to find u, v
        pass

    @staticmethod
    def style_id() -> str:
        return "quiver"

    @staticmethod
    def supported_variants() -> list[str] | None:
        return ["arrows"]

    @staticmethod
    def supported_colormaps() -> list[str]:
        return []

    @staticmethod
    def default_palette() -> str:
        return "arrows"

    @classmethod
    def describe_style(cls, palette: str) -> dict[str, str]:
        return {
            "id": f"{cls.style_id()}/{palette}",
            "title": f"Quiver - {palette.title()}",
            "description": f"Vector field rendering using {palette} style",
        }
