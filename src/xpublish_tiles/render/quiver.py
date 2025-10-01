from typing import TYPE_CHECKING

from xpublish_tiles.render import Renderer, register_renderer

if TYPE_CHECKING:
    from xpublish_tiles.types import RenderContext


@register_renderer
class QuiverRenderer(Renderer):
    def validate(self, contexts: dict[str, "RenderContext"]) -> None:
        assert len(contexts) in [2, 3]
        # assert we can find u,v

    def render(
        self,
        *,
        contexts: dict[str, "RenderContext"],
        buffer,
        width: int,
        height: int,
        variant: str,
        colorscalerange=None,
        format=None,
        context_logger=None,
        colormap: dict[str, str] | None = None,
    ) -> None:
        # Handle "default" alias
        if variant == "default":
            variant = self.default_variant()

        # look at CF metadata to find u, v
        pass

    @staticmethod
    def style_id() -> str:
        return "quiver"

    @staticmethod
    def supported_variants() -> list[str]:
        return ["arrows"]

    @staticmethod
    def default_variant() -> str:
        return "arrows"

    @classmethod
    def describe_style(cls, variant: str) -> dict[str, str]:
        return {
            "id": f"{cls.style_id()}/{variant}",
            "title": f"Quiver - {variant.title()}",
            "description": f"Vector field rendering using {variant} style",
        }
