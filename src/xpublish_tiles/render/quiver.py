from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xpublish_tiles.pipeline import RenderContext


class QuiverRenderer:
    def validate(self, contexts: dict[str, "RenderContext"]) -> None:
        assert len(contexts) in [2, 3]
        # assert we can find u,v

    def render(self, *, data: dict[str, "RenderContext"]) -> None:
        # look at CF metadata to find u, v
        pass
