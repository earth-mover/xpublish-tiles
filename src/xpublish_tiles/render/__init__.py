import io
from numbers import Number
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xpublish_tiles.pipeline import RenderContext


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
