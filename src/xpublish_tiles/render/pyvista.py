#!/usr/bin/env python3
import io
import threading
from typing import TYPE_CHECKING

import matplotlib as mpl

from xpublish_tiles.grids import Curvilinear, RasterAffine, Rectilinear
from xpublish_tiles.render import Renderer, register_renderer
from xpublish_tiles.types import (
    ContinuousData,
    ImageFormat,
    NullRenderContext,
    PopulatedRenderContext,
    RenderContext,
)

LOCK = threading.Lock()


@register_renderer
class PyVistaRasterRenderer(Renderer):
    def render(
        self,
        *,
        contexts: dict[str, "RenderContext"],
        buffer: io.BytesIO,
        width: int,
        height: int,
        cmap: str,
        colorscalerange: tuple[float, float] | None = None,
        format: ImageFormat = ImageFormat.PNG,
    ) -> None:
        import pvxarray  # noqa: F401
        import pyvista as pv

        (context,) = contexts.values()
        if isinstance(context, NullRenderContext):
            raise NotImplementedError("no overlap with requested bbox.")
        if TYPE_CHECKING:
            assert isinstance(context, PopulatedRenderContext)

        bbox, grid = context.bbox, context.grid
        if TYPE_CHECKING:
            assert isinstance(grid, RasterAffine | Rectilinear | Curvilinear)
            assert hasattr(grid, "X") and hasattr(grid, "Y")
        mesh = context.da.pyvista.mesh(x=grid.X, y=grid.Y).point_data_to_cell_data()
        image_grid = pv.ImageData(
            dimensions=(width, height, 1),
            origin=(bbox.west, bbox.south, 0),
            spacing=(
                (bbox.east - bbox.west) / (width - 1),
                (bbox.north - bbox.south) / (height - 1),
                1,
            ),
        )
        sampled = image_grid.sample(mesh)
        sampled.set_active_scalars(context.da.name)

        if isinstance(context.datatype, ContinuousData):
            if colorscalerange is None:
                valid_min = context.datatype.valid_min
                valid_max = context.datatype.valid_max
                if valid_min is not None and valid_max is not None:
                    colorscalerange = (valid_min, valid_max)

        with LOCK:
            plotter = pv.Plotter(
                off_screen=True,
                window_size=(width, height),
                notebook=False,
                lighting="none",
            )
            plotter.add_mesh(
                sampled.ptc().threshold(mesh.GetScalarRange()[0]),
                scalars=context.da.name,
                clim=colorscalerange,
                show_scalar_bar=False,
                render=True,
            )
            plotter.camera_position = "xy"
            plotter.camera.zoom(1.5)  # WTF
            # plotter.camera.zoom("tight")
            # plotter.camera.tight(adjust_render_window=False)
            plotter.screenshot(buffer, transparent_background=True, return_img=True)

    @staticmethod
    def style_id() -> str:
        return "pyvista_raster"

    @staticmethod
    def supported_variants() -> list[str]:
        colormaps = list(mpl.colormaps)
        return [name for name in sorted(colormaps) if not name.endswith("_r")]

    @staticmethod
    def default_variant() -> str:
        return "viridis"

    @classmethod
    def describe_style(cls, variant: str) -> dict[str, str]:
        return {
            "id": f"{cls.style_id()}/{variant}",
            "title": f"Raster - {variant.title()}",
            "description": f"Raster rendering using {variant} colormap & PyVista",
        }
