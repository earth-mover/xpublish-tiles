import enum
from dataclasses import dataclass
from typing import Any, NewType, Self

import pyproj
import pyproj.aoi

import xarray as xr
from xpublish_tiles.grids import GridSystem

InputCRS = NewType("InputCRS", pyproj.CRS)
OutputCRS = NewType("OutputCRS", pyproj.CRS)
InputBBox = NewType("InputBBox", pyproj.aoi.BBox)
OutputBBox = NewType("OutputBBox", pyproj.aoi.BBox)


class ImageFormat(enum.StrEnum):
    PNG = enum.auto()
    JPEG = enum.auto()


class DataType(enum.Enum):
    DISCRETE = enum.auto()
    CONTINUOUS = enum.auto()


class Style(enum.StrEnum):
    RASTER = enum.auto()
    QUIVER = enum.auto()
    NUMPY = enum.auto()
    VECTOR = enum.auto()


@dataclass
class QueryParams:
    variables: list[str]
    crs: OutputCRS
    bbox: OutputBBox
    # decision: are time and vertical special?
    #    they are not; only selection is allowed
    #    notice that we are effectively interpolating along X, Y
    #    so there is some "interpretation" here
    selectors: dict[str, Any]
    style: Style
    width: int
    height: int
    cmap: str
    format: ImageFormat
    colorscalerange: tuple[float, float] | None = None

    def get_renderer(self):
        from xpublish_tiles.render.raster import DatashaderRasterRenderer

        if self.style is Style.RASTER:
            return DatashaderRasterRenderer()
        else:
            raise NotImplementedError("Unknown style type: %r" % self.style)  # noqa: UP031


@dataclass(kw_only=True)
class ValidatedArray:
    da: xr.DataArray
    datatype: DataType
    grid: GridSystem


@dataclass
class RenderContext:
    pass


@dataclass
class NullRenderContext(RenderContext):
    async def async_load(self) -> Self:
        return type(self)()

    def load(self) -> Self:
        return type(self)()


@dataclass
class PopulatedRenderContext(RenderContext):
    """all information needed to render the output."""

    da: xr.DataArray
    datatype: DataType
    grid: GridSystem
    bbox: OutputBBox

    async def async_load(self) -> Self:
        new_data = await self.da.async_load()
        return type(self)(
            da=new_data, datatype=self.datatype, grid=self.grid, bbox=self.bbox
        )

    def load(self) -> Self:
        new_data = self.da.load()
        return type(self)(
            da=new_data, datatype=self.datatype, grid=self.grid, bbox=self.bbox
        )
