import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NewType, Self

import pyproj
import pyproj.aoi

import xarray as xr
from xpublish_tiles.grids import GridSystem
from xpublish_tiles.utils import async_time_debug, time_debug

InputCRS = NewType("InputCRS", pyproj.CRS)
OutputCRS = NewType("OutputCRS", pyproj.CRS)
InputBBox = NewType("InputBBox", pyproj.aoi.BBox)
OutputBBox = NewType("OutputBBox", pyproj.aoi.BBox)


class ImageFormat(enum.StrEnum):
    PNG = enum.auto()
    JPEG = enum.auto()


@dataclass
class DataType:
    pass


@dataclass
class DiscreteData(DataType):
    values: Sequence[Any]
    meanings: Sequence[str]
    colors: Sequence[str] | None

    def __post_init__(self) -> None:
        assert len(self.values) == len(self.meanings)
        if self.colors is not None:
            assert len(self.colors) == len(self.values), (
                len(self.colors),
                len(self.values),
            )


@dataclass
class ContinuousData(DataType):
    valid_min: Any | None
    valid_max: Any | None

    def __post_init__(self) -> None:
        valid_min, valid_max = self.valid_min, self.valid_max
        if valid_min is not None and valid_max is not None:
            if valid_max < valid_min:
                raise ValueError(f"{valid_max=!r} < {valid_min=!r} specified in attrs.")
        elif valid_min is None and valid_max is None:
            pass
        else:
            raise ValueError(
                f"Either both `valid_max` and `valid_min` must be set or unset. "
                f"Received {valid_max=!r}, {valid_min=!r}."
            )


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
    style: str
    width: int
    height: int
    variant: str
    format: ImageFormat
    colorscalerange: tuple[float, float] | None = None

    def get_renderer(self):
        from xpublish_tiles.render import RenderRegistry

        renderer_cls = RenderRegistry.get(self.style)
        return renderer_cls()


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

    def sync_load(self) -> Self:
        return type(self)()


@dataclass
class PopulatedRenderContext(RenderContext):
    """all information needed to render the output."""

    da: xr.DataArray
    datatype: DataType
    grid: GridSystem
    bbox: OutputBBox

    @async_time_debug
    async def async_load(self) -> Self:
        new_data = await self.da.load_async()
        return type(self)(
            da=new_data, datatype=self.datatype, grid=self.grid, bbox=self.bbox
        )

    @time_debug
    def sync_load(self) -> Self:
        new_data = self.da.load()
        return type(self)(
            da=new_data, datatype=self.datatype, grid=self.grid, bbox=self.bbox
        )
