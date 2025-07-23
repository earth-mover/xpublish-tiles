import asyncio
import enum
import io
from dataclasses import dataclass
from numbers import Number
from typing import Any, NewType, Self

import numpy as np
import pyproj
import pyproj.aoi

import xarray as xr
from xpublish_tiles.render import Renderer
from xpublish_tiles.render.raster import DatashaderRasterRenderer

InputCRS = NewType("InputCRS", pyproj.CRS)
OutputCRS = NewType("OutputCRS", pyproj.CRS)
InputBBox = NewType("InputBBox", pyproj.aoi.BBox)
OutputBBox = NewType("OutputBBox", pyproj.aoi.BBox)


class ImageFormat(enum.StrEnum):
    PNG = enum.auto()
    JPEG = enum.auto()


class GridType(enum.Enum):
    REGULAR = enum.auto()
    RECTILINEAR = enum.auto()
    CURVILINEAR = enum.auto()
    TRIANGULAR = enum.auto()
    POLYGONS = enum.auto()
    DGGS = enum.auto()


class DataType(enum.Enum):
    DISCRETE = enum.auto()
    CONTINUOUS = enum.auto()


class Style(enum.Enum):
    RASTER = enum.auto()
    QUIVER = enum.auto()
    NUMPY = enum.auto()
    VECTOR = enum.auto()


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
    colorscalerange: tuple[Number, Number] | None = None

    def get_renderer(self) -> Renderer:
        if self.style is Style.RASTER:
            return DatashaderRasterRenderer()
        else:
            raise NotImplementedError("Unknown style type: %r" % self.style)  # noqa: UP031


@dataclass
class ValidatedArray:
    """ """

    da: xr.DataArray
    datatype: DataType
    grid: GridType
    bbox: InputBBox
    crs: InputCRS
    # all vars needed to construct a *horizontal* mesh for the dataarray
    # X, Y are in the input data CRS
    X: np.ndarray
    Y: np.ndarray
    # TODO: for Tri
    nvertex: np.ndarray


@dataclass
class RenderContext:
    """all information needed to render the output"""

    da: xr.DataArray
    datatype: DataType
    grid: GridType
    bbox: OutputBBox
    # all vars needed to construct a *horizontal* mesh for the dataarray
    # X, Y have been transformed to the output CRS
    X: np.ndarray
    Y: np.ndarray
    # TODO: for Tri
    nvertex: np.ndarray | None = None

    async def async_load(self) -> Self:
        new_data = await self.da.async_load()
        return type(self)(
            da=new_data,
            datatype=self.datatype,
            grid=self.grid,
            bbox=self.bbox,
            X=self.X,
            Y=self.Y,
            nvertex=self.nvertex,
        )


async def pipeline(ds, query: QueryParams) -> io.BytesIO:
    validated = apply_query(ds, variables=query.variables, selectors=query.selectors)
    subsets = subset_to_bbox(validated, bbox=query.bbox, crs=query.crs)
    contexts = await asyncio.gather(*(sub.async_load() for sub in subsets.values()))
    context_dict = dict(zip(subsets.keys(), contexts, strict=True))

    buffer = io.BytesIO()
    renderer = query.get_renderer()
    renderer.render(
        contexts=context_dict,
        buffer=buffer,
        width=query.width,
        height=query.height,
        cmap=query.cmap,
        colorscalerange=query.colorscalerange,
    )
    buffer.seek(0)
    return buffer


def apply_query(
    ds: xr.Dataset, *, variables: list[str], selectors: dict[str, Any]
) -> dict[str, ValidatedArray]:
    """
    This method does all automagic detection necessary for the rest of the pipeline to work.
    """
    # For each var:
    #     infer data crs
    #     get X, Y
    #     apply the right index????
    #     infer data type
    #     infer GridType
    #     construct ValidatedArray?
    raise NotImplementedError("apply_query not yet implemented")


def subset_to_bbox(
    validated: dict[str, ValidatedArray], *, bbox: OutputBBox, crs: OutputCRS
) -> dict[str, RenderContext]:
    # transform desired bbox to input data?
    # transform coordinates to output CRS
    raise NotImplementedError("subset_to_bbox not yet implemented")
