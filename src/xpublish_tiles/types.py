import enum
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NewType, Self

import numba
import numpy as np
import pyproj
import pyproj.aoi

import xarray as xr
from xpublish_tiles.config import config
from xpublish_tiles.grids import GridSystem, GridSystem2D, Rectilinear, UgridIndexer
from xpublish_tiles.logger import get_context_logger

InputCRS = NewType("InputCRS", pyproj.CRS)
OutputCRS = NewType("OutputCRS", pyproj.CRS)
InputBBox = NewType("InputBBox", pyproj.aoi.BBox)
OutputBBox = NewType("OutputBBox", pyproj.aoi.BBox)


class ImageFormat(enum.StrEnum):
    PNG = enum.auto()
    JPEG = enum.auto()


class SelectionMethod(str, enum.Enum):
    """Selection methods for dimension indexing.

    These map to xarray's .sel() method parameter.
    EXACT maps to method=None for exact matching.
    """

    EXACT = "exact"
    NEAREST = "nearest"
    PAD = "pad"
    FFILL = "ffill"
    BACKFILL = "backfill"
    BFILL = "bfill"

    @property
    def xarray_method(self) -> str | None:
        """Return the method argument to pass to xarray's .sel().

        Returns None for EXACT (exact matching).
        """
        if self == SelectionMethod.EXACT:
            return None
        return self.value


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
    colormap: dict[str, str] | None = None

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
class RenderContext(ABC):
    @abstractmethod
    async def maybe_rewrite_to_rectilinear(self, *, width: int, height: int) -> Self:
        pass


@dataclass
class NullRenderContext(RenderContext):
    async def maybe_rewrite_to_rectilinear(
        self, *, width: int, height: int, logger=None
    ) -> Self:
        return self


@dataclass
class PopulatedRenderContext(RenderContext):
    """all information needed to render the output."""

    da: xr.DataArray
    datatype: DataType
    grid: GridSystem
    bbox: OutputBBox
    ugrid_indexer: UgridIndexer | None = None

    async def maybe_rewrite_to_rectilinear(
        self, *, width: int, height: int, logger=None
    ) -> Self:
        data = self.da
        grid = self.grid
        bbox = self.bbox

        # Check if approximate rectilinear detection is enabled
        if not config.get("detect_approx_rectilinear"):
            return self

        if data.size < config.get("rectilinear_check_min_size") ** 2:
            # too small to really matter
            return self

        if not isinstance(grid, GridSystem2D):
            return self

        if data[grid.X].ndim == 1 and data[grid.Y].ndim == 1:
            return self

        xcheck, xmin, xmax = check_rectilinear(
            data[grid.X].data,
            origin=bbox.west,
            span=bbox.east - bbox.west,
            canvas_size=width,
            axis=data[grid.X].get_axis_num(grid.Ydim),
            threshold=max(config.get("default_pad") - 1, 1),
        )
        # logger.debug(f"===> max x pix difference: {xpix!r}")
        if not xcheck:
            return self

        ycheck, ymin, ymax = check_rectilinear(
            data[grid.Y].data,
            origin=bbox.west,
            span=bbox.east - bbox.west,
            canvas_size=width,
            axis=data[grid.Y].get_axis_num(grid.Xdim),
            threshold=max(config.get("default_pad") - 1, 1),
        )
        # logger.debug(f"===> max y pix difference: {ypix!r}")
        if not ycheck:
            return self

        newx = data[grid.X].isel({grid.Ydim: 0}).data
        newy = data[grid.Y].isel({grid.Xdim: 0}).data
        newx[[0, -1]] = [xmin, xmax] if newx[0] < newx[-1] else [xmax, xmin]
        newy[[0, -1]] = [ymin, ymax] if newy[0] < newy[-1] else [ymax, ymin]
        data = data.assign_coords(
            {grid.Xdim: (grid.Xdim, newx), grid.Ydim: (grid.Ydim, newy)}
        )
        grid = Rectilinear(
            crs=grid.crs,
            bbox=grid.bbox,
            X=grid.Xdim,
            Y=grid.Ydim,
            indexes=(),  # type: ignore[arg-type]
            Z=None,
        )
        logger = logger or get_context_logger()
        logger.debug("✏️ rewriting to rectilinear")
        return type(self)(da=data, datatype=self.datatype, grid=self.grid, bbox=self.bbox)


# def check_rectilinear(
#     array: np.ndarray,
#     *,
#     origin: float,
#     canvas_size: int,
#     span: float,
#     threshold: int,
#     axis: int,
# ) -> bool:
#     pix = array - origin
#     pix *= canvas_size / span
#     np.trunc(pix, out=pix)
#     selector = [slice(None), slice(None)]
#     selector[axis] = slice(1)
#     pix -= pix[tuple(selector)]
#     np.abs(pix, out=pix)
#     np.less_equal(pix, threshold, out=pix)
#     return pix.all()


@numba.jit(nopython=True, nogil=True)
def check_rectilinear(
    array: np.ndarray,
    *,
    origin: float,
    canvas_size: int,
    span: float,
    threshold: int,
    axis: int,
) -> tuple[bool, float, float]:
    frac = canvas_size / span
    res = True
    # maxpix = -1
    amin = array[0, 0]
    amax = array[0, 0]
    for i in range(array.shape[0]):
        origy = np.trunc((array[i, 0] - origin) * frac)
        for j in range(array.shape[1]):
            origx = np.trunc((array[0, j] - origin) * frac)
            pix = np.trunc((array[i, j] - origin) * frac)
            pix -= origx * (1 - axis) + origy * axis
            # maxpix = max(maxpix, abs(pix))
            res &= abs(pix) <= threshold
            if not res:
                break
            amax = max(amax, array[i, j])
            amin = min(amin, array[i, j])
    return res, amin, amax  # type: ignore[return-value]
