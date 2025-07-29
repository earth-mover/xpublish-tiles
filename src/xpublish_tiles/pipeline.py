import asyncio
import io
from functools import lru_cache, partial
from typing import Any, cast

import pyproj
import pyproj.aoi

import xarray as xr
from xpublish_tiles.grids import Curvilinear, Rectilinear, guess_grid_system
from xpublish_tiles.types import (
    DataType,
    OutputBBox,
    OutputCRS,
    PopulatedRenderContext,
    QueryParams,
    ValidatedArray,
)

# https://pyproj4.github.io/pyproj/stable/advanced_examples.html#caching-pyproj-objects
transformer_from_crs = lru_cache(partial(pyproj.Transformer.from_crs, always_xy=True))


def pad_bbox(
    bbox: pyproj.aoi.BBox, da: xr.DataArray, x_dim: str, y_dim: str
) -> pyproj.aoi.BBox:
    """
    Extend bbox slightly to account for discrete coordinate sampling.
    This prevents transparency gaps at tile edges due to coordinate resolution.
    """
    x_coord = da[x_dim]
    y_coord = da[y_dim]

    # Calculate coordinate spacing (use first difference for consistency across platforms)
    if x_coord.size > 1:
        x_spacing = float(x_coord.values[1] - x_coord.values[0])
    else:
        x_spacing = 0.0

    if y_coord.size > 1:
        y_spacing = float(y_coord.values[1] - y_coord.values[0])
    else:
        y_spacing = 0.0

    # Extend bbox by half a coordinate spacing on each side
    x_pad = abs(x_spacing) * 0.5
    y_pad = abs(y_spacing) * 0.5

    return pyproj.aoi.BBox(
        west=float(bbox.west - x_pad),
        east=float(bbox.east + x_pad),
        south=float(bbox.south - y_pad),
        north=float(bbox.north + y_pad),
    )


async def pipeline(ds, query: QueryParams) -> io.BytesIO:
    validated = apply_query(ds, variables=query.variables, selectors=query.selectors)
    subsets = subset_to_bbox(validated, bbox=query.bbox, crs=query.crs)
    loaded_contexts = await asyncio.gather(
        *(sub.async_load() for sub in subsets.values())
    )
    context_dict = dict(zip(subsets.keys(), loaded_contexts, strict=True))

    buffer = io.BytesIO()
    renderer = query.get_renderer()
    renderer.render(
        contexts=context_dict,
        buffer=buffer,
        width=query.width,
        height=query.height,
        cmap=query.cmap,
        colorscalerange=query.colorscalerange,
        format=query.format,
    )
    buffer.seek(0)
    return buffer


def _infer_datatype(array: xr.DataArray) -> DataType:
    # return DataType.DISCRETE if array.cf.is_flag_variable else DataType.CONTINUOUS
    # FIXME: enable DISCRETE detection soon, for CTrees.
    return DataType.CONTINUOUS


# FIXME: apply a decorator to time this
def apply_query(
    ds: xr.Dataset, *, variables: list[str], selectors: dict[str, Any]
) -> dict[str, ValidatedArray]:
    """
    This method does all automagic detection necessary for the rest of the pipeline to work.
    """
    validated: dict[str, ValidatedArray] = {}
    ds = ds.cf.sel(**selectors)
    for name in variables:
        grid_system = guess_grid_system(ds, name)
        array = ds.cf[name]
        validated[name] = ValidatedArray(
            da=array,
            grid=grid_system,
            datatype=_infer_datatype(array),
        )
    return validated


def subset_to_bbox(
    validated: dict[str, ValidatedArray], *, bbox: OutputBBox, crs: OutputCRS
) -> dict[str, PopulatedRenderContext]:
    # transform desired bbox to input data?
    # transform coordinates to output CRS
    result = {}
    for var_name, array in validated.items():
        grid = array.grid
        if not isinstance(grid, Rectilinear | Curvilinear):
            raise NotImplementedError(f"{grid=!r} not supported yet.")
        # Cast to help type checker understand narrowed type
        grid = cast(Rectilinear | Curvilinear, grid)
        input_to_output = transformer_from_crs(crs_from=grid.crs, crs_to=crs)
        output_to_input = transformer_from_crs(crs_from=crs, crs_to=grid.crs)

        # FIXME: check bounds overlap, return NullRenderContext if applicable

        input_bbox = output_to_input.transform_bounds(
            left=bbox.west, right=bbox.east, top=bbox.north, bottom=bbox.south
        )

        # Create input bbox and extend it to prevent coordinate sampling gaps
        input_bbox_obj = pyproj.aoi.BBox(
            west=input_bbox[0],
            east=input_bbox[2],
            south=input_bbox[1],
            north=input_bbox[3],
        )
        extended_bbox = pad_bbox(input_bbox_obj, array.da, grid.X, grid.Y)

        subset = grid.sel(array.da, bbox=extended_bbox)

        bx, by = xr.broadcast(subset[grid.X], subset[grid.Y])
        newX, newY = input_to_output.transform(bx.data, by.data)
        newda = subset.assign_coords(
            {bx.name: bx.copy(data=newX), by.name: by.copy(data=newY)}
        )
        result[var_name] = PopulatedRenderContext(
            da=newda,
            grid=grid,
            datatype=array.datatype,
            bbox=bbox,
        )
    return result
