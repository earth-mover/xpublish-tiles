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
    NullRenderContext,
    OutputBBox,
    OutputCRS,
    PopulatedRenderContext,
    QueryParams,
    ValidatedArray,
)

# https://pyproj4.github.io/pyproj/stable/advanced_examples.html#caching-pyproj-objects
transformer_from_crs = lru_cache(partial(pyproj.Transformer.from_crs, always_xy=True))


def check_bbox_overlap(
    input_bbox: pyproj.aoi.BBox, grid_bbox: pyproj.aoi.BBox, is_geographic: bool
) -> bool:
    """Check if bboxes overlap, handling longitude wrapping for geographic data."""
    # Standard intersection check
    if input_bbox.intersects(grid_bbox):
        return True

    # For geographic data, check longitude wrapping
    if is_geographic:
        # Try converting input bbox to both 0-360 and -180-180 conventions to see if either overlaps

        # Convert input bbox to -180 to 180 range
        normalized_west = ((input_bbox.west + 180) % 360) - 180
        normalized_east = ((input_bbox.east + 180) % 360) - 180

        # Handle the case where normalization creates an anti-meridian crossing
        if normalized_west > normalized_east:
            # Check both parts: [normalized_west, 180] and [-180, normalized_east]
            bbox1 = pyproj.aoi.BBox(
                west=normalized_west,
                south=input_bbox.south,
                east=180.0,
                north=input_bbox.north,
            )
            bbox2 = pyproj.aoi.BBox(
                west=-180.0,
                south=input_bbox.south,
                east=normalized_east,
                north=input_bbox.north,
            )
            if bbox1.intersects(grid_bbox) or bbox2.intersects(grid_bbox):
                return True
        else:
            # Normal case - single normalized bbox
            normalized_input = pyproj.aoi.BBox(
                west=normalized_west,
                south=input_bbox.south,
                east=normalized_east,
                north=input_bbox.north,
            )
            if normalized_input.intersects(grid_bbox):
                return True

        # Also try converting input bbox to 0-360 range
        wrapped_west_360 = input_bbox.west % 360
        wrapped_east_360 = input_bbox.east % 360

        # Handle case where wrapping creates crossing at 0°/360°
        if wrapped_west_360 > wrapped_east_360:
            # Check both parts: [wrapped_west_360, 360] and [0, wrapped_east_360]
            bbox1 = pyproj.aoi.BBox(
                west=wrapped_west_360,
                south=input_bbox.south,
                east=360.0,
                north=input_bbox.north,
            )
            bbox2 = pyproj.aoi.BBox(
                west=0.0,
                south=input_bbox.south,
                east=wrapped_east_360,
                north=input_bbox.north,
            )
            if bbox1.intersects(grid_bbox) or bbox2.intersects(grid_bbox):
                return True
        else:
            # Normal case - single wrapped bbox
            wrapped_input = pyproj.aoi.BBox(
                west=wrapped_west_360,
                south=input_bbox.south,
                east=wrapped_east_360,
                north=input_bbox.north,
            )
            if wrapped_input.intersects(grid_bbox):
                return True

    return False


def pad_bbox(
    bbox: pyproj.aoi.BBox, da: xr.DataArray, x_dim: str, y_dim: str
) -> pyproj.aoi.BBox:
    """
    Extend bbox slightly to account for discrete coordinate sampling.
    This prevents transparency gaps at tile edges due to coordinate resolution.

    The function ensures that the padded bbox does not cross the anti-meridian
    by checking if padding would cause west > east.
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

    # Extend bbox by one coordinate spacing on each side
    # This is needed for high zoom tiles smaller than coordinate spacing
    x_pad = abs(x_spacing)
    y_pad = abs(y_spacing)

    # Calculate padded values
    padded_west = float(bbox.west - x_pad)
    padded_east = float(bbox.east + x_pad)
    padded_south = float(bbox.south - y_pad)
    padded_north = float(bbox.north + y_pad)

    # Check if padding would cause anti-meridian crossing
    # This happens when the padded west > padded east
    if padded_west > padded_east:
        # Don't pad in the x direction to avoid crossing
        padded_west = float(bbox.west)
        padded_east = float(bbox.east)

    return pyproj.aoi.BBox(
        west=padded_west,
        east=padded_east,
        south=padded_south,
        north=padded_north,
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

        # Check bounds overlap, return NullRenderContext if no overlap
        input_bbox_tuple = output_to_input.transform_bounds(
            left=bbox.west, right=bbox.east, top=bbox.north, bottom=bbox.south
        )
        input_bbox = pyproj.aoi.BBox(
            west=input_bbox_tuple[0],
            south=input_bbox_tuple[1],
            east=input_bbox_tuple[2],
            north=input_bbox_tuple[3],
        )

        # Check bounds overlap, accounting for longitude wrapping in geographic data
        has_overlap = check_bbox_overlap(input_bbox, grid.bbox, grid.crs.is_geographic)
        if not has_overlap:
            # No overlap - return NullRenderContext
            result[var_name] = NullRenderContext()
            continue

        # Create extended bbox to prevent coordinate sampling gaps
        extended_bbox = pad_bbox(input_bbox, array.da, grid.X, grid.Y)

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
