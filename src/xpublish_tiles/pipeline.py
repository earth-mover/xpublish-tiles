import asyncio
import io
from functools import lru_cache, partial
from typing import Any, cast

import numpy as np
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
        # Check for insufficient data - either dimension has too few points
        if min(array.da.shape) < 2:
            raise ValueError(f"Data too small for rendering: {array.da.sizes!r}.")

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
        # This is a lot easier to do in coordinate space because of anti-meridian handling
        extended_bbox = grid.pad_bbox(input_bbox, array.da)
        subset = grid.sel(array.da, bbox=extended_bbox)

        # Check for insufficient data - either dimension has too few points
        if min(subset.shape) < 2:
            raise ValueError("Tile request resulted in insufficient data for rendering.")

        bx, by = xr.broadcast(subset[grid.X], subset[grid.Y])
        newX, newY = input_to_output.transform(bx.data, by.data)

        # Smart coordinate fixing: detect large gaps and apply precise corrections
        # This fixes coordinate discontinuities that cause transparent pixels in rendered tiles
        if grid.crs.is_geographic:
            newX_flat = newX.flatten()
            newX_sorted = np.sort(newX_flat)
            gaps = np.diff(newX_sorted)

            if len(gaps) > 0:
                max_gap = gaps.max()

                # Calculate coordinate space width using transformer bounds
                x_neg180, _ = input_to_output.transform(-180.0, 0.0)
                x_pos180, _ = input_to_output.transform(180.0, 0.0)
                coordinate_space_width = abs(x_pos180 - x_neg180)

                # Apply fix if gap is significant (>30% of coordinate space width)
                if max_gap > coordinate_space_width * 0.3:
                    gap_idx = np.argmax(gaps)
                    split_value = newX_sorted[gap_idx]

                    # Identify coordinates on each side of the gap
                    low_side_mask = newX <= split_value
                    high_side_mask = newX > split_value

                    # Apply offset to the smaller group to make coordinates continuous
                    low_count = np.sum(low_side_mask)
                    high_count = np.sum(high_side_mask)

                    if low_count < high_count:
                        # More coordinates on high side, shift low side up
                        newX = np.where(
                            low_side_mask, newX + coordinate_space_width, newX
                        )
                    else:
                        # More coordinates on low side, shift high side down
                        newX = np.where(
                            high_side_mask, newX - coordinate_space_width, newX
                        )

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
