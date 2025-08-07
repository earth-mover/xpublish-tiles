import asyncio
import copy
import io
import logging
import math
import os
from functools import lru_cache, partial
from typing import Any, cast

import numpy as np
import pyproj
from pyproj.aoi import BBox

import xarray as xr
from xpublish_tiles.grids import Curvilinear, RasterAffine, Rectilinear, guess_grid_system
from xpublish_tiles.lib import EXECUTOR, check_transparent_pixels, transform_blocked
from xpublish_tiles.types import (
    ContinuousData,
    DataType,
    DiscreteData,
    NullRenderContext,
    OutputBBox,
    OutputCRS,
    PopulatedRenderContext,
    QueryParams,
    ValidatedArray,
)

logger = logging.getLogger("xpublish-tiles")

# https://pyproj4.github.io/pyproj/stable/advanced_examples.html#caching-pyproj-objects
transformer_from_crs = lru_cache(partial(pyproj.Transformer.from_crs, always_xy=True))

# benchmarked with
# import numpy as np
# import pyproj
# from src.xpublish_tiles.lib import transform_blocked

# x = np.linspace(2635840.0, 3874240.0, 500)
# y = np.linspace(5415940.0, 2042740, 500)

# transformer = pyproj.Transformer.from_crs(3035, 4326, always_xy=True)
# grid = np.meshgrid(x, y)

# %timeit transform_blocked(*grid, chunk_size=(20, 20), transformer=transformer)
# %timeit transform_blocked(*grid, chunk_size=(100, 100), transformer=transformer)
# %timeit transform_blocked(*grid, chunk_size=(250, 250), transformer=transformer)
# %timeit transform_blocked(*grid, chunk_size=(500, 500), transformer=transformer)
# %timeit transformer.transform(*grid)
#
# 500 x 500 grid:
# 19.1 ms ± 1.64 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 10.9 ms ± 113 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# 13.8 ms ± 222 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# 48.6 ms ± 318 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 49.6 ms ± 3.38 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
#
# 2000 x 2000 grid:
# 302 ms ± 21.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 156 ms ± 1.36 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 155 ms ± 2.75 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 156 ms ± 5.07 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 772 ms ± 27 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
CHUNKED_TRANSFORM_CHUNK_SIZE = (250, 250)


def has_coordinate_discontinuity(coordinates: np.ndarray) -> bool:
    """
    Detect coordinate discontinuities in geographic longitude coordinates.

    This function analyzes longitude coordinates to detect antimeridian crossings
    that will cause discontinuities when transformed to projected coordinate systems.

    Parameters
    ----------
    coordinates : np.ndarray
        Geographic longitude coordinates to analyze

    Returns
    -------
    bool
        True if a coordinate discontinuity is detected, False otherwise

    Notes
    -----
    The function detects antimeridian crossings in different coordinate conventions:
    - For -180→180 system: Looks for gaps > 180°
    - For 0→360 system: Looks for data crossing the 180° longitude line

    Examples of discontinuity cases:
    - [-179°, -178°, ..., 178°, 179°] → Large gap when wrapped
    - [350°, 351°, ..., 10°, 11°] → Crosses 0°/360° boundary
    - [180°, 181°, ..., 190°] → Crosses antimeridian in 0→360 system
    """
    if len(coordinates) == 0:
        return False

    x_min, x_max = coordinates.min(), coordinates.max()
    x_sorted = np.sort(coordinates)
    gaps = np.diff(x_sorted)

    if len(gaps) == 0:
        return False

    max_gap = gaps.max()

    # Detect antimeridian crossing in different coordinate systems:
    # 1. For -180→180: look for gaps > 180°
    # 2. For 0→360: look for data crossing 180° longitude (antimeridian)
    if max_gap > 180.0:
        return True
    elif x_min <= 180.0 <= x_max:  # Data crosses the antimeridian (180°/-180°)
        return True

    return False


def fix_coordinate_discontinuities(
    coordinates: np.ndarray, transformer: pyproj.Transformer
) -> np.ndarray:
    """
    Fix coordinate discontinuities that occur during coordinate transformation.

    When transforming geographic coordinates that cross the antimeridian (±180°)
    to projected coordinates (like Web Mercator), large gaps can appear in the
    transformed coordinate space. This function detects such gaps and applies
    intelligent offset corrections to make coordinates continuous.

    The algorithm:
    1. Finds the largest gap in sorted coordinates
    2. Calculates the expected coordinate space width using transformer bounds
    3. If the gap is >30% of coordinate space width, applies an offset
    4. Chooses which side to offset based on which has fewer coordinates
    """
    coords_sorted = np.sort(coordinates.flat)
    gaps = np.diff(coords_sorted)

    if len(gaps) == 0:
        return coordinates

    max_gap = gaps.max()

    # Calculate coordinate space width using ±180° transform
    # This is unavoidable since AreaOfUse for a CRS is always in lat/lon
    x_bounds, _ = transformer.transform([-180.0, 180.0], [0.0, 0.0])
    coordinate_space_width = abs(x_bounds[1] - x_bounds[0])

    # Apply fix if gap is significant (>30% of coordinate space width)
    if max_gap > coordinate_space_width * 0.3:
        gap_idx = np.argmax(gaps)
        split_value = coords_sorted[gap_idx]
        low_side_mask = coordinates <= split_value
        low_count = np.sum(low_side_mask)
        if low_count < (coordinates.size / 2):
            # More coordinates on high side, shift low side up
            coordinates = np.where(
                low_side_mask, coordinates + coordinate_space_width, coordinates
            )
        else:
            # More coordinates on low side, shift high side down
            coordinates = np.where(
                ~low_side_mask, coordinates - coordinate_space_width, coordinates
            )

    return coordinates


def check_bbox_overlap(input_bbox: BBox, grid_bbox: BBox, is_geographic: bool) -> bool:
    """Check if bboxes overlap, handling longitude wrapping for geographic data."""
    # Standard intersection check
    if input_bbox.intersects(grid_bbox):
        return True

    # For geographic data, check longitude wrapping
    if is_geographic:
        # If the bbox spans more than 360 degrees, it covers the entire globe
        if (input_bbox.east - input_bbox.west) >= 360:
            return True

        # Convert input bbox to -180 to 180 range
        normalized_west = ((input_bbox.west + 180) % 360) - 180
        normalized_east = ((input_bbox.east + 180) % 360) - 180

        # Handle the case where normalization creates an anti-meridian crossing
        if normalized_west > normalized_east:
            # Check both parts: [normalized_west, 180] and [-180, normalized_east]
            bbox1 = BBox(
                west=normalized_west,
                south=input_bbox.south,
                east=180.0,
                north=input_bbox.north,
            )
            bbox2 = BBox(
                west=-180.0,
                south=input_bbox.south,
                east=normalized_east,
                north=input_bbox.north,
            )
            if bbox1.intersects(grid_bbox) or bbox2.intersects(grid_bbox):
                return True
        else:
            # Normal case - single normalized bbox
            normalized_input = BBox(
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
            bbox1 = BBox(
                west=wrapped_west_360,
                south=input_bbox.south,
                east=360.0,
                north=input_bbox.north,
            )
            bbox2 = BBox(
                west=0.0,
                south=input_bbox.south,
                east=wrapped_east_360,
                north=input_bbox.north,
            )
            if bbox1.intersects(grid_bbox) or bbox2.intersects(grid_bbox):
                return True
        else:
            # Normal case - single wrapped bbox
            wrapped_input = BBox(
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
    subsets = await subset_to_bbox(validated, bbox=query.bbox, crs=query.crs)
    if int(os.environ.get("XPUBLISH_TILES_ASYNC_LOAD", "1")):
        logger.debug("Using async_load for data loading")
        loaded_contexts = await asyncio.gather(
            *(sub.async_load() for sub in subsets.values())
        )
    else:
        logger.debug("Using synchronous load for data loading")
        loaded_contexts = tuple(sub.load() for sub in subsets.values())
    context_dict = dict(zip(subsets.keys(), loaded_contexts, strict=True))

    buffer = io.BytesIO()
    renderer = query.get_renderer()

    # Run render in executor to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        EXECUTOR,
        lambda: renderer.render(
            contexts=context_dict,
            buffer=buffer,
            width=query.width,
            height=query.height,
            cmap=query.cmap,
            colorscalerange=query.colorscalerange,
            format=query.format,
        ),
    )
    buffer.seek(0)
    if int(os.environ.get("XPUBLISH_TILES_DEBUG_CHECKS", "0")):
        assert check_transparent_pixels(copy.deepcopy(buffer).read()) == 0, query
    return buffer


def _infer_datatype(array: xr.DataArray) -> DataType:
    if (flag_values := array.attrs.get("flag_values")) and (
        flag_meanings := array.attrs.get("flag_meanings")
    ):
        flag_colors = array.attrs.get("flag_colors")

        return DiscreteData(
            values=flag_values,
            meanings=flag_meanings.split(" "),
            colors=flag_colors.split(" ") if isinstance(flag_colors, str) else None,
        )
    return ContinuousData(
        valid_min=array.attrs.get("valid_min"),
        valid_max=array.attrs.get("valid_max"),
    )


def apply_query(
    ds: xr.Dataset, *, variables: list[str], selectors: dict[str, Any]
) -> dict[str, ValidatedArray]:
    """
    This method does all automagic detection necessary for the rest of the pipeline to work.
    """
    validated: dict[str, ValidatedArray] = {}
    if selectors:
        ds = ds.cf.sel(**selectors)
    for name in variables:
        grid = guess_grid_system(ds, name)
        array = ds[name]
        if extra_dims := (set(array.dims) - grid.dims):
            # Note: this will handle squeezing of label-based selection
            # along datetime coordinates
            array = array.isel({dim: -1 for dim in extra_dims})
        validated[name] = ValidatedArray(
            da=array,
            grid=grid,
            datatype=_infer_datatype(array),
        )
    return validated


async def subset_to_bbox(
    validated: dict[str, ValidatedArray], *, bbox: OutputBBox, crs: OutputCRS
) -> dict[str, PopulatedRenderContext]:
    # transform desired bbox to input data?
    # transform coordinates to output CRS
    result = {}
    for var_name, array in validated.items():
        grid = array.grid
        if (ndim := array.da.ndim) > 2:
            raise ValueError(f"Attempting to visualize array with {ndim=!r} > 2.")
        # Check for insufficient data - either dimension has too few points
        if min(array.da.shape) < 2:
            raise ValueError(f"Data too small for rendering: {array.da.sizes!r}.")

        if not isinstance(grid, RasterAffine | Rectilinear | Curvilinear):
            raise NotImplementedError(f"{grid=!r} not supported yet.")
        # Cast to help type checker understand narrowed type
        grid = cast(RasterAffine | Rectilinear | Curvilinear, grid)
        input_to_output = transformer_from_crs(crs_from=grid.crs, crs_to=crs)
        output_to_input = transformer_from_crs(crs_from=crs, crs_to=grid.crs)

        # Check bounds overlap, return NullRenderContext if no overlap
        input_bbox_tuple = output_to_input.transform_bounds(
            left=bbox.west, right=bbox.east, top=bbox.north, bottom=bbox.south
        )
        input_bbox = BBox(
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

        has_discontinuity = (
            has_coordinate_discontinuity(subset[grid.X].data)
            if grid.crs.is_geographic
            else False
        )
        bx, by = xr.broadcast(subset[grid.X], subset[grid.Y])
        if bx.size > math.prod(CHUNKED_TRANSFORM_CHUNK_SIZE):
            loop = asyncio.get_event_loop()
            newX, newY = await loop.run_in_executor(
                EXECUTOR,
                transform_blocked,
                bx.data,
                by.data,
                input_to_output,
                CHUNKED_TRANSFORM_CHUNK_SIZE,
            )
        else:
            newX, newY = input_to_output.transform(bx.data, by.data)

        # Fix coordinate discontinuities in transformed coordinates if detected
        # this is important because the transformation may introduce discontinuities
        # at the anti-meridian
        if has_discontinuity:
            newX = fix_coordinate_discontinuities(newX, input_to_output)

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
