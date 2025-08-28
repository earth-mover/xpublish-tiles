import asyncio
import copy
import io
import os
from typing import Any, cast

import numpy as np
import pyproj
from pyproj.aoi import BBox

import xarray as xr
from xpublish_tiles.grids import (
    Curvilinear,
    GridSystem,
    RasterAffine,
    Rectilinear,
    guess_grid_system,
)
from xpublish_tiles.lib import (
    EXECUTOR,
    TileTooBigError,
    check_transparent_pixels,
    transform_coordinates,
    transformer_from_crs,
)
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
from xpublish_tiles.utils import async_time_debug, time_debug

# This takes the pipeline ~ 1s
MAX_RENDERABLE_SIZE = 10_000 * 10_000


def round_bbox(bbox: BBox) -> BBox:
    # https://github.com/developmentseed/morecantile/issues/175
    # the precision in morecantile tile bounds isn't perfect,
    # a good way to test is `tms.bounds(Tile(0,0,0))` which should
    # match the spec exactly: https://docs.ogc.org/is/17-083r4/17-083r4.html#toc48
    # Example: tests/test_pipeline.py::test_pipeline_tiles[-90->90,0->360-wgs84_prime_meridian(2/2/1)]
    return BBox(
        west=round(bbox.west, 8),
        south=round(bbox.south, 8),
        east=round(bbox.east, 8),
        north=round(bbox.north, 8),
    )


async def apply_slicers(
    da: xr.DataArray, *, grid: GridSystem, slicers: dict[str, list[slice]]
) -> xr.DataArray:
    grid = cast(RasterAffine | Rectilinear | Curvilinear, grid)

    # Y dimension should always have exactly one slice
    y_slice = slicers[grid.Ydim][0]

    subsets = [
        da.isel({grid.Xdim: lon_slice, grid.Ydim: y_slice})
        for lon_slice in slicers[grid.Xdim]
    ]
    if sum(subset.size for subset in subsets) > MAX_RENDERABLE_SIZE:
        raise TileTooBigError("Tile request too big. Please choose a higher zoom level.")
    if (
        np.sum(np.stack([np.array(subset.shape) for subset in subsets], axis=0), axis=0)
        < 2
    ).any():
        raise ValueError("Tile request resulted in insufficient data for rendering.")

    if int(os.environ.get("XPUBLISH_TILES_ASYNC_LOAD", "1")):
        coros = [subset.load_async() for subset in subsets]
        results = await asyncio.gather(*coros)
    else:
        results = [subset.load() for subset in subsets]
    subset = xr.concat(results, dim=grid.Xdim) if len(results) > 1 else results[0]
    return subset


@time_debug
def has_coordinate_discontinuity(coordinates: np.ndarray, *, axis: int) -> bool:
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
    gaps = np.abs(np.diff(coordinates, axis=axis))

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


@time_debug
def fix_coordinate_discontinuities(
    coordinates: np.ndarray, transformer: pyproj.Transformer, *, axis: int, bbox: BBox
) -> np.ndarray:
    """
    Fix coordinate discontinuities that occur during coordinate transformation.

    When transforming geographic coordinates that cross the antimeridian (±180°)
    to projected coordinates (like Web Mercator), large gaps can appear in the
    transformed coordinate space. This function detects such gaps and applies
    intelligent offset corrections to make coordinates continuous.

    The algorithm:
    1. Uses np.unwrap to fix coordinate discontinuities automatically
    2. Calculates the expected coordinate space width using transformer bounds
    3. Shifts the result to maximize overlap with the bbox

    Examples
    --------
    >>> import numpy as np
    >>> import pyproj
    >>> from pyproj.aoi import BBox
    >>> coords = np.array([350, 355, 360, 0, 5, 10])  # Wrap from 360 to 0
    >>> transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=True)
    >>> bbox = BBox(west=-10, east=20, south=-90, north=90)
    >>> fixed = fix_coordinate_discontinuities(coords, transformer, axis=0, bbox=bbox)
    >>> gaps = np.diff(fixed)
    >>> assert np.all(np.abs(gaps) < 20), f"Large gap remains: {gaps}"
    """
    # Only handle X coordinates (longitude) for now
    if axis != 0:
        return coordinates

    # Calculate coordinate space width using ±180° transform
    # This is unavoidable since AreaOfUse for a CRS is always in lat/lon
    x_bounds, _ = transformer.transform([-180.0, 180.0], [0.0, 0.0])
    coordinate_space_width = abs(x_bounds[1] - x_bounds[0])

    if coordinate_space_width == 0:
        # ETRS89 returns +N for both -180 & 180
        # it's area of use is (-35.58, 24.6, 44.83, 84.73)
        # we ignore such things for now
        return coordinates

    # Step 1: Use np.unwrap to fix discontinuities
    unwrapped_coords = np.unwrap(
        coordinates,
        discont=coordinate_space_width / 2,
        axis=axis,
        period=coordinate_space_width,
    )

    # Step 2: Determine optimal shift based on coordinate and bbox bounds
    coord_min, coord_max = unwrapped_coords.min(), unwrapped_coords.max()
    bbox_center = (bbox.west + bbox.east) / 2
    coord_center = (coord_min + coord_max) / 2

    # Calculate how many coordinate_space_widths we need to shift to align centers
    center_diff = bbox_center - coord_center
    shift_multiple = round(center_diff / coordinate_space_width)

    # Apply the calculated shift
    result = unwrapped_coords + (shift_multiple * coordinate_space_width)
    return result


@time_debug
def bbox_overlap(input_bbox: BBox, grid_bbox: BBox, is_geographic: bool) -> bool:
    """Check if bboxes overlap, handling longitude wrapping for geographic data."""
    # Standard intersection check
    if input_bbox.intersects(grid_bbox):
        return True

    # For geographic data, check longitude wrapping
    if is_geographic:
        # If the bbox spans more than 360 degrees, it covers the entire globe
        if (input_bbox.east - input_bbox.west) >= 359:
            return True

        if (grid_bbox.east - grid_bbox.west) >= 359:
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


@async_time_debug
async def pipeline(ds, query: QueryParams) -> io.BytesIO:
    validated = apply_query(ds, variables=query.variables, selectors=query.selectors)
    subsets = await subset_to_bbox(validated, bbox=query.bbox, crs=query.crs)

    buffer = io.BytesIO()
    renderer = query.get_renderer()

    # Run render in executor to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        EXECUTOR,
        lambda: renderer.render(
            contexts=subsets,
            buffer=buffer,
            width=query.width,
            height=query.height,
            variant=query.variant,
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


@time_debug
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
        if grid.Z in array.dims:
            array = array.sel({grid.Z: 0}, method="nearest")
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


@async_time_debug
async def subset_to_bbox(
    validated: dict[str, ValidatedArray], *, bbox: OutputBBox, crs: OutputCRS
) -> dict[str, PopulatedRenderContext | NullRenderContext]:
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
        west, south, east, north = output_to_input.transform_bounds(
            left=bbox.west, right=bbox.east, top=bbox.north, bottom=bbox.south
        )
        if grid.crs.is_geographic:
            west = west - 360 if west > east else west

        input_bbox = BBox(west=west, south=south, east=east, north=north)
        input_bbox = round_bbox(input_bbox)

        if input_bbox.west > input_bbox.east:
            raise ValueError(f"Invalid Bbox after transformation: {input_bbox!r}")

        # Check bounds overlap, accounting for longitude wrapping in geographic data
        if not bbox_overlap(input_bbox, grid.bbox, grid.crs.is_geographic):
            # No overlap - return NullRenderContext
            result[var_name] = NullRenderContext()
            continue

        slicers = grid.sel(array.da, bbox=input_bbox)
        da = grid.assign_index(array.da)
        subset = await apply_slicers(da, grid=grid, slicers=slicers)
        has_discontinuity = (
            has_coordinate_discontinuity(
                subset[grid.X].data, axis=subset[grid.X].get_axis_num(grid.Xdim)
            )
            if grid.crs.is_geographic
            else False
        )
        newX, newY = await transform_coordinates(subset, grid.X, grid.Y, input_to_output)

        # Fix coordinate discontinuities in transformed coordinates if detected
        if has_discontinuity:
            fixed = fix_coordinate_discontinuities(
                newX.data,
                input_to_output,
                axis=subset[grid.X].get_axis_num(grid.Xdim),
                bbox=bbox,
            )
            newX = newX.copy(data=fixed)

        newda = subset.assign_coords({grid.X: newX, grid.Y: newY})
        result[var_name] = PopulatedRenderContext(
            da=newda,
            grid=grid,
            datatype=array.datatype,
            bbox=bbox,
        )
    return result
