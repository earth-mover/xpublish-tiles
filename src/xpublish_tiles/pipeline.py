import asyncio
import io
import math
from functools import partial
from typing import Any, cast

import numpy as np
import pandas as pd
import pyproj
from pyproj.aoi import BBox

import xarray as xr
from xpublish_tiles.config import config
from xpublish_tiles.grids import (
    GridMetadata,
    GridSystem,
    GridSystem2D,
    Triangular,
    UgridIndexer,
    guess_grid_system,
)
from xpublish_tiles.lib import (
    Fill,
    IndexingError,
    MissingParameterError,
    PadDimension,
    TileTooBigError,
    VariableNotFoundError,
    async_run,
    pad_slicers,
    transform_coordinates,
    transformer_from_crs,
)
from xpublish_tiles.logger import get_context_logger, log_duration
from xpublish_tiles.types import (
    ContinuousData,
    DataType,
    DiscreteData,
    NullRenderContext,
    OutputBBox,
    OutputCRS,
    PopulatedRenderContext,
    QueryParams,
    SelectionMethod,
    ValidatedArray,
)
from xpublish_tiles.utils import NUMBA_THREADING_LOCK


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


def sum_tuples(*tuples):
    return tuple(sum(values) for values in zip(*tuples, strict=False))


def check_data_is_renderable_size(
    slicers: dict[str, list[slice | Fill | UgridIndexer]],
    da: xr.DataArray,
    grid: GridSystem,
    alternate: GridMetadata,
) -> bool:
    """
    Check if given slicers produce data of renderable size without loading data.

    This replicates the logic from apply_slicers that checks for tile size limits.
    But is less accurate because we aren't careful about coordinate variables.
    We do *not* want to apply isel here because this function is used in a loop to determine
    minzoom for many TMS-es on the metadata route.

    Parameters
    ----------
    slicers : dict[str, list[slice | Fill | UgridIndexer]]
        Slicers for data selection
    da : xr.DataArray
        Data array (only metadata used, no data loaded)
    grid : GridSystem2D
        Grid system information
    alternate : GridMetadata
        Alternate grid metadata

    Returns
    -------
    bool
        True if data is within renderable size limits, False if too big
    """
    has_alternate = alternate.crs != grid.crs
    # Factor calculation matches apply_slicers logic:
    # if we have crs matching the desired CRS,
    # then we load that data from disk;
    # and double the limit to allow slightly larger tiles
    # = (1 data var + 2 coord vars) * 2
    factor = 3 if has_alternate else 1

    # Calculate total shape using the same logic as apply_slicers
    total_shape = shape_from_slicers(slicers, da, grid)

    # Check if it's within the limit
    return math.prod(total_shape) * da.dtype.itemsize <= factor * config.get(
        "max_renderable_size"
    )


def shape_from_slicers(
    slicers: dict[str, list[slice | Fill | UgridIndexer]],
    da: xr.DataArray,
    grid: GridSystem,
) -> tuple[int, ...]:
    """
    Calculate the total shape from slicers for X and Y dimensions.

    Parameters
    ----------
    slicers : dict[str, list[slice | Fill]]
        Slicers for data selection
    ds : xr.Dataset
        Dataset being processed
    grid : GridSystem
        Grid system information

    Returns
    -------
    tuple[int, int]
        Total shape (width, height) from all slicers
    """

    def get_size(sl, dim_size):
        if isinstance(sl, Fill):
            return sl.size
        elif isinstance(sl, UgridIndexer):
            return sl.vertices.size
        elif isinstance(sl, slice):
            start = sl.start if sl.start is not None else 0
            stop = sl.stop if sl.stop is not None else dim_size
            return stop - start
        else:
            raise TypeError(f"Unknown indexer type: {type(sl)!r}")

    if isinstance(grid, Triangular):
        return (get_size(next(iter(slicers[grid.dim])), None),)

    # Find the one Y slice that's actually a slice (not Fill)
    yslice = None
    for candidate in slicers[grid.Ydim]:
        if isinstance(candidate, slice):
            yslice = candidate
            break

    if yslice is None:
        # If no slice found, take the first item (should be a Fill or slice)
        yslice = slicers[grid.Ydim][0]

    return sum_tuples(
        *(
            (
                get_size(sl, da.sizes[grid.Xdim]),
                get_size(yslice, da.sizes[grid.Ydim]),
            )
            for sl in slicers[grid.Xdim]
        )
    )


def get_coarsen_factors(
    shape: tuple[int, ...],
    max_shape: tuple[int, ...],
    dims: list[str],
    slicers: dict[str, list[slice | Fill | UgridIndexer]],
    da: xr.DataArray,
    grid: GridSystem,
) -> tuple[dict[str, int], dict[str, list[slice | Fill | UgridIndexer]]]:
    """
    Calculate coarsening factors and adjust slicers for data to fit within maximum shape constraints.

    Parameters
    ----------
    shape : tuple[int, int]
        Current data shape (width, height)
    max_shape : tuple[int, int]
        Maximum allowed shape (width, height)
    dims : list[str]
        Dimension names corresponding to shape elements
    slicers : dict[str, list[slice | Fill | UgridIndexer]]
        Original slicers for data selection
    ds : xr.Dataset
        Dataset being processed
    grid : GridSystem
        Grid system information

    Returns
    -------
    tuple[dict[str, int], dict[str, list[slice | Fill | UgridIndexer]]]
        Coarsening factors (>= 2) and adjusted slicers with padding
    """

    if not isinstance(grid, GridSystem2D):
        return {}, slicers

    # After the isinstance check, we know grid is GridSystem2D
    grid = cast(GridSystem2D, grid)

    def largest_even_le(a, b):
        quotient = a // b
        return quotient if quotient % 2 == 0 else quotient - 1

    coarsen_factors = {
        dim: largest_even_le(size, maxsize)
        for size, maxsize, dim in zip(shape, max_shape, dims, strict=True)
    }
    coarsen_factors = {k: v for k, v in coarsen_factors.items() if v >= 2}

    sizes = dict(zip(dims, shape, strict=False))
    padders = []
    for dim, factor in coarsen_factors.items():
        if factor < 2:
            continue
        assert factor % 2 == 0
        size = sizes[dim]
        remainder = factor - (size % factor)
        left_pad = factor // 2
        right_pad = remainder + factor // 2
        padders.append(
            PadDimension(
                name=dim,
                size=da.sizes[dim],
                left_pad=left_pad,
                right_pad=right_pad,
                wraparound=grid.lon_spans_globe and dim == grid.Xdim,
                # for coarsening we always want to increase the size,
                # even if use of xr.pad is necessary
                fill=True,
            )
        )
    new_slicers = pad_slicers(slicers, dimensions=padders)

    return coarsen_factors, new_slicers


def estimate_coarsen_factors_and_slicers(
    da: xr.DataArray,
    *,
    grid: GridSystem2D,
    slicers: dict[str, list[slice | Fill | UgridIndexer]],
    max_shape: tuple[int, int],
    datatype: DataType,
) -> tuple[dict[str, int], dict[str, list[slice | Fill | UgridIndexer]]]:
    """
    Estimate coarsening factors and adjusted slicers for the given data array.

    Parameters
    ----------
    da : xr.DataArray
        Data array to process
    grid : GridSystem2D
        Grid system information
    slicers : dict[str, list[slice | Fill | UgridIndexer]]
        Original slicers for data selection
    max_shape : tuple[int, int]
        Maximum allowed shape (width, height)
    datatype : DataType
        Data type information

    Returns
    -------
    tuple[dict[str, int], dict[str, list[slice | Fill | UgridIndexer]]]
        Coarsening factors and adjusted slicers
    """
    if isinstance(datatype, DiscreteData):
        # TODO: Implement coarsening for categorical data (DiscreteData)
        new_slicers = slicers
        coarsen_factors = {}
    else:
        shape = shape_from_slicers(slicers, da, grid)
        coarsen_factors, new_slicers = get_coarsen_factors(
            shape=shape,
            max_shape=max_shape,
            dims=[grid.Xdim, grid.Ydim],
            slicers=slicers,
            da=da,
            grid=grid,
        )
    return coarsen_factors, new_slicers


async def apply_slicers(
    da: xr.DataArray,
    *,
    grid: GridSystem,
    alternate: GridMetadata,
    slicers: dict[str, list[slice | Fill | UgridIndexer]],
    datatype: DataType,
) -> xr.DataArray:
    logger = get_context_logger()

    has_alternate = alternate.crs != grid.crs
    pick = [alternate.X, alternate.Y]
    ds = (
        da.to_dataset()
        # drop any coordinate vars we don't need
        .reset_coords()[[da.name, *pick]]
    )

    if isinstance(grid, GridSystem2D):
        # Find the one Y slice that's actually a slice (not Fill)
        y_slice = None
        for candidate in slicers[grid.Ydim]:
            if isinstance(candidate, slice):
                y_slice = candidate
                break
        assert y_slice is not None, "No valid Y slice found after padding"

        # Create subsets only for X slices that are actual slices (not Fill)
        subsets = [
            ds.isel({grid.Xdim: x_slice, grid.Ydim: y_slice})
            for x_slice in slicers[grid.Xdim]
            if isinstance(x_slice, slice)
        ]
    elif isinstance(grid, Triangular):
        subsets = [
            ds.isel({grid.Xdim: sl.vertices})
            for sl in slicers[grid.Xdim]
            if isinstance(sl, UgridIndexer)
        ]
    else:
        raise TypeError(f"Unknown grid system type: {type(grid)!r}")

    # if we have crs matching the desired CRS,
    # then we load that data from disk;
    # and double the limit to allow slightly larger tiles
    # = (1 data var + 2 coord vars)
    # this memory estimate should be identical to loading two 1D coordinates
    # & transforming to two 2D coordinates
    factor = 3 if has_alternate else 1
    total_shape = sum_tuples(
        *(
            sum_tuples(*[var.shape for var in subset.data_vars.values()])
            for subset in subsets
        )
    )
    total_size = sum(
        sum([var.size for var in subset.data_vars.values()]) for subset in subsets
    )
    if math.prod(total_shape) * da.dtype.itemsize > factor * config.get(
        "max_renderable_size"
    ):
        msg = (
            f"Tile request too big, requires loading data of total shape: {total_shape!r} "
            f"and total size: {total_size / 1024 / 1024}MB. Please choose a higher zoom level."
        )
        logger = get_context_logger()
        logger.error(
            "Tile request too big",
            total_shape=total_shape,
            max_renderable_size=config.get("max_renderable_size"),
        )
        raise TileTooBigError(msg)

    nvars = sum(len(subset.data_vars) for subset in subsets)
    # if any subset has shape < (2, 2) raise.
    if any(total_size < 2 * nvars for total_size in total_shape):
        logger.error("Tile request resulted in insufficient data for rendering.")
        raise AssertionError("Tile request resulted in insufficient data for rendering.")

    if config.get("async_load"):
        with log_duration("async_load data subsets", "📥"):
            coros = [subset.load_async() for subset in subsets]
            results = await asyncio.gather(*coros)
    else:
        with log_duration("load data subsets", "📥"):
            results = [subset.load() for subset in subsets]
    subset = xr.concat(results, dim=grid.Xdim) if len(results) > 1 else results[0]
    subset_da = subset.set_coords(pick)[da.name]
    return subset_da


def coarsen(
    da: xr.DataArray, coarsen_factors: dict[str, int], *, grid: GridSystem2D
) -> xr.DataArray:
    # TODO: I am skipping padding to center the image here;
    # Note that we specify `boundary="pad"` to avoid missing pixels at the high end of the coordinate.
    # 1. Avoiding explicit padding with DataArray.pad here also means we avoid the complexity of having to
    #    extrapolate out possibly-2D coordinate variables to avoid introducing NaNs in coordinate variables.
    # 2. We could solve this by adding support for `side="both"` upstream in Xarray.
    #    This would simply pad in Variable.coarsen (as currently) and is unaffected by the complexities of NaNs
    #    in indexed coordinate variables.
    # This choice introduces a bias in spatial location of the plot but let's think about this:
    # 1. Global datasets are unaffected. we can always pad in lon as much as we need.
    # 2. the bottom left corner of a regional dataset:
    #    coarsening will push the left boundary to the right, and the bottom boundary up,
    #    introducing a few NaNs in the rendered output; but this doesn't matter visually
    #    if we trigger coarsening only when quite zoomed out
    # 3. the upper right corner of a regional dataset; similarly we may add NaNs at the boundaries
    #    of the viewport, but like in (2) this should not matter as long as we trigger the coarsening
    #    at appropriate zoom levels.
    # if pad_instr := slicers_to_pad_instruction(new_slicers):
    #     subset_da = subset_da.pad(**pad_instr)
    with log_duration(f"coarsen {da.shape} by {coarsen_factors!r}", "🔲"):
        # a further complication: the padding for periodic longitude introduces
        # a discontinuity at the anti-meridian; which we end up averaging over below.
        # So fix that here.
        if grid.lon_spans_globe:
            newX = fix_coordinate_discontinuities(
                da[grid.X].data,
                # FIXME: test 0->360 also!
                transformer_from_crs(grid.crs, grid.crs),
                axis=da[grid.X].get_axis_num(grid.Xdim),
                bbox=grid.bbox,
            )
            da = da.assign_coords({grid.X: da[grid.X].copy(data=newX)})
        with NUMBA_THREADING_LOCK:
            coarsened = da.coarsen(coarsen_factors, boundary="pad").mean()  # type: ignore[unresolved-attribute]
    return coarsened


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
    unwrapped_coords = np.unwrap(coordinates, axis=axis, period=coordinate_space_width)

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


async def pipeline(ds, query: QueryParams) -> io.BytesIO:
    validated = await async_run(
        partial(apply_query, ds, variables=query.variables, selectors=query.selectors)
    )
    pixel_factor = config.get("max_pixel_factor")
    max_shape = (pixel_factor * query.width, pixel_factor * query.height)

    # Capture the context logger before entering thread pool
    context_logger = get_context_logger()

    subsets = await async_run(
        partial(
            subset_to_bbox, validated, bbox=query.bbox, crs=query.crs, max_shape=max_shape
        )
    )

    tasks = [
        async_run(
            lambda s=subset: asyncio.run(
                s.maybe_rewrite_to_rectilinear(
                    width=query.width, height=query.height, logger=context_logger
                )
            )
        )
        for subset in subsets.values()
    ]
    results = await asyncio.gather(*tasks)
    new_subsets = dict(zip(subsets.keys(), results, strict=False))

    buffer = io.BytesIO()
    renderer = query.get_renderer()

    await async_run(
        lambda: renderer.render(
            contexts=new_subsets,
            buffer=buffer,
            width=query.width,
            height=query.height,
            variant=query.variant,
            colorscalerange=query.colorscalerange,
            format=query.format,
            context_logger=context_logger,
            colormap=query.colormap,
        ),
    )
    buffer.seek(0)
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


def parse_selector(value: str) -> tuple[str | None, str]:
    """Parse 'method::value' or 'value' selector syntax.

    Parameters
    ----------
    value : str
        Selector value, optionally prefixed with method using :: separator
        (e.g., 'nearest::2000-01-01T12:00:00')

    Returns
    -------
    tuple[str | None, str]
        (method, value) tuple where method is None for exact matching

    Raises
    ------
    ValueError
        If an invalid method is specified
    """
    if "::" not in value:
        return None, value

    # Split on first :: only
    method_str, actual_value = value.split("::", 1)
    method_str_lower = method_str.lower()

    # Check if the part before :: is a valid method
    try:
        method = SelectionMethod(method_str_lower)
    except ValueError:
        valid_methods = ", ".join(m.value for m in SelectionMethod)
        raise ValueError(
            f"Invalid selection method '{method_str}'. Valid methods are: {valid_methods}"
        ) from None

    return method.xarray_method, actual_value


def apply_query(
    ds: xr.Dataset, *, variables: list[str], selectors: dict[str, Any]
) -> dict[str, ValidatedArray]:
    """
    This method does all automagic detection necessary for the rest of the pipeline to work.
    """
    validated: dict[str, ValidatedArray] = {}
    if selectors:
        # Apply selections serially to meet xarray limitations
        for name, value_str in selectors.items():
            if name not in ds:
                logger = get_context_logger()
                logger.warning(
                    "Selector not found in dataset, skipping",
                    selector=name,
                    value=value_str,
                )
                continue

            # Parse the selector to extract method and value
            try:
                method, value = parse_selector(str(value_str))
            except ValueError as e:
                raise IndexingError(str(e)) from None

            # If the value is not the same type as the variable, try to cast it
            try:
                typed_value = ds[name].dtype.type(value)
            except ValueError as e:
                logger = get_context_logger()

                if ds[name].dtype.kind == "m":
                    # Custom casting for timedelta64 if it fails
                    try:
                        typed_value = pd.to_timedelta(value).to_timedelta64()
                    except ValueError as tde:
                        logger.warning(
                            "Failed to cast selector to timedelta64",
                            selector=name,
                            value=value,
                            expected_type=ds[name].dtype,
                        )
                        raise tde
                elif ds[name].dtype.kind == "M":
                    # Custom casting for datetime64 if it fails
                    try:
                        typed_value = pd.to_datetime(value).to_datetime64()
                    except ValueError as tde:
                        logger.warning(
                            "Failed to cast selector to datetime64",
                            selector=name,
                            value=value,
                            expected_type=ds[name].dtype,
                        )
                        raise tde
                else:
                    logger.warning(
                        "Failed to cast selector",
                        selector=name,
                        value=value,
                        expected_type=ds[name].dtype,
                    )
                    raise e

            # Apply selection serially
            try:
                ds = ds.sel({name: typed_value}, method=method)
            except KeyError as e:
                raise IndexingError(str(e)) from None

    for name in variables:
        if name not in ds:
            raise VariableNotFoundError(
                f"Variable {name!r} not found in dataset."
            ) from None

        grid = guess_grid_system(ds, name)
        array = ds[name]
        if grid.Z in array.dims:
            # This code assumes all datasets are ocean datasets :/
            try:
                array = array.sel({grid.Z: 0}, method="nearest")
            except Exception as e:
                raise MissingParameterError(
                    f"Please pass an appropriate coordinate location for {grid.Z!r}. "
                    f"Automatic selection failed with error: {str(e)!r}."
                ) from None

        if extra_dims := (set(array.dims) - grid.dims):
            # Note: this will handle squeezing of label-based selection
            # along datetime coordinates
            array = array.isel(dict.fromkeys(extra_dims, -1))
        validated[name] = ValidatedArray(
            da=array,
            grid=grid,
            datatype=_infer_datatype(array),
        )
    return validated


def subset_to_bbox(
    validated: dict[str, ValidatedArray],
    *,
    bbox: OutputBBox,
    crs: OutputCRS,
    max_shape: tuple[int, int],
) -> dict[str, PopulatedRenderContext | NullRenderContext]:
    result = {}
    for var_name, array in validated.items():
        grid = array.grid
        if (ndim := array.da.ndim) > 2:
            raise ValueError(f"Attempting to visualize array with {ndim=!r} > 2.")
        # Check for insufficient data - either dimension has too few points
        if min(array.da.shape) < 2:
            raise ValueError(f"Data too small for rendering: {array.da.sizes!r}.")

        input_to_output = transformer_from_crs(crs_from=grid.crs, crs_to=crs)
        output_to_input = transformer_from_crs(crs_from=crs, crs_to=grid.crs)

        west, south, east, north = output_to_input.transform_bounds(
            left=bbox.west, right=bbox.east, top=bbox.north, bottom=bbox.south
        )
        if grid.crs.is_geographic:
            west = west - 360 if west > east else west

        input_bbox = BBox(west=west, south=south, east=east, north=north)
        input_bbox = round_bbox(input_bbox)

        if input_bbox.west > input_bbox.east:
            raise ValueError(f"Invalid Bbox after transformation: {input_bbox!r}")

        if not bbox_overlap(input_bbox, grid.bbox, grid.crs.is_geographic):
            result[var_name] = NullRenderContext()
            continue

        slicers = grid.sel(bbox=input_bbox)
        da = grid.assign_index(array.da)

        # Estimate coarsen factors and adjusted slicers
        coarsen_factors, new_slicers = estimate_coarsen_factors_and_slicers(
            da,
            grid=grid,
            slicers=slicers,
            max_shape=max_shape,
            datatype=array.datatype,
        )
        alternate = grid.pick_alternate_grid(crs, coarsen_factors=coarsen_factors)

        subset = asyncio.run(
            apply_slicers(
                da,
                grid=grid,
                alternate=alternate,
                slicers=new_slicers,
                datatype=array.datatype,
            )
        )

        if grid.crs.is_geographic:
            if isinstance(grid, GridSystem2D):
                has_discontinuity = has_coordinate_discontinuity(
                    subset[grid.X].data, axis=subset[grid.X].get_axis_num(grid.Xdim)
                )
            elif isinstance(grid, Triangular):
                anti = next(iter(slicers[grid.Xdim])).antimeridian_vertices
                has_discontinuity = anti["pos"].size > 0 or anti["neg"].size > 0
            else:
                raise NotImplementedError
        else:
            has_discontinuity = False

        if coarsen_factors:
            subset = coarsen(subset, coarsen_factors, grid=grid)

        with log_duration("transform_coordinates", "🔄"):
            newX, newY = transform_coordinates(
                subset, alternate.X, alternate.Y, transformer_from_crs(alternate.crs, crs)
            )

        # Fix coordinate discontinuities in transformed coordinates if detected
        if has_discontinuity:
            if isinstance(grid, GridSystem2D):
                fixed = fix_coordinate_discontinuities(
                    newX.data,
                    input_to_output,
                    axis=newX.get_axis_num(grid.Xdim),
                    bbox=bbox,
                )
                newX = newX.copy(data=fixed)
            elif isinstance(grid, Triangular):
                anti = next(iter(slicers[grid.dim])).antimeridian_vertices
                for verts in [anti["pos"], anti["neg"]]:
                    if verts.size > 0:
                        newX.data[verts] = fix_coordinate_discontinuities(
                            newX.data[verts],
                            input_to_output,
                            axis=newX.get_axis_num(grid.dim),
                            bbox=bbox,
                        )
        newda = subset.assign_coords({grid.X: newX, grid.Y: newY})
        result[var_name] = PopulatedRenderContext(
            da=newda,
            grid=grid,
            datatype=array.datatype,
            bbox=bbox,
            ugrid_indexer=next(iter(slicers[grid.Xdim]))
            if isinstance(grid, Triangular)
            else None,
        )
    return result
