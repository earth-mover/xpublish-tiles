import asyncio
import io
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
    Polar,
    Triangular,
    UgridIndexer,
    guess_grid_system,
)
from xpublish_tiles.lib import (
    AsyncLoadTimeoutError,
    CoarsenedCoordinateIndices,
    Fill,
    IndexingError,
    MissingParameterError,
    PadDimension,
    TileTooBigError,
    VariableNotFoundError,
    _iter_subset_shapes,
    apply_default_pad,
    async_run,
    coarsen_mean_pad,
    get_data_load_semaphore,
    max_render_shape,
    pad_slicers,
    transform_coordinates,
    transformer_from_crs,
    unwrap,
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


def shape_from_slicers(
    slicers: dict[str, list[slice | Fill | UgridIndexer]],
    da: xr.DataArray,
    grid: GridSystem,
) -> tuple[int, ...]:
    """
    Calculate the total shape from slicers (element-wise sum of all subset shapes).

    Parameters
    ----------
    slicers : dict[str, list[slice | Fill | UgridIndexer]]
        Slicers for data selection
    da : xr.DataArray
        Data array (only metadata used, no data loaded)
    grid : GridSystem
        Grid system information

    Returns
    -------
    tuple[int, ...]
        Total shape from summing dimensions element-wise across all subsets
    """
    return sum_tuples(*_iter_subset_shapes(slicers, da, grid))


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

    def largest_odd_ge(a, b):
        """Return largest odd integer >= a/b (minimum 3), or None if < 2."""
        quotient = a // b
        if quotient < 2:
            return None
        if quotient < 3:
            return 3
        return quotient if quotient % 2 == 1 else quotient - 1

    coarsen_factors = {
        dim: factor
        for size, maxsize, dim in zip(shape, max_shape, dims, strict=True)
        if size > maxsize and (factor := largest_odd_ge(size, maxsize)) is not None
    }

    # For global datasets, pad longitude with wraparound to make exactly divisible.
    # Other dimensions use boundary="pad" in coarsen().
    sizes = dict(zip(dims, shape, strict=False))
    coarsen_padders = []
    for dim, factor in coarsen_factors.items():
        if not (grid.lon_spans_globe and dim == grid.Xdim):
            continue
        size = sizes[dim]
        remainder = size % factor
        if remainder == 0:
            continue
        pad_needed = factor - remainder
        coarsen_padders.append(
            PadDimension(
                name=dim,
                size=da.sizes[dim],
                left_pad=0,
                right_pad=pad_needed,
                wraparound=True,
                fill=False,
            )
        )
    new_slicers = pad_slicers(slicers, dimensions=coarsen_padders)

    return coarsen_factors, new_slicers


def estimate_coarsen_factors_and_slicers(
    da: xr.DataArray,
    *,
    grid: GridSystem,
    slicers: dict[str, list[slice | Fill | UgridIndexer]],
    max_shape: tuple[int, int],
    datatype: DataType,
    style: str,
) -> tuple[dict[str, int], dict[str, list[slice | Fill | UgridIndexer]]]:
    """
    Estimate coarsening factors and adjusted slicers for the given data array.

    Parameters
    ----------
    da : xr.DataArray
        Data array to process
    grid : GridSystem
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
    # Triangular grids can't be coarsened and don't need default_pad
    if not isinstance(grid, GridSystem2D):
        return {}, slicers

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
    if style != "polygons":
        new_slicers = apply_default_pad(new_slicers, da, grid)
    return coarsen_factors, new_slicers


async def apply_slicers(
    da: xr.DataArray,
    *,
    grid: GridSystem,
    alternate: GridMetadata,
    slicers: dict[str, list[slice | Fill | UgridIndexer]],
    datatype: DataType,
    min_dim_size: int = 2,
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
    # Slightly larger just in case the subset required for some tiles at the advertised minzoom
    # is just slightly too big
    fudge_factor = 1.1
    if total_size * da.dtype.itemsize > fudge_factor * factor * config.get(
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
    if any(total_size < min_dim_size * nvars for total_size in total_shape):
        logger.error("Tile request resulted in insufficient data for rendering.")
        raise AssertionError("Tile request resulted in insufficient data for rendering.")

    async with get_data_load_semaphore():
        if config.get("async_load"):
            with log_duration("async_load data subsets", "📥"):
                timeout = config.get("async_load_timeout_per_tile")
                try:
                    if timeout is not None:
                        async with asyncio.timeout(timeout), asyncio.TaskGroup() as tg:
                            tasks = [
                                tg.create_task(subset.load_async()) for subset in subsets
                            ]
                    else:
                        async with asyncio.TaskGroup() as tg:
                            tasks = [
                                tg.create_task(subset.load_async()) for subset in subsets
                            ]
                    results = [task.result() for task in tasks]
                except TimeoutError as e:
                    logger.error(
                        "Async data loading timed out", timeout=timeout, exc_info=e
                    )
                    raise AsyncLoadTimeoutError(
                        f"Async data loading timed out after {timeout}s. Server may be overloaded."
                    ) from None
                except ExceptionGroup as eg:
                    logger.error(
                        "Unhandled errors in TaskGroup",
                        error_count=len(eg.exceptions),
                        errors=[str(e) for e in eg.exceptions],
                    )
                    raise
        else:
            with log_duration("load data subsets", "📥"):
                results = [subset.load() for subset in subsets]
    subset = xr.concat(results, dim=grid.Xdim) if len(results) > 1 else results[0]
    subset_da = subset.set_coords(pick)[da.name]
    return subset_da


def coarsen(
    da: xr.DataArray, coarsen_factors: dict[str, int], *, grid: GridSystem2D
) -> xr.DataArray:
    """Coarsen data using odd integer factors with coordinate subselection.

    Uses boundary='pad' to handle incomplete windows (NaN-padded).
    For global datasets, longitude should already be padded via slicers
    to be exactly divisible.

    Coordinates are subselected at window centers rather than averaged.
    With this approach, we preserve exact coordinate values as present
    in the dataset. That in turn requires that coarsen_factors be odd.
    """
    with log_duration(f"coarsen {da.shape} by {coarsen_factors!r}", "🔲"):
        # Drop coordinates before coarsening to avoid extra work
        coord_names = list(da.coords)
        da_no_coords = da.drop_vars(coord_names)

        coarsened = coarsen_mean_pad(da_no_coords, coarsen_factors)

        # Subselect coordinates at window centers
        indexers = {}
        for dim, factor in coarsen_factors.items():
            assert factor % 2 == 1, f"{factor} should be odd."
            indexers[dim] = CoarsenedCoordinateIndices(da.sizes[dim], factor).centers()

        # Subselect coordinates using relevant indexers
        new_coords = {}
        for coord_name in coord_names:
            coord = da.coords[coord_name]
            coord_indexers = {dim: indexers[dim] for dim in coord.dims if dim in indexers}
            if coord_indexers:
                new_coords[coord_name] = coord.isel(coord_indexers)
            else:
                new_coords[coord_name] = coord

    return coarsened.assign_coords(new_coords)


def has_coordinate_discontinuity(
    coordinates: np.ndarray,
    coordinate_space_width: float,
    *,
    axis: int,
    check_antimeridian: bool = False,
) -> bool:
    """
    Detect coordinate discontinuities by checking for gaps > half the coordinate space width.

    Parameters
    ----------
    coordinates : np.ndarray
        Coordinates to analyze (geographic or projected)
    coordinate_space_width : float
        Width of the coordinate space (360.0 for geographic degrees,
        ~40M meters for Web Mercator, etc.)
    axis : int
        Axis along which to check for discontinuities
    check_antimeridian : bool
        If True, also check for 0→360 data spanning the antimeridian (180°).

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
    if coordinates.size == 0 or coordinate_space_width == 0:
        return False

    gaps = np.abs(np.diff(coordinates, axis=axis))
    if gaps.size == 0:
        return False

    if gaps.max() > coordinate_space_width / 2:
        return True

    # For 0→360 geographic data, also check if data spans the antimeridian (180°)
    if check_antimeridian:
        x_min, x_max = coordinates.min(), coordinates.max()
        if x_min <= 180.0 <= x_max:
            return True

    return False


def fix_antimeridian_vertices(
    x_data: np.ndarray,
    anti: dict[str, np.ndarray],
    transformer: pyproj.Transformer,
    *,
    bbox: BBox,
) -> None:
    """Fix antimeridian discontinuities in-place for the given vertex indices.

    Used by both the raster and polygon paths for triangular grids.
    ``anti`` is ``UgridIndexer.antimeridian_vertices``, mapping "pos"/"neg"
    to arrays of indices into ``x_data``.
    """
    for verts in [anti["pos"], anti["neg"]]:
        if verts.size > 0:
            x_data[verts] = fix_coordinate_discontinuities(
                x_data[verts], transformer, bbox=bbox
            )


def fix_coordinate_discontinuities(
    coordinates: np.ndarray,
    transformer: pyproj.Transformer,
    *,
    bbox: BBox,
    axis: int | None = None,
) -> np.ndarray:
    """
    Fix coordinate discontinuities that occur during coordinate transformation.

    When transforming geographic coordinates that cross the antimeridian (±180°)
    to projected coordinates (like Web Mercator), large gaps can appear in the
    transformed coordinate space. This function detects such gaps and applies
    intelligent offset corrections to make coordinates continuous.

    The algorithm:
    1. Uses unwrap to fix coordinate discontinuities automatically
    2. Calculates the expected coordinate space width using transformer bounds
    3. Shifts the result to maximize overlap with the bbox

    Parameters
    ----------
    coordinates : np.ndarray
        Coordinate values to fix.
    transformer : pyproj.Transformer
        Transformer from source to target CRS.
    bbox : BBox
        Target bounding box to align coordinates to.
    axis : int or None
        Axis along which to unwrap. None uses scikit-image unwrap_phase
        (spatial unwrapping across all dimensions). An integer uses
        numpy unwrap along that axis (independent per-slice unwrapping).

    Examples
    --------
    >>> import numpy as np
    >>> import pyproj
    >>> from pyproj.aoi import BBox
    >>> coords = np.array([350, 355, 360, 0, 5, 10])  # Wrap from 360 to 0
    >>> transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=True)
    >>> bbox = BBox(west=-10, east=20, south=-90, north=90)
    >>> fixed = fix_coordinate_discontinuities(coords, transformer, bbox=bbox)
    >>> gaps = np.diff(fixed)
    >>> assert np.all(np.abs(gaps) < 20), f"Large gap remains: {gaps}"
    """
    # Calculate coordinate space width using ±180° transform
    # This is unavoidable since AreaOfUse for a CRS is always in lat/lon
    # We are assuming that the "from" CRS for the transformer is geographic.
    assert transformer.source_crs is not None and transformer.source_crs.is_geographic
    left, _, right, _ = transformer.transform_bounds(-180, -90, 180, 90)
    coordinate_space_width = abs(right - left)

    if coordinate_space_width == 0:
        # ETRS89 returns +N for both -180 & 180
        # it's area of use is (-35.58, 24.6, 44.83, 84.73)
        # we ignore such things for now
        return coordinates

    # Step 1: Use unwrap to fix discontinuities
    unwrapped_coords = unwrap(coordinates, width=coordinate_space_width, axis=axis)

    # Step 2: Determine optimal shift based on coordinate and bbox bounds
    bbox_center = (bbox.west + bbox.east) / 2
    coord_center = (unwrapped_coords.min(axis=axis) + unwrapped_coords.max(axis=axis)) / 2

    # Calculate how many coordinate_space_widths we need to shift to align centers
    center_diff = bbox_center - coord_center
    shift_multiple = np.round(center_diff / coordinate_space_width)

    # Apply the calculated shift
    if axis is not None:
        # Expand shift for broadcasting (e.g. per-polygon shifts)
        shift = shift_multiple * coordinate_space_width
        result = unwrapped_coords + np.expand_dims(shift, axis=axis)
    else:
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
    max_shape = max_render_shape(
        style=query.style, width=query.width, height=query.height
    )

    # Capture the context logger before entering thread pool
    context_logger = get_context_logger()

    subsets = await subset_to_bbox(
        validated,
        bbox=query.bbox,
        crs=query.crs,
        max_shape=max_shape,
        style=query.style,
    )

    # Transform coordinates to output CRS
    subsets = await transform_for_render(
        subsets, bbox=query.bbox, crs=query.crs, style=query.style
    )
    renderer = query.get_renderer()

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
            abovemaxcolor=query.abovemaxcolor,
            belowmincolor=query.belowmincolor,
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


async def subset_to_bbox(
    validated: dict[str, ValidatedArray],
    *,
    bbox: OutputBBox,
    crs: OutputCRS,
    max_shape: tuple[int, int],
    style: str = "raster",
) -> dict[str, PopulatedRenderContext | NullRenderContext]:
    result = {}
    for var_name, array in validated.items():
        grid = array.grid
        if (ndim := array.da.ndim) > 2:
            raise ValueError(f"Attempting to visualize array with {ndim=!r} > 2.")
        # Check for insufficient data - either dimension has too few points
        if min(array.da.shape) < 2:
            raise ValueError(f"Data too small for rendering: {array.da.sizes!r}.")

        output_to_input = transformer_from_crs(crs_from=crs, crs_to=grid.crs)

        west, south, east, north = output_to_input.transform_bounds(
            left=bbox.west, right=bbox.east, top=bbox.north, bottom=bbox.south
        )
        if grid.crs.is_geographic:
            # Handle antimeridian crossing: west > east means bbox crosses -180/180
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

        coarsen_factors, new_slicers = estimate_coarsen_factors_and_slicers(
            da,
            grid=grid,
            slicers=slicers,
            max_shape=max_shape,
            datatype=array.datatype,
            style=style,
        )
        alternate = grid.pick_alternate_grid(crs, coarsen_factors=coarsen_factors)

        subset = await apply_slicers(
            da,
            grid=grid,
            alternate=alternate,
            slicers=new_slicers,
            datatype=array.datatype,
            min_dim_size=1 if style == "polygons" else 2,
        )

        # For Polar, compute lon/lat coordinates after subsetting
        if isinstance(grid, Polar):
            subset = grid.assign_index(subset)

        if coarsen_factors:
            subset = await async_run(partial(coarsen, subset, coarsen_factors, grid=grid))

        result[var_name] = PopulatedRenderContext(
            da=subset,
            grid=grid,
            datatype=array.datatype,
            bbox=bbox,
            ugrid_indexer=cast(UgridIndexer, next(iter(slicers[grid.Xdim])))
            if isinstance(grid, Triangular)
            else None,
            alternate=alternate,
            slicers=new_slicers,
            coarsen_factors=coarsen_factors,
        )
    return result


async def transform_for_render(
    contexts: dict[str, PopulatedRenderContext | NullRenderContext],
    *,
    bbox: OutputBBox,
    crs: OutputCRS,
    style: str,
) -> dict[str, PopulatedRenderContext | NullRenderContext]:
    """Transform coordinates to output CRS.

    For raster style: transform X/Y center coordinates to output CRS.
    For polygons style: transform cell corner coordinates, then assemble polygons.
    Both share the same transform + discontinuity-fix machinery.
    """
    result = {}
    for var_name, context in contexts.items():
        if isinstance(context, NullRenderContext):
            result[var_name] = context
            continue

        grid = context.grid
        alternate = context.alternate
        subset = context.da

        if alternate is None:
            result[var_name] = context
            continue

        if style == "polygons" and isinstance(grid, GridSystem2D):
            to_transform = grid.cell_corners(
                slicers=context.slicers,
                coarsen_factors=context.coarsen_factors,
            )
        else:
            to_transform = subset

        input_to_output = transformer_from_crs(alternate.crs, crs)
        with log_duration("transform_coordinates", "🔄"):
            newX, newY = await transform_coordinates(
                to_transform, alternate.X, alternate.Y, input_to_output
            )

        # Check for discontinuities on geographic coords before projection.
        # This is an optimization - we could also detect after projecting to the target CRS.
        # However, that would be a tax on every render. So instead we look for
        # the anti-meridian discontinuity specifically.
        if grid.crs.is_geographic:
            if isinstance(grid, Polar):
                has_discontinuity = False
            elif isinstance(grid, GridSystem2D):
                has_discontinuity = has_coordinate_discontinuity(
                    to_transform[grid.X].data,
                    360.0,
                    axis=to_transform[grid.X].get_axis_num(grid.Xdim),
                    check_antimeridian=True,
                )
            elif isinstance(grid, Triangular):
                assert context.ugrid_indexer is not None
                anti = context.ugrid_indexer.antimeridian_vertices
                has_discontinuity = anti["pos"].size > 0 or anti["neg"].size > 0
            else:
                raise NotImplementedError
        else:
            has_discontinuity = False

        # Fix coordinate discontinuities in transformed coordinates if detected
        # For example, when transforming to WebMercator pyproj will always return values
        # in the approximate range -20e6 -> 20e6 m range. So if the dataset's anti-meridian
        # discontinuity is in the tile; it will be preserved by the transformation,
        # regardless of how we may modify the coordinates *before* transforming.
        if has_discontinuity:
            if isinstance(grid, Triangular):
                assert context.ugrid_indexer is not None
                fix_antimeridian_vertices(
                    newX.data,
                    context.ugrid_indexer.antimeridian_vertices,
                    input_to_output,
                    bbox=bbox,
                )
            else:
                newX = newX.copy(
                    data=fix_coordinate_discontinuities(
                        newX.data,
                        input_to_output,
                        bbox=bbox,
                        # axis=None here helps handle tripole
                        axis=None,
                    )
                )

        if style == "polygons":
            if newX.ndim == 2:
                newX = newX.transpose(*subset.dims)
                newY = newY.transpose(*subset.dims)
            cell_rings = grid.corners_to_rings(
                newX.data, newY.data, ugrid_indexer=context.ugrid_indexer
            )
            # Flatten 2D data to match the 1D ring array from corners_to_rings.
            da = context.da
            if da.values.ndim > 1:
                da = xr.DataArray(da.values.ravel(), dims=["cell"])
        else:
            cell_rings = None
            da = subset.assign_coords({grid.X: newX, grid.Y: newY})

        result[var_name] = PopulatedRenderContext(
            da=da,
            grid=grid,
            datatype=context.datatype,
            bbox=bbox,
            ugrid_indexer=context.ugrid_indexer,
            alternate=alternate,
            cell_rings=cell_rings,
        )
    return result
