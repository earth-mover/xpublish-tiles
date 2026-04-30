import asyncio
import io
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import numpy as np
import pandas as pd
import pyproj
from pyproj.aoi import BBox

import xarray as xr
from xpublish_tiles.config import config
from xpublish_tiles.grids import (
    Curvilinear,
    FacetedGridSystem,
    FacetedIndexer,
    GridMetadata,
    GridSystem,
    GridSystem2D,
    Healpix,
    HealpixIndexer,
    Polar,
    Slicers,
    Triangular,
    UgridIndexer,
    guess_grid_system,
)
from xpublish_tiles.lib import (
    AsyncLoadTimeoutError,
    CoarsenedCoordinateIndices,
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
    normalize_slicers,
    pad_slicers,
    round_bbox,
    sum_tuples,
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
    Patch,
    PopulatedRenderContext,
    QueryParams,
    SelectionMethod,
    ValidatedArray,
)


def _apply_polar_pole_split(
    rings: np.ndarray, *, axis: int, pole_lat_tol: float = 1e-3
) -> np.ndarray:
    """Expand each pole-touching ring's apex into a horizontal edge at lat=±90.

    Polar-cap cells share a single corner with the geographic pole. In flat
    (lat, lon) rasterization that corner is a degenerate point, so only the
    pole-vertex pixel of the lat=±90 row is filled. Replace that vertex with
    two vertices at ``(ring_neighbour_lon, ±90)`` (one per adjacent ring edge),
    turning the apex into a horizontal segment along ``lat=±90`` that fills
    the cell's lon arc on the pole row.

    Input rings: ``(N, 5, 2)`` — 4 corners + closing vertex. The trailing
    pair holds ``(x, y)`` in the output CRS; ``axis`` (0 or 1) selects which
    entry holds the latitude — typically ``1`` (i.e. ``y`` is lat) for
    geographic output CRSes such as ``WorldCRS84Quad``. The other entry holds
    longitude.

    Output rings: ``(N, 6, 2)`` — pole rings carry the split; non-pole rings
    are padded with a repeat of the closing vertex (a zero-length edge that
    does not change rasterization) so the array stays uniform-width.

    Note: this is not triggered for WebMercator (which is limited to |lat| < 85°).
          It *is* triggered for WorldCRS84.
    """
    assert axis in (0, 1), axis
    lon_axis = 1 - axis
    n, m_in, _ = rings.shape
    assert m_in == 5

    interior = rings[:, :4, :]
    pole_mask = np.abs(np.abs(interior[..., axis]) - 90.0) < pole_lat_tol
    ring_has_pole = pole_mask.any(axis=1)

    out = np.empty((n, 6, 2), dtype=rings.dtype)
    out[:, :5, :] = rings
    out[:, 5, :] = rings[:, 4, :]

    pole_idx = np.where(ring_has_pole)[0]
    k_arr = np.argmax(pole_mask[pole_idx], axis=1)
    # Tiny loop: at most one cubed-sphere corner sits at each pole, so up to
    # 4 cells per pole share that vertex (8 total in a complete cubed-sphere).
    for ridx, k in zip(pole_idx.tolist(), k_arr.tolist(), strict=True):
        v = interior[ridx]
        pole_val = v[k, axis]
        lon_prev = v[(k - 1) % 4, lon_axis]
        lon_next = v[(k + 1) % 4, lon_axis]
        new = np.empty((6, 2), dtype=rings.dtype)
        j = 0
        for i in range(4):
            if i == k:
                new[j, lon_axis] = lon_prev
                new[j, axis] = pole_val
                j += 1
                new[j, lon_axis] = lon_next
                new[j, axis] = pole_val
                j += 1
            else:
                new[j] = v[i]
                j += 1
        new[j] = new[0]
        out[ridx] = new

    return out


def shape_from_slicers(
    slicers: Slicers,
    da: xr.DataArray,
    grid: GridSystem,
) -> tuple[int, ...]:
    """
    Calculate the total shape from slicers (element-wise sum of all subset shapes).

    Parameters
    ----------
    slicers : Slicers
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
    slicers: Slicers,
    da: xr.DataArray,
    grid: GridSystem,
) -> tuple[dict[str, int], Slicers]:
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
    slicers : Slicers
        Original slicers for data selection
    ds : xr.Dataset
        Dataset being processed
    grid : GridSystem
        Grid system information

    Returns
    -------
    tuple[dict[str, int], Slicers]
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
    slicers: Slicers,
    max_shape: tuple[int, int],
    datatype: DataType,
) -> tuple[dict[str, int], Slicers]:
    """
    Estimate coarsening factors and adjusted slicers for the given data array.

    Parameters
    ----------
    da : xr.DataArray
        Data array to process
    grid : GridSystem
        Grid system information
    slicers : Slicers
        Original slicers for data selection
    max_shape : tuple[int, int]
        Maximum allowed shape (width, height)
    datatype : DataType
        Data type information

    Returns
    -------
    tuple[dict[str, int], Slicers]
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
    new_slicers = apply_default_pad(new_slicers, da, grid)
    return coarsen_factors, new_slicers


@dataclass
class SubsetPlan:
    da_name: Hashable
    subsets: list[xr.Dataset]
    concat_dim: str
    pick: list[str]
    total_shape: tuple[int, ...]
    size_bytes: int


def apply_slicers(
    da: xr.DataArray,
    *,
    grid: GridSystem,
    alternate: GridMetadata,
    slicers: Slicers,
    datatype: DataType,
    min_dim_size: int = 2,
) -> SubsetPlan:
    has_alternate = alternate.crs != grid.crs
    pick = [alternate.X, alternate.Y]
    # For Healpix, also keep the cell_ids coordinate
    if isinstance(grid, Healpix):
        pick.append(grid.cell_ids_name)
    ds = cast(
        xr.Dataset,
        da.to_dataset()
        # drop any coordinate vars we don't need
        .reset_coords()[[da.name, *pick]],
    )

    subsets: list[xr.Dataset]
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
        concat_dim = grid.Xdim
    elif isinstance(grid, Triangular):
        subsets = [
            ds.isel({grid.Xdim: sl.vertices})
            for sl in slicers[grid.Xdim]
            if isinstance(sl, UgridIndexer)
        ]
        concat_dim = grid.Xdim
    elif isinstance(grid, Healpix):
        subsets = [
            ds.isel({grid.dim: sl.indices})
            for sl in slicers[grid.dim]
            if isinstance(sl, HealpixIndexer)
        ]
        concat_dim = grid.dim
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

    nvars = sum(len(subset.data_vars) for subset in subsets)
    if any(dim_total < min_dim_size * nvars for dim_total in total_shape):
        get_context_logger().error(
            "Tile request resulted in insufficient data for rendering."
        )
        raise AssertionError("Tile request resulted in insufficient data for rendering.")

    return SubsetPlan(
        da_name=da.name,
        subsets=subsets,
        concat_dim=concat_dim,
        pick=pick,
        total_shape=total_shape,
        size_bytes=total_size * da.dtype.itemsize * factor,
    )


async def load_plans(plans: list[SubsetPlan]) -> list[xr.DataArray]:
    """Aggregate TileTooBigError budget across ``plans`` and load everything
    in one TaskGroup; returns one xr.DataArray per plan, same order.
    """
    # Slight slack so subsets at the advertised minzoom that are just barely
    # over the limit are still served.
    fudge_factor = 1.1
    max_size = config.get("max_renderable_size")
    total_bytes = sum(p.size_bytes for p in plans)
    if total_bytes > fudge_factor * max_size:
        shapes = [p.total_shape for p in plans]
        msg = (
            f"Tile request too big, requires loading data of total size: "
            f"{total_bytes / 1024 / 1024}MB across shapes {shapes!r}. "
            "Please choose a higher zoom level."
        )
        get_context_logger().error(
            "Tile request too big",
            total_bytes=total_bytes,
            max_renderable_size=max_size,
            shapes=shapes,
        )
        raise TileTooBigError(msg)

    logger = get_context_logger()
    subsets: list[xr.Dataset] = []
    offsets: list[int] = [0]
    for plan in plans:
        subsets.extend(plan.subsets)
        offsets.append(len(subsets))

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
                    loaded_flat = [task.result() for task in tasks]
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
                loaded_flat = [s.load() for s in subsets]

    results: list[xr.DataArray] = []
    for i, plan in enumerate(plans):
        chunk = loaded_flat[offsets[i] : offsets[i + 1]]
        merged = xr.concat(chunk, dim=plan.concat_dim) if len(chunk) > 1 else chunk[0]
        results.append(merged.set_coords(plan.pick)[plan.da_name])
    return results


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

    # For 0→360 geographic data, also check if data spans the antimeridian (180°).
    # Symmetric across both equivalent unwrap branches: a face whose lons land in
    # [170, 190] crosses +180; the same face shifted by −360° lands in [−190, −170]
    # and crosses −180. Both branches denote the same antimeridian-straddling face
    # and must trigger the discontinuity fix.
    if check_antimeridian:
        x_min, x_max = coordinates.min(), coordinates.max()
        if x_min <= 180.0 <= x_max or x_min <= -180.0 <= x_max:
            return True

    return False


def fix_triangular_discontinuity(
    x_data: np.ndarray,
    groups: Iterable[np.ndarray],
    transformer: pyproj.Transformer,
    *,
    bbox: BBox,
) -> None:
    """Fix antimeridian discontinuities in-place, per group.

    Each element of ``groups`` is a 1-D index array into ``x_data``; the
    corresponding slice is unwrapped independently so the ``unwrap_phase``
    pass in :func:`fix_coordinate_discontinuities` doesn't bleed between
    unrelated sets of vertices (e.g. distinct Healpix cells).

    Used by the raster and polygon paths for triangular and healpix grids.
    """
    for verts in groups:
        if verts.size > 0:
            x_data[verts] = fix_coordinate_discontinuities(
                x_data[verts], transformer, bbox=bbox
            )


def fix_healpix_discontinuity(
    x_data: np.ndarray,
    y_data: np.ndarray,
    *,
    antimeridian_mask: np.ndarray,
    transformer: pyproj.Transformer,
    bbox: BBox,
    style: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | slice]:
    """Unwrap antimeridian-crossing HEALPix cell vertices, and for polygons
    also append ±width-shifted copies of those cells so whole-canvas tiles
    (e.g. WebMercator z=0) render both Cartesian sides. Non-spanning tiles
    clip the shifted copies off-canvas at no cost. ugrid achieves the same
    via duplicated pos/neg vertices baked into the CellTree.

    Returns ``(x, y, cell_indexer)``. ``cell_indexer`` is a per-cell index
    into the original data array — ``np.arange(n_cells)`` with AM cells
    appended twice when polygon cells were duplicated, else ``slice(None)``.
    """
    am_idxs = np.flatnonzero(antimeridian_mask)
    x_data = x_data.copy()
    left, _, right, _ = transformer.transform_bounds(-180, -90, 180, 90)
    width = abs(right - left)
    if style == "polygons":
        # Corners laid out (n_cells * 4,). AM-cell vertices split across the
        # seam by sign of projected x: pull the negative side by +width so each
        # cell is self-consistent. Vectorized over all seam cells (no per-cell
        # loop).
        am_verts = (am_idxs[:, None] * 4 + np.arange(4)).ravel()
        neg = x_data[am_verts] < 0
        x_data[am_verts[neg]] += width
    else:
        fix_triangular_discontinuity(x_data, [am_idxs], transformer, bbox=bbox)

    if style != "polygons" or not am_idxs.size:
        return x_data, y_data, slice(None)

    # now pad with cells that cross the anti-meridian
    dup_x = x_data[am_verts]
    dup_y = y_data[am_verts]
    x_data = np.concatenate([x_data, dup_x - width, dup_x + width])
    y_data = np.concatenate([y_data, dup_y, dup_y])
    n_cells = antimeridian_mask.size
    cell_indexer = np.concatenate([np.arange(n_cells), am_idxs, am_idxs])
    return x_data, y_data, cell_indexer


def fix_coordinate_discontinuities(
    coordinates: np.ndarray,
    transformer: pyproj.Transformer,
    *,
    bbox: BBox,
) -> np.ndarray:
    """
    Fix coordinate discontinuities that occur during coordinate transformation.

    When transforming geographic coordinates that cross the antimeridian (±180°)
    to projected coordinates (like Web Mercator), large gaps can appear in the
    transformed coordinate space. This function detects such gaps and applies
    intelligent offset corrections to make coordinates continuous.

    The algorithm:
    1. Uses skimage.restoration.unwrap_phase to fix coordinate discontinuities automatically
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
    unwrapped_coords = unwrap(coordinates, width=coordinate_space_width)

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
    renderer = query.get_renderer()
    subsets = await transform_for_render(
        subsets, bbox=query.bbox, crs=query.crs, style=renderer.style_id()
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
        if grid.Z is not None and grid.Z in array.coords:
            # This code assumes all datasets are ocean datasets :/
            if grid.Z not in array.xindexes:
                array = array.set_xindex(grid.Z)
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
    result: dict[str, PopulatedRenderContext | NullRenderContext] = {}
    plans: list[SubsetPlan] = []
    # ``plan_patches`` is aligned with ``plans``: each entry's loaded
    # DataArray is mutated back onto the patch's ``.da`` after ``load_plans``
    # returns. ``pending`` carries the per-var bookkeeping needed to build
    # the ``PopulatedRenderContext`` once data has loaded.
    plan_patches: list[Patch] = []
    pending: list[tuple[str, GridSystem, DataType, list[Patch]]] = []

    for var_name, array in validated.items():
        grid = array.grid

        output_to_input = transformer_from_crs(crs_from=crs, crs_to=grid.crs)

        west, south, east, north = output_to_input.transform_bounds(
            left=bbox.west, right=bbox.east, top=bbox.north, bottom=bbox.south
        )
        if grid.crs.is_geographic:
            west = west - 360 if west > east else west

        input_bbox = round_bbox(BBox(west=west, south=south, east=east, north=north))

        if input_bbox.west > input_bbox.east:
            raise ValueError(f"Invalid Bbox after transformation: {input_bbox!r}")

        # Build patches: one per overlapping face for FacetedGridSystem, one
        # patch for everything else.
        # Pre-load ``patch.da`` is the unloaded source array fed to ``apply_slicers``;
        # ``_post_load`` below mutates it to the loaded subset once ``load_plans`` returns.
        patches: list[Patch] = []

        if isinstance(grid, FacetedGridSystem):
            if style != "polygons":
                raise ValueError(
                    f"FacetedGridSystem (e.g. cubed sphere) only supports style='polygons'; got {style!r}."
                )
            slicers = grid.sel(bbox=input_bbox)
            face_indexer = cast(FacetedIndexer, next(iter(slicers[grid.face_dim])))
            for sel_slicers in face_indexer.selections:
                face_index = cast(slice, sel_slicers[grid.face_dim][0]).start
                face_grid = grid.faces[face_index]
                face_da = array.da.isel({grid.face_dim: face_index})
                face_slicers = normalize_slicers(
                    {k: v for k, v in sel_slicers.items() if k != grid.face_dim},
                    dict(face_da.sizes),
                )
                patches.append(
                    Patch(
                        grid=face_grid,
                        da=face_da,
                        slicers=face_slicers,
                        alternate=face_grid.to_metadata(),
                    )
                )
        else:
            if (ndim := array.da.ndim) > 2:
                raise ValueError(f"Attempting to visualize array with {ndim=!r} > 2.")
            if min(array.da.shape) < 2:
                raise ValueError(f"Data too small for rendering: {array.da.sizes!r}.")

            if not bbox_overlap(input_bbox, grid.bbox, grid.crs.is_geographic):
                result[var_name] = NullRenderContext()
                continue

            slicers = grid.sel(bbox=input_bbox)

            # ugly; figure out how to get rid of this
            if isinstance(grid, Healpix) and all(
                isinstance(s, HealpixIndexer) and s.is_empty for s in slicers[grid.dim]
            ):
                result[var_name] = NullRenderContext()
                continue

            da = grid.assign_index(array.da)
            coarsen_factors, new_slicers = estimate_coarsen_factors_and_slicers(
                da,
                grid=grid,
                slicers=slicers,
                max_shape=max_shape,
                datatype=array.datatype,
            )
            new_slicers = normalize_slicers(new_slicers, dict(da.sizes))
            alternate = grid.pick_alternate_grid(crs, coarsen_factors=coarsen_factors)

            # Note: These are handled specially inside apply_slicers.
            patch_indexer: UgridIndexer | HealpixIndexer | None = None
            if isinstance(grid, Triangular):
                patch_indexer = cast(UgridIndexer, next(iter(slicers[grid.Xdim])))
            elif isinstance(grid, Healpix):
                patch_indexer = cast(HealpixIndexer, next(iter(slicers[grid.dim])))

            patches.append(
                Patch(
                    grid=grid,
                    da=da,
                    slicers=new_slicers,
                    coarsen_factors=coarsen_factors,
                    alternate=alternate,
                    indexer=patch_indexer,
                )
            )

        if not patches:
            result[var_name] = NullRenderContext()
            continue

        plans.extend(
            apply_slicers(
                patch.da,
                grid=patch.grid,
                alternate=patch.alternate or patch.grid.to_metadata(),
                slicers=patch.slicers,
                datatype=array.datatype,
                min_dim_size=1 if style == "polygons" else 2,
            )
            for patch in patches
        )
        plan_patches.extend(patches)
        pending.append((var_name, grid, array.datatype, patches))

    loaded = await load_plans(plans) if plans else []

    async def _post_load(patch: Patch, ld: xr.DataArray) -> None:
        if isinstance(patch.grid, Polar):
            ld = patch.grid.assign_index(ld)
        if patch.coarsen_factors:
            ld = await async_run(
                partial(coarsen, ld, patch.coarsen_factors, grid=patch.grid)
            )
        patch.da = ld

    await asyncio.gather(
        *(_post_load(p, ld) for p, ld in zip(plan_patches, loaded, strict=True))
    )

    for var_name, grid, datatype, patches in pending:
        result[var_name] = PopulatedRenderContext(
            grid=grid,
            datatype=datatype,
            bbox=bbox,
            patches=patches,
        )
    return result


def _fix_discontinuity(
    grid: GridSystem,
    newX: xr.DataArray,
    newY: xr.DataArray,
    *,
    has_discontinuity: bool,
    transformer: pyproj.Transformer,
    bbox: OutputBBox,
    style: str,
    ugrid_indexer: UgridIndexer | None = None,
    hp_indexer: HealpixIndexer | None = None,
) -> tuple[xr.DataArray, xr.DataArray, np.ndarray | slice]:
    """Dispatch the post-transform antimeridian/discontinuity fix by grid type.

    Returns ``(newX, newY, healpix_cell_indexer)``. The third element is a
    slice/index array for Healpix augmentation (to apply to the values array);
    all other grids return ``slice(None)``.
    """
    if not has_discontinuity:
        return newX, newY, slice(None)
    if isinstance(grid, Triangular):
        assert ugrid_indexer is not None
        anti = ugrid_indexer.antimeridian_vertices
        fix_triangular_discontinuity(
            newX.data, [anti["pos"], anti["neg"]], transformer, bbox=bbox
        )
        return newX, newY, slice(None)
    if isinstance(grid, Healpix):
        assert hp_indexer is not None
        x_aug, y_aug, cell_idx = fix_healpix_discontinuity(
            newX.data,
            newY.data,
            antimeridian_mask=hp_indexer.antimeridian_mask,
            transformer=transformer,
            bbox=bbox,
            style=style,
        )
        return (
            xr.DataArray(x_aug, dims=newX.dims),
            xr.DataArray(y_aug, dims=newY.dims),
            cell_idx,
        )
    newX = newX.copy(
        data=fix_coordinate_discontinuities(newX.data, transformer, bbox=bbox)
    )
    return newX, newY, slice(None)


async def _transform_one_grid_polygons(
    grid: GridSystem2D,
    subset: xr.DataArray,
    *,
    slicers: Slicers,
    coarsen_factors: dict[str, int],
    bbox: OutputBBox,
    source_crs: pyproj.CRS,
    output_crs: OutputCRS,
    is_polar_cap: bool = False,
    out_width: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform one 2D grid's cell corners into output-CRS polygon rings.

    Per-patch core for the polygons path in :func:`transform_for_render`:
    cell_corners → discontinuity detect → transform → fix → corners_to_rings.

    When ``out_width`` is given (the faceted multi-patch case), also applies
    the ring-level fixups a local face patch needs: polar-cap seam shift or
    antimeridian ±width copy-stamping, so callers don't touch rings again.

    Returns ``(rings (N, 5, 2), values_1d (N,))``.
    """
    to_transform = grid.cell_corners(slicers=slicers, coarsen_factors=coarsen_factors)

    has_discontinuity = False
    if not is_polar_cap and source_crs.is_geographic:
        has_discontinuity = has_coordinate_discontinuity(
            to_transform[grid.X].data,
            360.0,
            axis=to_transform[grid.X].get_axis_num(grid.Xdim),
            check_antimeridian=True,
        )

    input_to_output = transformer_from_crs(source_crs, output_crs)
    with log_duration("transform_coordinates", "🔄"):
        newX, newY = await transform_coordinates(
            to_transform, grid.X, grid.Y, input_to_output
        )

    newX, newY, _ = _fix_discontinuity(
        grid,
        newX,
        newY,
        has_discontinuity=has_discontinuity,
        transformer=input_to_output,
        bbox=bbox,
        style="polygons",
    )

    if newX.ndim == 2:
        newX = newX.transpose(*subset.dims)
        newY = newY.transpose(*subset.dims)

    rings = grid.corners_to_rings(newX.data, newY.data)
    values = np.asarray(subset.values).ravel()

    if out_width is not None:
        if is_polar_cap:
            # A polar cap has a genuine topological singularity at the pole, so
            # the 2D `unwrap_phase` in `fix_coordinate_discontinuities` produces
            # garbage across large regions. pyproj has already normalized each
            # corner's x into one period; only cells straddling the antimeridian
            # need fixing. Within each such cell the vertices split into two
            # sign-groups across the seam; shift the negative side by
            # `+out_width` so each seam cell becomes self-consistent on one
            # side, then append ±out_width copies so the wrapped side renders.
            #
            # First, split pole-vertex apexes into horizontal edges at lat=±90
            # so the four cells per pole tile the lat=±90 row instead of all
            # converging to a single pixel. Widens rings from M=5 to M=6.
            rings = _apply_polar_pole_split(rings, axis=1)
            xs = rings[..., 0]
            seam_mask = (xs.max(axis=1) - xs.min(axis=1)) > out_width / 2
            if seam_mask.any():
                xs[seam_mask[:, None] & (xs < 0)] += out_width
                seam = rings[seam_mask]
                shift = np.array([out_width, 0.0])
                rings = np.concatenate([rings, seam - shift, seam + shift], axis=0)
                seam_values = values[seam_mask]
                values = np.concatenate([values, seam_values, seam_values])
        elif has_discontinuity:
            # The face straddles the antimeridian. fix_coordinate_discontinuities
            # unwraps to one side of the output CRS, so append ±width-shifted copies
            # to cover the wrapped side; off-canvas copies clip naturally.
            shift = np.array([out_width, 0.0])
            rings = np.concatenate([rings, rings - shift, rings + shift], axis=0)
            values = np.concatenate([values, values, values])

    return rings, values


async def _transform_polygon_patch(
    patch: Patch,
    *,
    bbox: OutputBBox,
    output_crs: OutputCRS,
    out_width: float | None = None,
) -> tuple[np.ndarray, np.ndarray, xr.DataArray]:
    """Transform one patch into ``(rings, values_1d, da)`` for the polygons style.

    Dispatches by patch grid type: GridSystem2D goes through the shared
    cell-corners → transform → corners-to-rings core; Triangular/Healpix have
    their own vertex/cell-corner machinery preserved here.

    ``out_width`` triggers the ring-level seam handling for Curvilinear face
    patches (polar caps and antimeridian-straddling faces); pass ``None`` for
    single-patch contexts.
    """
    grid = patch.grid
    subset = patch.da
    source_crs = patch.source_crs

    if isinstance(grid, GridSystem2D):
        is_polar_cap = isinstance(grid, Curvilinear) and grid.is_polar_cap
        rings, values = await _transform_one_grid_polygons(
            grid,
            subset,
            slicers=patch.slicers,
            coarsen_factors=patch.coarsen_factors,
            bbox=bbox,
            source_crs=source_crs,
            output_crs=output_crs,
            is_polar_cap=is_polar_cap,
            out_width=out_width,
        )
        return rings, values, xr.DataArray(values, dims=["cell"])

    # Triangular / Healpix polygon path. Source data is 1D per-vertex/per-cell;
    # cell_corners (Healpix) or pre-existing vertices (Triangular) feed the
    # transform.
    alternate = patch.alternate or grid.to_metadata()
    if isinstance(grid, Healpix):
        to_transform = grid.cell_corners(
            slicers=patch.slicers, coarsen_factors=patch.coarsen_factors
        )
    else:
        to_transform = subset

    has_discontinuity = False
    if source_crs.is_geographic:
        if isinstance(grid, Triangular):
            assert isinstance(patch.indexer, UgridIndexer)
            anti = patch.indexer.antimeridian_vertices
            has_discontinuity = anti["pos"].size > 0 or anti["neg"].size > 0
        elif isinstance(grid, Healpix):
            assert isinstance(patch.indexer, HealpixIndexer)
            has_discontinuity = bool(patch.indexer.antimeridian_mask.any())

    input_to_output = transformer_from_crs(source_crs, output_crs)
    with log_duration("transform_coordinates", "🔄"):
        newX, newY = await transform_coordinates(
            to_transform, alternate.X, alternate.Y, input_to_output
        )

    ugrid_indexer = patch.indexer if isinstance(patch.indexer, UgridIndexer) else None
    hp_indexer = patch.indexer if isinstance(patch.indexer, HealpixIndexer) else None
    newX, newY, healpix_cell_indexer = _fix_discontinuity(
        grid,
        newX,
        newY,
        has_discontinuity=has_discontinuity,
        transformer=input_to_output,
        bbox=bbox,
        ugrid_indexer=ugrid_indexer,
        hp_indexer=hp_indexer,
        style="polygons",
    )

    if isinstance(grid, Healpix):
        da = subset.isel({subset.dims[0]: healpix_cell_indexer})
    else:
        # Triangular: per-vertex values, renderer averages via connectivity.
        da = subset

    cell_rings = grid.corners_to_rings(newX.data, newY.data, ugrid_indexer=ugrid_indexer)
    values = (
        np.asarray(da.values).ravel() if da.values.ndim > 1 else np.asarray(da.values)
    )
    return cell_rings, values, da


async def _transform_raster_patch(
    patch: Patch,
    *,
    bbox: OutputBBox,
    output_crs: OutputCRS,
) -> xr.DataArray:
    """Transform one patch's coordinates for the raster style.

    Replaces the patch's X/Y center coords with the output-CRS values, with
    antimeridian/discontinuity fix applied. Returns the data array with new
    coords; the renderer consumes it via ``data[grid.X]``.
    """
    grid = patch.grid
    subset = patch.da
    source_crs = patch.source_crs
    alternate = patch.alternate or grid.to_metadata()

    has_discontinuity = False
    if source_crs.is_geographic:
        if isinstance(grid, Polar):
            has_discontinuity = False
        elif isinstance(grid, GridSystem2D):
            has_discontinuity = has_coordinate_discontinuity(
                subset[grid.X].data,
                360.0,
                axis=subset[grid.X].get_axis_num(grid.Xdim),
                check_antimeridian=True,
            )
        elif isinstance(grid, Triangular):
            assert isinstance(patch.indexer, UgridIndexer)
            anti = patch.indexer.antimeridian_vertices
            has_discontinuity = anti["pos"].size > 0 or anti["neg"].size > 0
        elif isinstance(grid, Healpix):
            assert isinstance(patch.indexer, HealpixIndexer)
            has_discontinuity = bool(patch.indexer.antimeridian_mask.any())
        else:
            raise NotImplementedError

    input_to_output = transformer_from_crs(source_crs, output_crs)
    with log_duration("transform_coordinates", "🔄"):
        newX, newY = await transform_coordinates(
            subset, alternate.X, alternate.Y, input_to_output
        )

    ugrid_indexer = patch.indexer if isinstance(patch.indexer, UgridIndexer) else None
    hp_indexer = patch.indexer if isinstance(patch.indexer, HealpixIndexer) else None
    newX, newY, _ = _fix_discontinuity(
        grid,
        newX,
        newY,
        has_discontinuity=has_discontinuity,
        transformer=input_to_output,
        bbox=bbox,
        ugrid_indexer=ugrid_indexer,
        hp_indexer=hp_indexer,
        style="raster",
    )

    return subset.assign_coords({grid.X: newX, grid.Y: newY})


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

    Iterates over ``context.patches`` uniformly: a non-faceted grid carries one
    patch, a FacetedGridSystem carries one per overlapping face. The polygons
    path concatenates rings/values across patches. The raster path requires a
    single patch (faceted-raster is rejected upstream).
    """
    result: dict[str, PopulatedRenderContext | NullRenderContext] = {}
    for var_name, context in contexts.items():
        if isinstance(context, NullRenderContext):
            result[var_name] = context
            continue

        if not context.patches:
            result[var_name] = NullRenderContext()
            continue

        grid = context.grid
        if style == "polygons":
            # Faceted grids need the output-CRS width for face seam handling
            # (polar caps and antimeridian-straddling faces) regardless of how
            # many faces overlap the tile. Non-faceted grids let
            # _transform_one_grid_polygons handle antimeridian via the standard
            # discontinuity-fix path.
            out_width: float | None = None
            if isinstance(grid, FacetedGridSystem):
                out_tx = transformer_from_crs(grid.crs, crs)
                out_left, _, out_right, _ = out_tx.transform_bounds(-180, -90, 180, 90)
                out_width = abs(out_right - out_left)

            rings_list: list[np.ndarray] = []
            values_list: list[np.ndarray] = []
            last_da: xr.DataArray | None = None
            for patch in context.patches:
                rings, values, da = await _transform_polygon_patch(
                    patch, bbox=bbox, output_crs=crs, out_width=out_width
                )
                rings_list.append(rings)
                values_list.append(values)
                last_da = da

            # Different patches may emit different ring widths (polar-cap
            # patches widen to M=6 to carry the pole-edge split; equatorial
            # cubed-sphere faces stay at M=5). Pad each by repeating the
            # closing vertex so that they are all a consistent size.
            # DC: if this ends up being a performance concern we could
            #     pass `rings_list` to the renderer where we convert each
            #     member of the list to a PolygonArray and then concatenate there.
            max_m = max(r.shape[1] for r in rings_list)
            for i, r in enumerate(rings_list):
                if r.shape[1] < max_m:
                    pad = np.repeat(r[:, -1:, :], max_m - r.shape[1], axis=1)
                    rings_list[i] = np.concatenate([r, pad], axis=1)
            cell_rings = np.concatenate(rings_list, axis=0)

            # Triangular patches keep per-vertex values (renderer averages
            # via connectivity); single-patch case preserves the original
            # DataArray so the renderer's connectivity lookup works.
            single = context.patches[0].grid if len(context.patches) == 1 else None
            if isinstance(single, Triangular):
                assert last_da is not None
                da_out = last_da
            else:
                values_concat = np.concatenate(values_list, axis=0)
                da_out = xr.DataArray(values_concat, dims=["cell"])

            result[var_name] = PopulatedRenderContext(
                grid=grid,
                datatype=context.datatype,
                bbox=bbox,
                patches=context.patches,
                da=da_out,
                cell_rings=cell_rings,
            )
            continue

        # Raster: single patch only (faceted-raster rejected in subset_to_bbox).
        assert len(context.patches) == 1, (
            f"raster style requires a single patch; got {len(context.patches)}"
        )
        (patch,) = context.patches
        da = await _transform_raster_patch(patch, bbox=bbox, output_crs=crs)
        result[var_name] = PopulatedRenderContext(
            grid=grid,
            datatype=context.datatype,
            bbox=bbox,
            patches=context.patches,
            da=da,
        )
    return result
