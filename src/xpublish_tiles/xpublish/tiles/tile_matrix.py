"""Tile matrix set definitions for OGC Tiles API"""

import functools
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import Any, cast

import cf_xarray as cfxr  # noqa: F401 - needed to enable .cf accessor
import cftime
import morecantile
import morecantile.errors
import numpy as np
import pandas as pd
import pyproj
import pyproj.aoi

import xarray as xr
from xpublish_tiles.config import config
from xpublish_tiles.grids import GridSystem, guess_grid_system
from xpublish_tiles.lib import TileTooBigError, async_run, cf_get, timedelta_to_iso8601
from xpublish_tiles.logger import logger
from xpublish_tiles.tiles_lib import get_min_zoom
from xpublish_tiles.types import OutputBBox, OutputCRS
from xpublish_tiles.xpublish.tiles.types import (
    DimensionExtent,
    DimensionType,
    Link,
    TileMatrix,
    TileMatrixSet,
    TileMatrixSetLimit,
    TileMatrixSetSummary,
)

# Path of the tileMatrixSets routes under the default ``app_router_prefix``
DEFAULT_TMS_HREF = "/tiles/tileMatrixSets"


def get_tile_matrix_set(tms_id: str) -> TileMatrixSet:
    """Get a complete tile matrix set definition for any morecantile TMS.

    Args:
        tms_id: The tile matrix set identifier (e.g., 'WebMercatorQuad')

    Returns:
        TileMatrixSet object with all tile matrices

    Raises:
        ValueError: If the TMS ID is not found in morecantile
    """
    try:
        tms = morecantile.tms.get(tms_id)
    except morecantile.errors.InvalidIdentifier as e:
        raise ValueError(f"Tile matrix set '{tms_id}' not found") from e

    tile_matrices = [
        TileMatrix(
            id=matrix.id,
            scaleDenominator=matrix.scaleDenominator,
            topLeftCorner=list(matrix.pointOfOrigin),
            tileWidth=matrix.tileWidth,
            tileHeight=matrix.tileHeight,
            matrixWidth=matrix.matrixWidth,
            matrixHeight=matrix.matrixHeight,
        )
        for matrix in tms.tileMatrices
    ]

    return TileMatrixSet(
        id=str(tms.id),
        title=str(tms.title) if tms.title else tms_id,
        uri=str(tms.uri) if tms.uri else None,
        crs=tms.crs,
        tileMatrices=tile_matrices,
    )


def get_tile_matrix_set_summary(tms_id: str) -> TileMatrixSetSummary:
    """Get summary information for any morecantile tile matrix set.

    Args:
        tms_id: The tile matrix set identifier (e.g., 'WebMercatorQuad')

    Returns:
        TileMatrixSetSummary object

    Raises:
        ValueError: If the TMS ID is not found in morecantile
    """
    try:
        tms = morecantile.tms.get(tms_id)
    except morecantile.errors.InvalidIdentifier as e:
        raise ValueError(f"Tile matrix set '{tms_id}' not found") from e

    tms_id_str = str(tms.id)
    tms_title = str(tms.title) if tms.title else tms_id
    return TileMatrixSetSummary(
        id=tms_id_str,
        title=tms_title,
        uri=str(tms.uri) if tms.uri else None,
        crs=tms.crs,
        links=[
            Link(
                href=f"{DEFAULT_TMS_HREF}/{tms_id_str}",
                rel="self",
                type="application/json",
                title=f"{tms_title} tile matrix set",
            )
        ],
    )


# Legacy functions for backward compatibility
def get_web_mercator_quad() -> TileMatrixSet:
    """Get the complete WebMercatorQuad tile matrix set definition using morecantile"""
    return get_tile_matrix_set("WebMercatorQuad")


def get_web_mercator_quad_summary() -> TileMatrixSetSummary:
    """Get summary information for WebMercatorQuad tile matrix set using morecantile"""
    return get_tile_matrix_set_summary("WebMercatorQuad")


# Generate registry of all available tile matrix sets from morecantile
def _create_tms_registries() -> tuple[dict, dict]:
    """Create registries for all available morecantile TMS."""
    tms_sets = {}
    tms_summaries = {}

    for tms_id in morecantile.tms.list():
        # Create lambda functions that capture the tms_id
        tms_sets[tms_id] = lambda tid=tms_id: get_tile_matrix_set(tid)
        tms_summaries[tms_id] = lambda tid=tms_id: get_tile_matrix_set_summary(tid)

    return tms_sets, tms_summaries


# Registry of available tile matrix sets
TILE_MATRIX_SETS, TILE_MATRIX_SET_SUMMARIES = _create_tms_registries()


def extract_tile_bbox_and_crs(
    tileMatrixSetId: str, tileMatrix: int, tileRow: int, tileCol: int
) -> tuple[OutputBBox, OutputCRS]:
    """Extract bounding box and CRS from tile coordinates using morecantile.

    Args:
        tileMatrixSetId: ID of the tile matrix set
        tileMatrix: Zoom level/tile matrix ID
        tileRow: Row index of the tile
        tileCol: Column index of the tile

    Returns:
        tuple: (bbox as OutputBBox, OutputCRS object)

    Raises:
        ValueError: If tile matrix set not found
    """
    try:
        tms = morecantile.tms.get(tileMatrixSetId)
    except morecantile.errors.InvalidIdentifier as e:
        raise ValueError(f"Tile matrix set '{tileMatrixSetId}' not found") from e
    tile = morecantile.Tile(x=tileCol, y=tileRow, z=tileMatrix)

    # Get the bounding box in the TMS's CRS (projected coordinates)
    bbox = tms.xy_bounds(tile)
    output_bbox = OutputBBox(
        pyproj.aoi.BBox(
            west=bbox.left, south=bbox.bottom, east=bbox.right, north=bbox.top
        )
    )
    crs = pyproj.CRS.from_wkt(tms.crs.to_wkt())
    return output_bbox, OutputCRS(crs)


@dataclass
class MinZoomSource:
    """What minzoom needs from a variable: its grid, and an array for dims,
    dtype and chunk layout. No data is read."""

    grid: GridSystem
    da: xr.DataArray
    # ``xarray_object_key(da)``, computed once by the caller (needs cf_coords)
    signature: tuple
    # the owning dataset's ``_xpublish_id``; the minzoom cache is keyed on it
    xpublish_id: str | None


async def get_min_zooms(tms_id: str, sources: dict[str, MinZoomSource]) -> dict[str, int]:
    """Minimum zoom per variable. Variables sharing a grid and spatial dims compute once.

    Variables with no renderable zoom in this TMS are left out.
    """
    tms = morecantile.tms.get(tms_id)
    groups: dict[tuple, list[str]] = {}
    for var_name, source in sources.items():
        groups.setdefault((id(source.grid), source.signature), []).append(var_name)

    min_zooms: dict[str, int] = {}
    for names in groups.values():
        source = sources[names[0]]
        try:
            zoom = await async_run(
                get_min_zoom,
                grid=source.grid,
                tms=tms,
                da=source.da,
                style="raster",
                xpublish_id=source.xpublish_id,
            )
        except TileTooBigError as e:
            logger.info(f"skipping {names} in {tms_id}: {e}")
            continue
        min_zooms.update(dict.fromkeys(names, zoom))
    return min_zooms


def get_tile_matrix_limits(tms_id: str, zoom_levels: range) -> list[TileMatrixSetLimit]:
    """Full-extent tile matrix limits for each zoom level in ``zoom_levels``."""
    tms = morecantile.tms.get(tms_id)
    limits = []
    for z in zoom_levels:
        minmax = tms.minmax(z)
        limits.append(
            TileMatrixSetLimit(
                tileMatrix=str(z),
                minTileRow=minmax["y"]["min"],
                maxTileRow=minmax["y"]["max"],
                minTileCol=minmax["x"]["min"],
                maxTileCol=minmax["x"]["max"],
            )
        )
    return limits


def get_all_tile_matrix_set_ids() -> list[str]:
    """Get list of all available tile matrix set IDs."""
    return list(TILE_MATRIX_SETS.keys())


# Square quads with one tile at zoom 0 come first: web maps can draw those in their own CRS.
DEFAULT_TMS_PREFERENCE = (
    "EuropeanETRS89_LAEAQuad",
    "NZTM2000Quad",
    "UPSArcticWGS84Quad",
    "UPSAntarcticWGS84Quad",
    "WorldMercatorWGS84Quad",
    "WebMercatorQuad",
    "WorldCRS84Quad",
    "WGS1984Quad",
)
MAX_DISTINCT_CRS = 8
# Snap for time-axis steps in repeating intervals.
ONE_SECOND = np.timedelta64(1, "s")


@functools.cache
def _tms_crs(tms_id: str) -> pyproj.CRS:
    return pyproj.CRS.from_wkt(morecantile.tms.get(tms_id).crs.to_wkt())


def _single_crs(grids: Iterable[GridSystem]) -> pyproj.CRS | None:
    distinct = list({id(grid.crs): grid.crs for grid in grids}.values())
    if not distinct or len(distinct) > MAX_DISTINCT_CRS:
        return None
    first, *rest = distinct
    return first if all(first.equals(crs) for crs in rest) else None


def _is_wgs84_like_geographic(crs: pyproj.CRS) -> bool:
    """2D lon/lat in degrees on the WGS84 or GRS80 ellipsoid (e.g. NAD83, ETRS89)."""
    ellipsoid = crs.ellipsoid
    return (
        crs.is_geographic
        and len(crs.axis_info) == 2
        and all(axis.unit_name == "degree" for axis in crs.axis_info)
        and crs.prime_meridian is not None
        and crs.prime_meridian.longitude == 0
        and ellipsoid is not None
        and ellipsoid.semi_major_metre == 6378137
        and abs(ellipsoid.inverse_flattening - 298.257223563) < 1e-5
    )


def _crs_matches(crs: pyproj.CRS, tms_crs: pyproj.CRS) -> bool:
    # GRS80-based datums differ from WGS84 by ~1 m: close enough to pick a default.
    return crs.equals(tms_crs, ignore_axis_order=True) or (
        _is_wgs84_like_geographic(crs) and _is_wgs84_like_geographic(tms_crs)
    )


def get_default_tile_matrix_set(
    grids: Iterable[GridSystem], tms_ids: Iterable[str]
) -> str | None:
    """Pick the tile matrix set whose CRS matches the data's CRS, if the data has one CRS."""
    crs = _single_crs(grids)
    if crs is None:
        return None
    available = set(tms_ids)
    ordered = [t for t in DEFAULT_TMS_PREFERENCE if t in available] + sorted(
        available - set(DEFAULT_TMS_PREFERENCE)
    )
    return next((t for t in ordered if _crs_matches(crs, _tms_crs(t))), None)


def _repeating_intervals(
    values: np.ndarray,
    fmt_value: Callable[[Any], str],
    fmt_step: Callable[[Any], str],
    quantum: np.timedelta64 | None = None,
) -> list[str] | None:
    """Describe a 1D axis as ISO 8601 repeating intervals ``R<count>/<start>/<step>``.

    One interval per maximal run of constant step, as OGC API EDR ``extent.values``.
    The jump between runs is not a step of either run (GFS ``lead_time``:
    0-120 h hourly, then 123-384 h 3-hourly). None past the ``max_repeating_intervals`` option.
    ``quantum`` snaps steps to its multiples, so jitter does not split a run
    (GOES-19 ABI scan starts: 300.0 s and 300.1 s).
    """
    n = len(values)
    if n < 2:
        return None
    steps = values[1:] - values[:-1]
    if quantum is not None:
        steps = np.round(steps / quantum).astype(np.int64) * quantum
    # Candidate end: the step changes here. A segment start is never an end, so within
    # each chain of consecutive candidates every second one, from the first, is an end.
    candidate = np.zeros(n, dtype=bool)
    candidate[1:-1] = steps[1:] != steps[:-1]
    idx = np.arange(n)
    chain_start = np.maximum.accumulate(
        np.where(candidate & ~np.roll(candidate, 1), idx, 0)
    )
    is_end = candidate & ((idx - chain_start) % 2 == 0)
    is_end[-1] = True
    ends = np.flatnonzero(is_end)
    if len(ends) > config.get("max_repeating_intervals"):
        return None
    starts = np.concatenate([[0], ends[:-1] + 1])
    zero = values[0] - values[0]
    return [
        f"R{b - a + 1}/{fmt_value(values[a])}/{fmt_step(steps[a] if a < b else zero)}"
        for a, b in zip(starts, ends, strict=True)
    ]


def _single_run_resolution(
    repeating: list[str] | None,
) -> tuple[str | None, list[str] | None]:
    """One run spans the whole axis: send its step as ``resolution``, not as a run."""
    if repeating is not None and len(repeating) == 1:
        return repeating[0].rsplit("/", 1)[1], None
    return None, repeating


def _numeric_resolution(values: np.ndarray) -> int | float | None:
    """The absolute step when every step is equal (floats: to rounding), else None."""
    if len(values) < 2:
        return None
    steps = np.diff(values)
    if np.issubdtype(steps.dtype, np.integer):
        regular = bool(np.all(steps == steps[0]))
    else:
        regular = bool(np.allclose(steps, steps[0], rtol=1e-6, atol=0))
    if not regular or steps[0] == 0:
        return None
    return abs(steps[0].item())


def _nominal_resolution(values: np.ndarray) -> str | None:
    """Mean step over whole multiples, rounded to whole seconds, as an ISO 8601 duration.

    For jittered stamps that defeat ``infer_freq`` and repeating intervals (GOES
    scans: 600 s ± 1 s with outage gaps). The ``nominal_step_min_fraction`` and
    ``nominal_step_rtol`` options set how many steps must be near a whole multiple
    of it. Clients select with ``nearest::``.
    """
    if len(values) < 3:
        return None
    steps = np.diff(values).astype("timedelta64[ns]").astype(np.int64)
    rtol = config.get("nominal_step_rtol")

    def near_multiples(step: float) -> tuple[np.ndarray, np.ndarray]:
        multiple = np.round(steps / step)
        close = (multiple >= 1) & (np.abs(steps - multiple * step) <= rtol * step)
        return multiple, close

    median = float(np.median(steps))
    if median <= 0:
        return None
    # A median off by 1 s drifts ~1 day over 10^5 steps; the mean over multiples does not.
    multiple, close = near_multiples(median)
    if not close.any():
        return None
    step = round(steps[close].sum() / multiple[close].sum() / 1e9) * 1_000_000_000
    if step <= 0:
        return None
    _, close = near_multiples(step)
    if close.mean() < config.get("nominal_step_min_fraction"):
        return None
    return timedelta_to_iso8601(np.timedelta64(step, "ns"))


async def extract_dimension_extents(
    ds: xr.Dataset,
    name: Hashable,
    max_actual_values: int = 50,
    *,
    cf_coords: dict | None = None,
) -> list[DimensionExtent]:
    """Extract dimension extent information from an xarray DataArray.

    Uses cf_xarray to detect CF-compliant axes for robust dimension classification.

    Args:
        ds: xarray Dataset
        name: Name of the data variable in ``ds``
        max_actual_values: Maximum number of actual values to extract,
            otherwise only extents and resolution will be extracted.

    Returns:
        List of DimensionExtent objects for non-spatial dimensions
    """
    # Deferred import: metadata.py imports from this module, so the temporal
    # resolution helper can only be imported lazily to avoid a cycle.
    from xpublish_tiles.xpublish.tiles.metadata import _calculate_temporal_resolution

    dimensions = []

    grid = await async_run(guess_grid_system, ds, name, cf_coords=cf_coords)
    # Read as detection did, so a Z that only `cf_get` promotes reports as VERTICAL.
    data_array = cf_get(ds, name)

    # Grid-owned dims (X, Y, and faceted ``face_dim``) are skipped for
    # custom dimension extents — they're an implementation detail of the grid.
    spatial_dims = grid.dims_for(data_array)
    # ``grid.Z`` may be a non-dim coord (e.g. ``deptht`` along dim ``k``).
    # Map the underlying dim to the coord name so the extent reports the
    # coord variable instead of the bare dim.
    z_dim_to_coord: dict[Hashable, Hashable] = {}
    if grid.Z and grid.Z in data_array.coords:
        z_coord = data_array.coords[grid.Z]
        if z_coord.ndim == 1:
            z_dim_to_coord[z_coord.dims[0]] = grid.Z

    for dim_name in data_array.dims:
        if dim_name in spatial_dims:
            continue

        coord_name = z_dim_to_coord.get(dim_name, dim_name)
        coord = data_array.coords.get(coord_name)
        if coord is None:
            continue

        # Determine dimension type using CF axes
        dim_type = DimensionType.CUSTOM
        if coord_name == grid.Z:
            dim_type = DimensionType.VERTICAL

        # Extract coordinate values
        values = coord.values

        # Handle different coordinate types
        extent: list[str | float | int] = []
        actual_values: list[str | float | int] | None = None
        resolution: str | float | int | None = None
        repeating: list[str] | None = None

        if len(values) == 0:
            extent = []
        elif np.issubdtype(values.dtype, np.timedelta64):
            dim_type = DimensionType.TEMPORAL
            extent = [
                timedelta_to_iso8601(values[0]),
                timedelta_to_iso8601(values[-1]),
            ]
            if len(values) <= max_actual_values:
                actual_values = [timedelta_to_iso8601(value) for value in values]
            if len(values) > 1:
                resolution = _calculate_temporal_resolution(coord)
            if resolution is None and actual_values is None:
                repeating = _repeating_intervals(
                    values, timedelta_to_iso8601, timedelta_to_iso8601, ONE_SECOND
                )
                resolution, repeating = _single_run_resolution(repeating)
            if resolution is None and repeating is None and actual_values is None:
                resolution = _nominal_resolution(values)
        elif np.issubdtype(values.dtype, np.datetime64):
            dim_type = DimensionType.TEMPORAL
            # Convert datetime to ISO strings - only get first and last values for extent
            first_datetime = pd.to_datetime(values[0]).isoformat()
            last_datetime = pd.to_datetime(values[-1]).isoformat()
            extent = [first_datetime, last_datetime]
            if len(values) <= max_actual_values:
                actual_values = [
                    value.isoformat()
                    for value in cast(list[pd.Timestamp], pd.to_datetime(values))
                ]
            if len(values) > 1:
                resolution = _calculate_temporal_resolution(coord)
            if resolution is None and actual_values is None:
                repeating = _repeating_intervals(
                    values,
                    lambda v: pd.Timestamp(v).isoformat(),
                    timedelta_to_iso8601,
                    ONE_SECOND,
                )
                resolution, repeating = _single_run_resolution(repeating)
            if resolution is None and repeating is None and actual_values is None:
                resolution = _nominal_resolution(values)
        elif isinstance(values[0], cftime.datetime):
            # Non-standard calendars decode to cftime objects, not datetime64.
            dim_type = DimensionType.TEMPORAL
            extent = [values[0].isoformat(), values[-1].isoformat()]
            if len(values) <= max_actual_values:
                actual_values = [value.isoformat() for value in values]
            if len(values) > 1:
                resolution = _calculate_temporal_resolution(coord)
            if resolution is None and actual_values is None:
                repeating = _repeating_intervals(
                    values, lambda v: v.isoformat(), timedelta_to_iso8601
                )
        elif np.issubdtype(values.dtype, np.number):
            # ``.item()`` keeps ints as JSON ints, so clients can echo them back.
            extent = [values.min().item(), values.max().item()]
            if len(values) <= max_actual_values:
                actual_values = values.tolist()
            numeric = values
            if np.issubdtype(values.dtype, np.unsignedinteger):
                numeric = values.astype(np.int64)
            resolution = _numeric_resolution(numeric)
            if resolution is None and actual_values is None:
                repeating = _repeating_intervals(
                    numeric, lambda v: str(v.item()), lambda v: str(v.item())
                )
        else:
            extent = [str(values[0]), str(values[-1])]
            if len(values) <= max_actual_values:
                actual_values = [str(value) for value in values]

        # Get units and description from attributes
        units = coord.attrs.get("units")
        if np.issubdtype(values.dtype, np.timedelta64):
            # Extents are ISO 8601 durations, which the coord's own units
            # attribute (e.g. "nanoseconds") would contradict.
            units = None
        description = coord.attrs.get("long_name") or coord.attrs.get("description")

        # Determine default value (first value)
        default = None
        if extent:
            if dim_type == DimensionType.VERTICAL:
                default = extent[0]
            else:
                default = extent[-1]

        dimension = DimensionExtent(
            name=str(coord_name),
            type=dim_type,
            extent=extent,
            units=units,
            description=description,
            default=default,
            values=actual_values,
            resolution=resolution,
            repeating_intervals=repeating,
        )
        dimensions.append(dimension)

    return dimensions
