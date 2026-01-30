"""
Swath grid support for radar and satellite data using pyresample.

This module provides a SwathGrid class that efficiently resamples swath/radar data
to map tiles using pyresample's KDTree-based nearest neighbor algorithm.

Performance characteristics:
- Neighbour info computation: ~20ms (cached per tile bbox)
- Resampling: ~3ms per variable/time step
- Total with cache: ~3-5ms per tile request
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cachetools
import numpy as np
from pyproj import CRS
from pyproj.aoi import BBox

import xarray as xr

if TYPE_CHECKING:
    from pyresample.geometry import SwathDefinition
    from pyresample.kd_tree import XArrayResamplerNN

# Lazy imports to avoid hard dependency
_pyresample_available = None


def _check_pyresample():
    """Check if pyresample is available."""
    global _pyresample_available
    if _pyresample_available is None:
        try:
            import pyresample  # noqa: F401

            _pyresample_available = True
        except ImportError:
            _pyresample_available = False
    return _pyresample_available


# Cache for resamplers (keyed by geometry + tile spec)
_RESAMPLER_CACHE: cachetools.LRUCache = cachetools.LRUCache(maxsize=128)
_RESAMPLER_CACHE_LOCK = threading.Lock()

# Default radius of influence in meters (50km covers most radar cell sizes)
DEFAULT_RADIUS_OF_INFLUENCE = 50000


def _compute_geometry_hash(lats: np.ndarray, lons: np.ndarray) -> str:
    """Compute a hash of the source geometry for caching."""
    lat_bytes = lats.tobytes() if lats.size < 10000 else lats[::100].tobytes()
    lon_bytes = lons.tobytes() if lons.size < 10000 else lons[::100].tobytes()
    return hashlib.md5(lat_bytes + lon_bytes).hexdigest()[:16]


def _create_target_area(
    bbox: BBox,
    width: int,
    height: int,
    crs: CRS | None = None,
) -> Any:
    """Create a pyresample AreaDefinition for the target tile."""
    from pyresample import create_area_def

    if crs is None or crs.to_epsg() == 4326:
        proj_dict = {"proj": "latlong", "datum": "WGS84"}
    else:
        proj_dict = crs.to_proj4()

    return create_area_def(
        "tile",
        proj_dict,
        area_extent=[bbox.west, bbox.south, bbox.east, bbox.north],
        shape=(height, width),
    )


def _create_source_definition(
    lats: xr.DataArray,
    lons: xr.DataArray,
) -> SwathDefinition:
    """Create a pyresample SwathDefinition from lat/lon coordinates."""
    from pyresample.geometry import SwathDefinition

    return SwathDefinition(lons=lons, lats=lats)


def _get_or_create_resampler(
    source_def: SwathDefinition,
    target_area: Any,
    geometry_hash: str,
    bbox: BBox,
    width: int,
    height: int,
    radius_of_influence: float = DEFAULT_RADIUS_OF_INFLUENCE,
) -> XArrayResamplerNN:
    """Get a cached resampler or create a new one."""
    from pyresample.kd_tree import XArrayResamplerNN

    cache_key = (
        geometry_hash,
        bbox.west,
        bbox.south,
        bbox.east,
        bbox.north,
        width,
        height,
    )

    with _RESAMPLER_CACHE_LOCK:
        if cache_key in _RESAMPLER_CACHE:
            return _RESAMPLER_CACHE[cache_key]

    # Create new resampler
    resampler = XArrayResamplerNN(
        source_def,
        target_area,
        radius_of_influence=radius_of_influence,
    )
    resampler.get_neighbour_info()

    with _RESAMPLER_CACHE_LOCK:
        _RESAMPLER_CACHE[cache_key] = resampler

    return resampler


@dataclass(kw_only=True)
class SwathGrid:
    """
    Grid system for swath/radar data using pyresample-based resampling.

    This grid type is optimized for data with:
    - Native polar coordinates (azimuth, range) or scan coordinates
    - 2D lat/lon coordinate arrays
    - Need for fast nearest-neighbor resampling to map tiles

    Unlike other grid types, SwathGrid performs resampling directly to the
    output tile resolution, bypassing the traditional sel→transform→render
    pipeline.

    Attributes
    ----------
    crs : CRS
        Coordinate reference system (typically EPSG:4326 for geographic)
    bbox : BBox
        Bounding box of the data extent
    lat_name : str
        Name of the latitude coordinate variable
    lon_name : str
        Name of the longitude coordinate variable
    dims : tuple[str, str]
        Dimension names (e.g., ('azimuth', 'range'))
    source_def : SwathDefinition
        Pyresample swath definition for the source data
    geometry_hash : str
        Hash of the source geometry for caching
    radius_of_influence : float
        Search radius in meters for nearest neighbor lookup
    """

    crs: CRS
    bbox: BBox
    lat_name: str
    lon_name: str
    dim_names: tuple[str, str]
    source_def: Any = field(repr=False)
    geometry_hash: str = field(repr=False)
    radius_of_influence: float = DEFAULT_RADIUS_OF_INFLUENCE

    # For compatibility with GridSystem interface
    X: str = field(init=False)
    Y: str = field(init=False)
    Xdim: str = field(init=False)
    Ydim: str = field(init=False)
    Z: str | None = None
    indexes: tuple = field(default_factory=tuple)
    alternates: tuple = field(default_factory=tuple)
    dXmin: float = 0.0
    dYmin: float = 0.0

    def __post_init__(self):
        self.X = self.lon_name
        self.Y = self.lat_name
        self.Xdim = self.dim_names[0]
        self.Ydim = self.dim_names[1]

    @property
    def dims(self) -> set[str]:
        """Return the set of dimension names."""
        return set(self.dim_names)

    @classmethod
    def from_dataset(
        cls,
        ds: xr.Dataset,
        lat_name: str = "lat",
        lon_name: str = "lon",
        radius_of_influence: float = DEFAULT_RADIUS_OF_INFLUENCE,
    ) -> SwathGrid:
        """
        Create a SwathGrid from a dataset with 2D lat/lon coordinates.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset with 2D lat/lon coordinate arrays
        lat_name : str
            Name of the latitude coordinate
        lon_name : str
            Name of the longitude coordinate
        radius_of_influence : float
            Search radius in meters for nearest neighbor lookup

        Returns
        -------
        SwathGrid
            Configured swath grid instance
        """
        if not _check_pyresample():
            raise ImportError(
                "pyresample is required for SwathGrid. "
                "Install with: pip install pyresample"
            )

        lats = ds[lat_name]
        lons = ds[lon_name]

        if lats.ndim != 2 or lons.ndim != 2:
            raise ValueError(
                f"SwathGrid requires 2D lat/lon coordinates. "
                f"Got lat.ndim={lats.ndim}, lon.ndim={lons.ndim}"
            )

        if lats.dims != lons.dims:
            raise ValueError(
                f"lat and lon must have the same dimensions. "
                f"Got lat.dims={lats.dims}, lon.dims={lons.dims}"
            )

        # Compute bounding box
        lat_min = float(lats.min())
        lat_max = float(lats.max())
        lon_min = float(lons.min())
        lon_max = float(lons.max())

        bbox = BBox(west=lon_min, south=lat_min, east=lon_max, north=lat_max)

        # Create source definition
        source_def = _create_source_definition(lats, lons)

        # Compute geometry hash for caching
        geometry_hash = _compute_geometry_hash(
            lats.values if hasattr(lats, "values") else np.asarray(lats),
            lons.values if hasattr(lons, "values") else np.asarray(lons),
        )

        return cls(
            crs=CRS.from_epsg(4326),
            bbox=bbox,
            lat_name=lat_name,
            lon_name=lon_name,
            dim_names=tuple(str(d) for d in lats.dims),
            source_def=source_def,
            geometry_hash=geometry_hash,
            radius_of_influence=radius_of_influence,
        )

    def resample_to_tile(
        self,
        data: xr.DataArray,
        bbox: BBox,
        width: int = 256,
        height: int = 256,
        target_crs: CRS | None = None,
    ) -> np.ndarray:
        """
        Resample data to a tile using cached pyresample resampler.

        Parameters
        ----------
        data : xr.DataArray
            Data to resample (must have same dimensions as lat/lon)
        bbox : BBox
            Bounding box of the output tile
        width : int
            Tile width in pixels
        height : int
            Tile height in pixels
        target_crs : CRS, optional
            Target CRS (default: EPSG:4326)

        Returns
        -------
        np.ndarray
            Resampled data as 2D array of shape (height, width)
        """
        target_area = _create_target_area(bbox, width, height, target_crs)

        resampler = _get_or_create_resampler(
            self.source_def,
            target_area,
            self.geometry_hash,
            bbox,
            width,
            height,
            self.radius_of_influence,
        )

        result = resampler.get_sample_from_neighbour_info(data)

        if hasattr(result, "values"):
            return result.values
        return np.asarray(result)

    def clear_cache(self):
        """Clear the resampler cache for this geometry."""
        with _RESAMPLER_CACHE_LOCK:
            keys_to_remove = [k for k in _RESAMPLER_CACHE if k[0] == self.geometry_hash]
            for k in keys_to_remove:
                del _RESAMPLER_CACHE[k]


def is_swath_data(ds: xr.Dataset, var_name: str | None = None) -> bool:
    """
    Detect if a dataset contains swath/radar data.

    Criteria:
    1. Has 2D 'lat' and 'lon' coordinates (or similar names)
    2. Dimensions are NOT standard geographic names
    3. Optionally checks a specific variable

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to check
    var_name : str, optional
        Specific variable to check dimensions against

    Returns
    -------
    bool
        True if dataset appears to be swath data
    """
    # Look for 2D geographic coordinates
    lat_candidates = ["lat", "latitude", "lats"]
    lon_candidates = ["lon", "longitude", "lons"]

    lat_name = None
    lon_name = None

    for name in lat_candidates:
        if name in ds.coords and ds[name].ndim == 2:
            lat_name = name
            break

    for name in lon_candidates:
        if name in ds.coords and ds[name].ndim == 2:
            lon_name = name
            break

    if lat_name is None or lon_name is None:
        return False

    # Check that dimensions are not standard geographic
    standard_dims = {"lat", "latitude", "lon", "longitude", "x", "y"}
    coord_dims = set(ds[lat_name].dims)

    if coord_dims & standard_dims:
        return False

    # Check for radar/swath-like dimension names
    swath_dims = {
        "azimuth",
        "range",
        "elevation",
        "scan",
        "along_track",
        "across_track",
        "scan_line",
        "pixel",
        "nrays",
        "ngates",
    }

    if coord_dims & swath_dims:
        return True

    # If we have 2D lat/lon with non-standard dims, likely swath data
    return True


def get_swath_coordinate_names(ds: xr.Dataset) -> tuple[str, str] | None:
    """
    Find the lat/lon coordinate names in a swath dataset.

    Returns
    -------
    tuple[str, str] | None
        (lat_name, lon_name) or None if not found
    """
    lat_candidates = ["lat", "latitude", "lats"]
    lon_candidates = ["lon", "longitude", "lons"]

    lat_name = None
    lon_name = None

    for name in lat_candidates:
        if name in ds.coords and ds[name].ndim == 2:
            lat_name = name
            break

    for name in lon_candidates:
        if name in ds.coords and ds[name].ndim == 2:
            lon_name = name
            break

    if lat_name and lon_name:
        return (lat_name, lon_name)
    return None
