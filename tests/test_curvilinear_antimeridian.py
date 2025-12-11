"""
Test case for curvilinear grid wraparound detection.

This test creates a synthetic curvilinear grid that mimics the structure of HYCOM data:
- Grid spans globally (-180 to 180 longitude)
- At the western edge (X=0), longitude varies across latitudes
- Selecting regional data (e.g., Alaska) returns slices starting from X=0

The fix: Instead of using the grid-level `lon_spans_globe` flag to determine wraparound,
we check for actual longitude discontinuities in the selected tile data. This prevents
regional tiles from incorrectly getting wraparound padding just because the overall grid
spans the globe.
"""

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS
from pyproj.aoi import BBox

from xpublish_tiles.grids import Curvilinear, CurvilinearCellIndex


def create_curvilinear_grid_like_hycom():
    """
    Create a curvilinear grid that mimics HYCOM structure.

    Key characteristics:
    - Western edge (X=0) has longitude varying from -180 at equator to 180 at poles
    - This mimics data where the coordinate system wraps around
    - Grid is regional (Alaska/North Pacific) but coordinate edges span ±180
    """
    # Create a regional grid covering Alaska and North Pacific
    # Y dimension: roughly 50°N to 70°N (500 points)
    # X dimension: roughly -180° to -110° (900 points)

    ny, nx = 500, 900

    # Create latitude - straightforward, increases with Y
    lat_1d = np.linspace(50, 70, ny)

    # Create longitude - this is the tricky part
    # At the western edge (X=0), we want longitude to vary with Y
    # to mimic the wraparound behavior seen in HYCOM
    lon_base = np.linspace(-180, -110, nx)

    # Make it 2D - longitude varies with both X and Y (curvilinear)
    lat = np.repeat(lat_1d[:, np.newaxis], nx, axis=1)
    lon = np.zeros((ny, nx))

    # Key insight from HYCOM data: at X=0, longitude spans -180 to +180 across Y values
    # Make it wrap around multiple times to ensure any Y slice spans ±180
    # This mimics a grid where the western edge has highly varying longitude
    periods = 5  # Multiple wraps to ensure coverage
    lon[:, 0] = -180 + (np.arange(ny) % (ny // periods)) * (360.0 / (ny // periods))

    # At later X indices, longitude should converge to the regional values (around -157 to -146 for Alaska)
    # Create a gradient from the wrapped values at X=0 to regional values
    for x_idx in range(nx):
        # Weight decreases from 1 at X=0 to 0 at X=100
        wrap_weight = max(0, 1.0 - x_idx / 100.0)
        # Regional longitude for this X position
        regional_lon = -180 + (x_idx / nx) * 70  # -180 to -110
        # Blend between wrapped and regional
        if wrap_weight > 0:
            lon[:, x_idx] = wrap_weight * lon[:, 0] + (1 - wrap_weight) * regional_lon
        else:
            lon[:, x_idx] = regional_lon

    # Create a simple data variable (sea surface temperature)
    # with some interesting pattern
    data = 10 + 5 * np.sin(lat * np.pi / 180) * np.cos(lon * np.pi / 180)

    # Create dataset
    ds = xr.Dataset(
        {
            'sst': (['Y', 'X'], data),
            'lat': (['Y', 'X'], lat),
            'lon': (['Y', 'X'], lon),
        },
        coords={
            'Y': np.arange(ny),
            'X': np.arange(nx),
        }
    )

    return ds


def test_curvilinear_selection_with_wraparound():
    """
    Test that wraparound is correctly detected for curvilinear grids.

    When selecting a regional tile (Alaska -157° to -146°) from a grid that
    spans the globe, wraparound should NOT be enabled because the tile itself
    doesn't cross the antimeridian.

    This test verifies that we check for actual longitude discontinuities in
    the selected data rather than blindly using the grid-level lon_spans_globe flag.
    """
    ds = create_curvilinear_grid_like_hycom()

    # Verify the dataset has the expected structure
    assert ds.lon.isel(X=0).min().values < -170  # Western edge near -180
    assert ds.lon.isel(X=0).max().values > 170   # Western edge near +180
    assert abs(ds.lon.isel(X=-1).min().values - (-110)) < 1  # Eastern edge near -110

    # Create a Curvilinear grid
    # This will detect lon_spans_globe=True because the grid spans -180 to 180
    index = CurvilinearCellIndex(
        X=ds.lon,
        Y=ds.lat,
        Xdim='X',
        Ydim='Y',
    )

    grid = Curvilinear(
        crs=CRS.from_user_input(4326),
        bbox=BBox(
            west=float(ds.lon.min()),
            south=float(ds.lat.min()),
            east=float(ds.lon.max()),
            north=float(ds.lat.max()),
        ),
        X='lon',
        Y='lat',
        Xdim='X',
        Ydim='Y',
        indexes=(index,),
    )

    # Grid should detect it spans the globe
    assert grid.lon_spans_globe

    # Select a regional bbox (Alaska region: -157° to -146° lon, 58° to 60° lat)
    # This does NOT cross the antimeridian
    bbox = BBox(west=-157.5, south=58.0, east=-146.2, north=60.0)
    slicers = grid.sel(bbox=bbox)

    # Check what data gets selected
    x_slices = slicers['X']
    y_slice = slicers['Y'][0]

    # The PROBLEM: we get multiple X slices due to wraparound
    # This should NOT happen for a regional bbox that doesn't cross the antimeridian
    if len(x_slices) > 1:
        # Compute the combined longitude range from all slices
        all_lon_values = []
        for x_slice in x_slices:
            if isinstance(x_slice, slice):
                all_lon_values.append(ds.lon.isel(X=x_slice, Y=y_slice).values.ravel())

        combined_lon = np.concatenate(all_lon_values)
        lon_span = np.nanmax(combined_lon) - np.nanmin(combined_lon)

        # This assertion will FAIL (expected due to xfail marker)
        # When the bug is fixed, wraparound should not be enabled for this tile
        # and we should get only a single slice with regional data
        assert lon_span < 300, (
            f"Wraparound incorrectly enabled for regional tile. "
            f"Combined longitude spans {lon_span:.1f}° (entire globe). "
            f"This tile bbox ({bbox.west}° to {bbox.east}°) does NOT cross the antimeridian. "
            f"Expected: single slice with regional data (~11° span). "
            f"Got: {len(x_slices)} slices spanning {lon_span:.1f}°. "
            f"Root cause: At X=0, lon spans ±180° within this Y range, triggering wraparound in pad_slicers()."
        )

    # If we get here without the assertion failing, the test passes
    # (meaning only one slice, regional data - the correct behavior)
    assert len(x_slices) == 1, f"Expected 1 X slice for regional bbox, got {len(x_slices)}"
