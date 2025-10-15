#!/usr/bin/env python3

import numpy as np
import pyproj
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from xpublish_tiles.lib import epsg4326to3857


@given(
    lon=npst.arrays(
        dtype=np.float64,
        shape=10,
        elements=st.floats(
            min_value=-180.0, max_value=360.0, allow_nan=False, allow_infinity=False
        ),
    ),
    lat=npst.arrays(
        dtype=np.float64,
        shape=10,
        elements=st.floats(
            min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False
        ),
    ),
)
def test_epsg4326to3857_matches_pyproj(lon, lat):
    """Test that epsg4326to3857 matches pyproj's transformation."""
    x_ours, y_ours = epsg4326to3857(lon, lat)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_pyproj, y_pyproj = transformer.transform(lon, lat)
    np.testing.assert_allclose(x_ours, x_pyproj)
    np.testing.assert_allclose(y_ours, y_pyproj)


def test_epsg4326to3857_handles_0_360_range():
    """Test that epsg4326to3857 correctly handles 0-360 longitude range."""
    # Test that our function matches pyproj's behavior for wrap-around values
    # Note: pyproj treats 180° and -180° as different points (opposite sides of the world)

    # Test wrap-around for values that should be equivalent after normalization
    lon_wrapped = np.array([270.0, 359.0, 361.0, -181.0])
    lon_normal = np.array([-90.0, -1.0, 1.0, 179.0])
    lat = np.array([30.0, 0.0, 0.0, 0.0])

    # Transform both ranges
    x_wrapped, y_wrapped = epsg4326to3857(lon_wrapped, lat)
    x_normal, y_normal = epsg4326to3857(lon_normal, lat)

    # Results should be identical for wrapped values
    np.testing.assert_allclose(x_wrapped, x_normal)
    np.testing.assert_allclose(y_wrapped, y_normal)

    # Test edge case: longitude 359 should map to -1
    lon_edge = np.array([359.0, 1.0])
    lat_edge = np.array([0.0, 0.0])
    x_edge, _ = epsg4326to3857(lon_edge, lat_edge)

    # Compare with expected values from -1 and 1 degrees
    lon_expected = np.array([-1.0, 1.0])
    x_expected, _ = epsg4326to3857(lon_expected, lat_edge)

    np.testing.assert_allclose(x_edge, x_expected)

    # Test that 180° and -180° are treated as different points (matching pyproj)
    lon_extremes = np.array([-180.0, 180.0])
    lat_extremes = np.array([0.0, 0.0])
    x_extremes, _ = epsg4326to3857(lon_extremes, lat_extremes)

    # These should be opposite values
    assert x_extremes[0] == -x_extremes[1], (
        "180° and -180° should map to opposite X coordinates"
    )
