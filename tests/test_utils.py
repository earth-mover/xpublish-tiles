import math

from xpublish_tiles.utils import normalize_longitude_deg, normalize_tilejson_bounds


def test_normalize_longitude_deg_basic():
    # Values within range unchanged
    assert normalize_longitude_deg(0.0) == 0.0
    assert normalize_longitude_deg(170.0) == 170.0
    assert normalize_longitude_deg(-170.0) == -170.0

    # Wrap-around cases
    assert normalize_longitude_deg(190.0) == -170.0
    assert normalize_longitude_deg(360.0) == 0.0
    assert normalize_longitude_deg(-190.0) == 170.0

    # Boundary values remain
    assert normalize_longitude_deg(180.0) == -180.0 or math.isclose(
        normalize_longitude_deg(180.0), -180.0
    )
    assert normalize_longitude_deg(-180.0) == -180.0


def test_normalize_tilejson_bounds():
    # Full world in 0..360 -> normalized to [-180, 180]
    assert normalize_tilejson_bounds([0.0, -90.0, 360.0, 90.0]) == [
        -180.0,
        -90.0,
        180.0,
        90.0,
    ]

    # Dateline-crossing region 350..10 -> normalized to full world
    assert normalize_tilejson_bounds([350.0, -10.0, 10.0, 10.0]) == [
        -180.0,
        -10.0,
        180.0,
        10.0,
    ]

    # Regular region stays the same
    assert normalize_tilejson_bounds([-170.0, -10.0, 170.0, 10.0]) == [
        -170.0,
        -10.0,
        170.0,
        10.0,
    ]
