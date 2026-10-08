import asyncio

import numpy as np
import pyproj
import pytest

import xarray as xr
from xpublish_tiles.lib import transform_coordinates
from xpublish_tiles.projections import (
    CONIC_ALLOWLIST,
    CONIC_TOLERANCE_METERS,
    conic_to_cylindrical,
    geos_to_cylindrical,
    has_null_datum_shift,
    is_mercator_like,
    transformer_from_crs,
)

WEB_MERCATOR = pyproj.CRS.from_epsg(3857)

GEOS_HEIGHT = 35785831.0
# Scan angles out past the limb (~0.1518 rad), so the off-disk mask is exercised.
GEOS_MAX_SCAN = 0.16
GEOS_ELLIPSOIDS = {
    "ellipsoid": "+a=6378137 +b=6356752.31414",
    "sphere": "+R=6371229",
}
# Target -> tolerance in target units: 1e-3 m, or about 1 mm in degrees.
GEOS_TARGETS = {3857: 1e-3, 4326: 1e-8}


def _geos_crs(*, sweep="y", lon_0=0.0, ellipsoid="ellipsoid", extra=""):
    return pyproj.CRS.from_proj4(
        f"+proj=geos +h={GEOS_HEIGHT} +lon_0={lon_0} +sweep={sweep} "
        f"{GEOS_ELLIPSOIDS[ellipsoid]} +units=m {extra}"
    )


def _scan_axes(nx=401, ny=399):
    """Projection metres, as Geostationary.assign_index hands them over."""
    x = np.linspace(-GEOS_MAX_SCAN, GEOS_MAX_SCAN, nx) * GEOS_HEIGHT
    y = np.linspace(GEOS_MAX_SCAN, -GEOS_MAX_SCAN, ny) * GEOS_HEIGHT
    return x, y


# (code, x range, y range) in the CRS's own units
CONIC_EXTENTS = {
    5070: ((-2.36e6, 2.26e6), (0.27e6, 3.17e6)),  # NAD83 / Conus Albers
    6350: ((-2.36e6, 2.26e6), (0.27e6, 3.17e6)),  # NAD83(2011) / Conus Albers
    3005: ((0.2e6, 1.9e6), (0.35e6, 1.75e6)),  # NAD83 / BC Albers
}


def test_every_allowlisted_crs_is_covered():
    """Nothing goes on the allowlist without an extent to check it against."""
    assert set(CONIC_EXTENTS) == CONIC_ALLOWLIST


@pytest.mark.parametrize("code", CONIC_EXTENTS)
def test_conic_map_factors(code):
    """The allowlist asserts the composed map is a function of (rho, theta).

    Nothing at runtime checks that any more, so check it here: sample the area of
    use, and confirm the factored form reproduces pyproj everywhere on it.
    """
    src = pyproj.CRS.from_epsg(code)
    factored = conic_to_cylindrical(src, WEB_MERCATOR)
    assert factored is not None

    area = src.area_of_use
    assert area is not None
    lon, lat = (
        v.ravel()
        for v in np.meshgrid(
            np.linspace(area.west, area.east, 32),
            np.linspace(area.south, area.north, 32),
        )
    )
    x, y = transformer_from_crs(src.geodetic_crs, src).transform(lon, lat)
    result = factored.transform(x, y)
    assert result is not None
    expected = transformer_from_crs(src, 3857).transform(x, y)
    np.testing.assert_allclose(result[0], expected[0], atol=CONIC_TOLERANCE_METERS)
    np.testing.assert_allclose(result[1], expected[1], atol=CONIC_TOLERANCE_METERS)


@pytest.mark.parametrize("code", CONIC_EXTENTS)
def test_conic_to_cylindrical_matches_pyproj(code):
    factored = conic_to_cylindrical(pyproj.CRS.from_epsg(code), WEB_MERCATOR)
    assert factored is not None

    (x0, x1), (y0, y1) = CONIC_EXTENTS[code]
    x = np.linspace(x0, x1, 401)
    y = np.linspace(y1, y0, 397)
    result = factored.transform_grid(x, y)
    assert result is not None
    out_x, out_y = result

    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    expected_x, expected_y = transformer_from_crs(code, 3857).transform(grid_x, grid_y)
    np.testing.assert_allclose(out_x, expected_x, atol=CONIC_TOLERANCE_METERS)
    np.testing.assert_allclose(out_y, expected_y, atol=CONIC_TOLERANCE_METERS)


def test_conic_matches_allowlist_without_epsg_code():
    """A dataset carrying the same projection as a PROJ string still matches."""
    proj4 = (
        "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 "
        "+x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"
    )
    crs = pyproj.CRS.from_proj4(proj4)
    assert crs.area_of_use is None
    factored = conic_to_cylindrical(crs, WEB_MERCATOR)
    assert factored is not None

    x = np.linspace(-2.36e6, 2.26e6, 401)
    y = np.linspace(3.17e6, 0.27e6, 397)
    result = factored.transform_grid(x, y)
    assert result is not None
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    expected = transformer_from_crs(crs, 3857).transform(grid_x, grid_y)
    np.testing.assert_allclose(result[0], expected[0], atol=CONIC_TOLERANCE_METERS)
    np.testing.assert_allclose(result[1], expected[1], atol=CONIC_TOLERANCE_METERS)


@pytest.mark.parametrize(
    "code",
    [
        32631,  # UTM: transverse aspect, no cone
        2193,  # NZGD2000 / NZTM
        3395,  # cylindrical: meridians never converge
        31370,  # a real conic, but its datum shift is position-dependent
        3978,  # a real conic, but Arctic latitudes need an unaffordable table
    ],
)
def test_conic_to_cylindrical_rejects(code):
    assert conic_to_cylindrical(pyproj.CRS.from_epsg(code), WEB_MERCATOR) is None


def test_conic_to_cylindrical_rejects_rho_outside_table():
    """Out of table range returns None so the caller falls back to pyproj."""
    factored = conic_to_cylindrical(pyproj.CRS.from_epsg(5070), WEB_MERCATOR)
    assert factored is not None
    far = np.array([0.0])
    assert factored.transform_grid(far, np.array([-9e6])) is None


@pytest.mark.parametrize("curvilinear", [False, True])
def test_transform_coordinates_conic(curvilinear):
    crs = pyproj.CRS.from_epsg(5070)
    (x0, x1), (y0, y1) = CONIC_EXTENTS[5070]
    x = np.linspace(x0, x1, 301)
    y = np.linspace(y1, y0, 293)
    if curvilinear:
        grid_x, grid_y = np.meshgrid(x, y)
        subset = xr.DataArray(
            np.zeros(grid_x.shape),
            coords={"xc": (("j", "i"), grid_x), "yc": (("j", "i"), grid_y)},
            dims=("j", "i"),
        )
        names = ("xc", "yc")
    else:
        subset = xr.DataArray(
            np.zeros((y.size, x.size)), coords={"y": y, "x": x}, dims=("y", "x")
        )
        names = ("x", "y")

    transformer = transformer_from_crs(crs, 3857)
    out_x, out_y = asyncio.run(transform_coordinates(subset, *names, transformer))

    src_x, src_y = xr.broadcast(subset[names[0]], subset[names[1]])
    expected_x, expected_y = transformer.transform(src_x.data, src_y.data)
    np.testing.assert_allclose(
        out_x.transpose(*src_x.dims).data, expected_x, atol=CONIC_TOLERANCE_METERS
    )
    np.testing.assert_allclose(
        out_y.transpose(*src_y.dims).data, expected_y, atol=CONIC_TOLERANCE_METERS
    )


@pytest.mark.parametrize(
    "code, expected",
    [
        (4326, True),
        (4269, True),  # NAD83 -> WGS84 is a registered null transformation
        (4258, True),
        (4267, False),  # NAD27 shifts by ~140m
        (4277, False),  # OSGB36 shifts by ~115m
    ],
)
def test_has_null_datum_shift(code, expected):
    assert has_null_datum_shift(pyproj.CRS.from_epsg(code)) is expected


@pytest.mark.parametrize(
    "epsg, expected",
    [(3857, True), (3395, True), (32633, False), (4326, False), (3031, False)],
)
def test_is_mercator_like(epsg, expected):
    assert is_mercator_like(pyproj.CRS.from_epsg(epsg)) is expected


@pytest.mark.parametrize("sweep", ["x", "y"])
@pytest.mark.parametrize("ellipsoid", list(GEOS_ELLIPSOIDS))
@pytest.mark.parametrize("target", list(GEOS_TARGETS))
@pytest.mark.parametrize("lon_0", [0.0, 140.7, -75.2])
def test_geos_to_cylindrical_matches_pyproj(sweep, ellipsoid, target, lon_0):
    src = _geos_crs(sweep=sweep, lon_0=lon_0, ellipsoid=ellipsoid)
    fast = geos_to_cylindrical(src, pyproj.CRS.from_epsg(target))
    assert fast is not None

    x, y = _scan_axes()
    out_x, out_y = fast.transform_grid(x, y)

    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    expected_x, expected_y = transformer_from_crs(src, target).transform(grid_x, grid_y)
    finite = np.isfinite(expected_x)
    assert np.array_equal(np.isfinite(out_x), finite)
    assert np.array_equal(np.isfinite(out_y), finite)
    assert 0 < finite.sum() < finite.size
    np.testing.assert_allclose(
        out_x[finite], expected_x[finite], rtol=0, atol=GEOS_TARGETS[target]
    )
    np.testing.assert_allclose(
        out_y[finite], expected_y[finite], rtol=0, atol=GEOS_TARGETS[target]
    )


@pytest.mark.parametrize(
    "source, target",
    [
        (_geos_crs(extra="+x_0=1000"), WEB_MERCATOR),
        (_geos_crs(extra="+y_0=1000"), WEB_MERCATOR),
        (_geos_crs(), pyproj.CRS.from_epsg(3395)),  # ellipsoidal Mercator
        (_geos_crs(), pyproj.CRS.from_epsg(4267)),  # NAD27 shifts by ~140m
        (pyproj.CRS.from_epsg(5070), WEB_MERCATOR),
    ],
)
def test_geos_to_cylindrical_rejects(source, target):
    assert geos_to_cylindrical(source, target) is None


@pytest.mark.parametrize("target", list(GEOS_TARGETS))
def test_transform_coordinates_geos(target):
    """Same output contract as the pyproj path, off-disk cells included."""
    x, y = _scan_axes(nx=201, ny=203)
    subset = xr.DataArray(
        np.zeros((y.size, x.size)), coords={"y": y, "x": x}, dims=("y", "x")
    )
    transformer = transformer_from_crs(_geos_crs(lon_0=140.7), target)
    out_x, out_y = asyncio.run(transform_coordinates(subset, "x", "y", transformer))
    assert out_x.dims == ("x", "y")

    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    expected_x, expected_y = transformer.transform(grid_x, grid_y)
    if target == 3857:
        # The pyproj path NaNs points that are off the disk.
        off_disk = ~np.isfinite(expected_x)
        expected_x[off_disk] = np.nan
        expected_y[off_disk] = np.nan
    np.testing.assert_allclose(out_x.data, expected_x, rtol=0, atol=GEOS_TARGETS[target])
    np.testing.assert_allclose(out_y.data, expected_y, rtol=0, atol=GEOS_TARGETS[target])
