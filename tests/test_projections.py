import asyncio

import numpy as np
import pyproj
import pytest

import xarray as xr
from xpublish_tiles.lib import transform_coordinates
from xpublish_tiles.projections import (
    CONIC_TOLERANCE_METERS,
    conic_to_cylindrical,
    has_null_datum_shift,
    transformer_from_crs,
)

WEB_MERCATOR = pyproj.CRS.from_epsg(3857)

# (code, x range, y range) in the CRS's own units
CONIC_EXTENTS = {
    5070: ((-2.36e6, 2.26e6), (0.27e6, 3.17e6)),  # NAD83 / Conus Albers
    3978: ((-2.4e6, 3.0e6), (-1.0e6, 3.5e6)),  # NAD83 / Canada Atlas Lambert
    3005: ((0.2e6, 1.9e6), (0.35e6, 1.75e6)),  # NAD83 / BC Albers
}


@pytest.mark.parametrize("code", CONIC_EXTENTS)
def test_conic_to_cylindrical_matches_pyproj(code):
    factored = conic_to_cylindrical(pyproj.CRS.from_epsg(code), WEB_MERCATOR)
    assert factored is not None

    (x0, x1), (y0, y1) = CONIC_EXTENTS[code]
    x = np.linspace(x0, x1, 401)
    y = np.linspace(y1, y0, 397)
    result = factored.transform(x, y, grid=True)
    assert result is not None
    out_x, out_y = result

    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    expected_x, expected_y = transformer_from_crs(code, 3857).transform(grid_x, grid_y)
    np.testing.assert_allclose(out_x, expected_x, atol=CONIC_TOLERANCE_METERS)
    np.testing.assert_allclose(out_y, expected_y, atol=CONIC_TOLERANCE_METERS)


@pytest.mark.parametrize(
    "code",
    [
        32631,  # UTM: transverse aspect, no cone
        2193,  # NZGD2000 / NZTM
        3395,  # cylindrical: meridians never converge
        31370,  # a real conic, but its datum shift is position-dependent
    ],
)
def test_conic_to_cylindrical_rejects(code):
    assert conic_to_cylindrical(pyproj.CRS.from_epsg(code), WEB_MERCATOR) is None


def test_conic_to_cylindrical_rejects_rho_outside_table():
    """Out of table range returns None so the caller falls back to pyproj."""
    factored = conic_to_cylindrical(pyproj.CRS.from_epsg(5070), WEB_MERCATOR)
    assert factored is not None
    far = np.array([0.0])
    assert factored.transform(far, np.array([-9e6]), grid=True) is None


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
