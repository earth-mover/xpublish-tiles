# FIXME: vendor these


import cf_xarray as cfxr
import cf_xarray.datasets
import pytest
from pyproj import CRS
from pyproj.aoi import BBox

import xarray as xr
from tests.tiles import TILES
from xpublish_tiles.datasets import EU3035, FORECAST, HRRR, HRRR_CRS_WKT, ROMSDS
from xpublish_tiles.grids import (
    X_COORD_PATTERN,
    Y_COORD_PATTERN,
    Curvilinear,
    GridSystem,
    RasterAffine,
    Rectilinear,
    guess_grid_system,
)

# FIXME: add tests for datasets with latitude, longitude but no attrs


@pytest.mark.parametrize(
    "ds, array_name, expected",
    (
        (
            FORECAST,
            "sst",
            Rectilinear(
                crs=CRS.from_user_input(4326),
                bbox=BBox(south=0, north=5, east=4, west=0),
                X="X",
                Y="Y",
                Z=None,
                indexes=(),  # type: ignore[arg-type]
            ),
        ),
        (
            ROMSDS,
            "temp",
            Curvilinear(
                crs=CRS.from_user_input(4326),
                bbox=BBox(south=0, north=11, east=11, west=0),
                X="lon_rho",
                Y="lat_rho",
                Z="s_rho",
                dims={"eta_rho", "xi_rho"},
                indexes=(),  # type: ignore[arg-type]
            ),
        ),
        (
            cfxr.datasets.popds,
            "UVEL",
            Curvilinear(
                crs=CRS.from_user_input(4326),
                bbox=BBox(south=2.5, north=2.5, east=0.5, west=0.5),
                X="ULONG",
                Y="ULAT",
                Z=None,
                dims={"nlon", "nlat"},
                indexes=(),  # type: ignore[arg-type]
            ),
        ),
        (
            cfxr.datasets.rotds,
            "temp",
            Rectilinear(
                crs=CRS.from_cf(
                    {
                        "grid_mapping_name": "rotated_latitude_longitude",
                        "grid_north_pole_latitude": 39.25,
                        "grid_north_pole_longitude": -162.0,
                    }
                ),
                bbox=BBox(south=21.615, north=21.835, east=18.155, west=17.935),
                X="rlon",
                Y="rlat",
                Z=None,
                indexes=(),  # type: ignore[arg-type]
            ),
        ),
        (
            HRRR.create(),
            "foo",
            Rectilinear(
                crs=CRS.from_wkt(HRRR_CRS_WKT),
                bbox=BBox(
                    west=-2697520.142522,
                    south=-1587306.152557,
                    east=2696479.857478,
                    north=1586693.847443,
                ),
                X="x",
                Y="y",
                Z=None,
                indexes=(),  # type: ignore[arg-type]
            ),
        ),
        (
            EU3035.create(),
            "foo",
            RasterAffine(
                crs=CRS.from_user_input(3035),
                bbox=BBox(
                    west=2635780.0,
                    south=1816000.0,
                    east=6235780.0,
                    north=5416000.0,
                ),
                X="x",
                Y="y",
                Z=None,
                indexes=(),  # type: ignore[arg-type]
            ),
        ),
    ),
)
def test_grid_detection(ds: xr.Dataset, array_name, expected: GridSystem) -> None:
    actual = guess_grid_system(ds, array_name)
    actual.indexes = ()  # FIXME
    assert expected == actual


@pytest.mark.parametrize("tile,tms", TILES)
def test_subset(global_datasets, tile, tms):
    """Test subsetting with tiles that span equator, anti-meridian, and poles."""
    ds = global_datasets
    grid = guess_grid_system(ds, "foo")
    geo_bounds = tms.bounds(tile)
    bbox_geo = BBox(
        west=geo_bounds[0], south=geo_bounds[1], east=geo_bounds[2], north=geo_bounds[3]
    )

    actual = grid.sel(ds.foo, bbox=bbox_geo)

    # Basic validation that we got a result
    assert isinstance(actual, xr.DataArray)
    assert actual.size > 0

    # Check that coordinates are within expected bounds (exact matching with controlled grid)
    lat_min, lat_max = actual.latitude.min().item(), actual.latitude.max().item()
    assert lat_min >= bbox_geo.south, f"Latitude too low: {lat_min} < {bbox_geo.south}"
    assert lat_max <= bbox_geo.north, f"Latitude too high: {lat_max} > {bbox_geo.north}"


def test_x_coordinate_regex_patterns():
    """Test that X coordinate regex patterns match expected coordinate names."""
    # Should match
    x_valid_names = [
        "x",
        "i",
        "nlon",
        "rlon",
        "ni",
        "lon",
        "longitude",
        "nav_lon",
        "glam",
        "glamv",
        "xlon",
        "xlongitude",
    ]

    for name in x_valid_names:
        assert X_COORD_PATTERN.match(name), f"X pattern should match '{name}'"

    # Should not match
    x_invalid_names = ["not_x", "X", "Y", "lat", "latitude", "foo", ""]

    for name in x_invalid_names:
        assert not X_COORD_PATTERN.match(name), f"X pattern should not match '{name}'"


def test_y_coordinate_regex_patterns():
    """Test that Y coordinate regex patterns match expected coordinate names."""
    # Should match
    y_valid_names = [
        "y",
        "j",
        "nlat",
        "rlat",
        "nj",
        "lat",
        "latitude",
        "nav_lat",
        "gphi",
        "gphiv",
        "ylat",
        "ylatitude",
    ]

    for name in y_valid_names:
        assert Y_COORD_PATTERN.match(name), f"Y pattern should match '{name}'"

    # Should not match
    y_invalid_names = ["not_y", "Y", "X", "lon", "longitude", "foo", ""]

    for name in y_invalid_names:
        assert not Y_COORD_PATTERN.match(name), f"Y pattern should not match '{name}'"
