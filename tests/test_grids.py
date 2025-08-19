# FIXME: vendor these


import cf_xarray as cfxr
import cf_xarray.datasets
import numpy as np
import pandas as pd
import pytest
import rasterix
from affine import Affine
from pyproj import CRS
from pyproj.aoi import BBox

import xarray as xr
from xpublish_tiles.grids import (
    X_COORD_PATTERN,
    Y_COORD_PATTERN,
    Curvilinear,
    GridSystem,
    LongitudeCellIndex,
    RasterAffine,
    Rectilinear,
    guess_grid_system,
)
from xpublish_tiles.testing.datasets import (
    ERA5,
    EU3035,
    FORECAST,
    HRRR,
    HRRR_CRS_WKT,
    IFS,
    ROMSDS,
)
from xpublish_tiles.testing.tiles import TILES

# FIXME: add tests for datasets with latitude, longitude but no attrs


@pytest.mark.parametrize(
    "ds, array_name, expected",
    [
        pytest.param(
            IFS.create(),
            "foo",
            Rectilinear(
                crs=CRS.from_epsg(4326),
                bbox=BBox(west=-180, south=-90, east=180, north=90),
                X="longitude",
                Y="latitude",
                Z=None,
                indexes=(
                    LongitudeCellIndex(
                        pd.IntervalIndex.from_breaks(
                            np.arange(-180.0, 180.0 + 0.25, 0.25), closed="left"
                        ),
                        "longitude",
                    ),
                    xr.indexes.PandasIndex(
                        pd.IntervalIndex.from_breaks(
                            np.arange(-90.125, 90.125 + 0.25, 0.25), closed="right"
                        )[::-1],
                        "latitude",
                    ),
                ),
            ),
            id="ifs",
        ),
        pytest.param(
            ERA5.create(),
            "foo",
            Rectilinear(
                crs=CRS.from_epsg(4326),
                bbox=BBox(west=0, south=-90, east=360, north=90),
                X="longitude",
                Y="latitude",
                Z=None,
                indexes=(
                    LongitudeCellIndex(
                        pd.IntervalIndex.from_breaks(
                            np.arange(-0.125, 359.875 + 0.25, 0.25), closed="left"
                        ),
                        "longitude",
                    ),
                    xr.indexes.PandasIndex(
                        pd.IntervalIndex.from_breaks(
                            np.arange(-90.125, 90.125 + 0.25, 0.25), closed="right"
                        )[::-1],
                        "latitude",
                    ),
                ),
            ),
            id="era5",
        ),
        pytest.param(
            FORECAST,
            "sst",
            Rectilinear(
                crs=CRS.from_user_input(4326),
                bbox=BBox(south=-0.5, north=5.5, east=4.5, west=-0.5),
                X="X",
                Y="Y",
                Z=None,
                indexes=(
                    LongitudeCellIndex(
                        pd.IntervalIndex.from_breaks(
                            np.arange(-0.5, 4.5 + 1.0, 1.0), closed="left"
                        ),
                        "X",
                    ),
                    xr.indexes.PandasIndex(
                        pd.IntervalIndex.from_breaks(
                            np.arange(-0.5, 5.5 + 1.0, 1.0), closed="left"
                        ),
                        "Y",
                    ),
                ),
            ),
            id="forecast",
        ),
        pytest.param(
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
            id="roms",
        ),
        pytest.param(
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
            id="pop",
        ),
        # pytest.param(
        #     cfxr.datasets.rotds,
        #     "temp",
        #     Rectilinear(
        #         crs=CRS.from_cf(
        #             {
        #                 "grid_mapping_name": "rotated_latitude_longitude",
        #                 "grid_north_pole_latitude": 39.25,
        #                 "grid_north_pole_longitude": -162.0,
        #             }
        #         ),
        #         bbox=BBox(south=21.615, north=21.835, east=18.155, west=17.935),
        #         X="rlon",
        #         Y="rlat",
        #         Z=None,
        #         indexes=(),  # type: ignore[arg-type]
        #     ),
        #     id="rotated_pole"
        # ),
        pytest.param(
            HRRR.create(),
            "foo",
            Rectilinear(
                crs=CRS.from_wkt(HRRR_CRS_WKT),
                bbox=BBox(
                    west=-2699020.143,
                    south=-1588806.153,
                    east=2697979.857,
                    north=1588193.847,
                ),
                X="x",
                Y="y",
                Z=None,
                indexes=(
                    xr.indexes.PandasIndex(
                        pd.IntervalIndex.from_breaks(
                            np.arange(-2699020.142522, 2697979.857478 + 3000, 3000),
                            closed="left",
                            name="x",
                        ),
                        "x",
                    ),
                    xr.indexes.PandasIndex(
                        pd.IntervalIndex.from_breaks(
                            np.arange(-1588806.152557, 1588193.847443 + 3000, 3000),
                            closed="left",
                            name="y",
                        ),
                        "y",
                    ),
                ),
            ),
            id="hrrr",
        ),
        pytest.param(
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
                indexes=(
                    rasterix.RasterIndex.from_transform(
                        Affine(1200.0, 0.0, 2635780.0, 0.0, -1200.0, 5416000.0),
                        x_dim="x",
                        y_dim="y",
                        width=3000,
                        height=3000,
                    ),
                ),
            ),
            id="eu3035",
        ),
    ],
)
def test_grid_detection(ds: xr.Dataset, array_name, expected: GridSystem) -> None:
    actual = guess_grid_system(ds, array_name)
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
    assert lat_min <= bbox_geo.south, f"Latitude too low: {lat_min} < {bbox_geo.south}"
    assert lat_max >= bbox_geo.north, f"Latitude too high: {lat_max} > {bbox_geo.north}"


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


def test_longitude_cell_index_sel():
    """Test that LongitudeCellIndex.sel() method works correctly."""
    # Create a simple longitude index with regular spacing
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])  # Simple regional grid
    lon_index = LongitudeCellIndex(
        pd.IntervalIndex.from_arrays(centers - 0.5, centers + 0.5), "longitude"
    )

    assert bool(lon_index.is_global) is False  # 5 degree span should not be global
    assert len(lon_index) == 5  # Should have 5 intervals from 5 centers
    result = lon_index.sel({"longitude": slice(0, 1)})
    assert result.dim_indexers == {"longitude": slice(2, 4)}
