import cf_xarray as cfxr  # noqa: F401
import numpy as np
import numpy.testing as npt
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
    CurvilinearCellIndex,
    GridSystem,
    LongitudeCellIndex,
    RasterAffine,
    Rectilinear,
    _prevent_slice_overlap,
    guess_grid_system,
)
from xpublish_tiles.lib import transformer_from_crs
from xpublish_tiles.pipeline import apply_slicers, fix_coordinate_discontinuities
from xpublish_tiles.testing.datasets import (
    CURVILINEAR,
    ERA5,
    EU3035,
    FORECAST,
    HRRR,
    HRRR_CRS_WKT,
    IFS,
    POPDS,
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
            CURVILINEAR.create(),
            "foo",
            Curvilinear(
                crs=CRS.from_user_input(4326),
                bbox=BBox(
                    south=-6.723446318626612,
                    north=12.551367116112495,
                    east=120.83720198174792,
                    west=115.1422776195327,
                ),
                X="lon",
                Y="lat",
                Xdim="xi_rho",
                Ydim="eta_rho",
                Z="s_rho",
                indexes=(
                    CurvilinearCellIndex(
                        X=CURVILINEAR.create().lon,
                        Y=CURVILINEAR.create().lat,
                        Xdim="xi_rho",
                        Ydim="eta_rho",
                    ),
                ),
            ),
            id="roms",
        ),
        pytest.param(
            POPDS,
            "UVEL",
            Curvilinear(
                crs=CRS.from_user_input(4326),
                bbox=BBox(south=2.5, north=2.5, east=0.5, west=0.5),
                X="ULONG",
                Y="ULAT",
                Xdim="nlon",
                Ydim="nlat",
                Z=None,
                indexes=(
                    CurvilinearCellIndex(
                        X=POPDS.cf["ULONG"], Y=POPDS.cf["ULAT"], Xdim="nlon", Ydim="nlat"
                    ),
                ),
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


@pytest.mark.asyncio
@pytest.mark.parametrize("tile,tms", TILES)
async def test_subset(global_datasets, tile, tms):
    """Test subsetting with tiles that span equator, anti-meridian, and poles."""
    ds = global_datasets
    grid = guess_grid_system(ds, "foo")
    geo_bounds = tms.bounds(tile)
    bbox_geo = BBox(
        west=geo_bounds[0], south=geo_bounds[1], east=geo_bounds[2], north=geo_bounds[3]
    )

    slicers = grid.sel(ds.foo, bbox=bbox_geo)
    assert isinstance(slicers["latitude"], list)
    assert isinstance(slicers["longitude"], list)
    assert len(slicers["latitude"]) == 1  # Y dimension should always have one slice

    # Check that coordinates are within expected bounds (exact matching with controlled grid)
    actual = await apply_slicers(ds.foo, grid=grid, slicers=slicers)
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


class TestLongitudeCellIndex:
    def test_longitude_cell_index_regional(self):
        """Test that LongitudeCellIndex.sel() method works correctly."""
        centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])  # Simple regional grid
        lon_index = LongitudeCellIndex(
            pd.IntervalIndex.from_arrays(centers - 0.5, centers + 0.5, closed="left"),
            "longitude",
        )
        assert not lon_index.is_global  # 5 degree span should not be global
        assert len(lon_index) == len(centers)  # Should have 5 intervals from 5 centers
        result = lon_index.sel({"longitude": slice(0, 1)})
        assert result.dim_indexers == {"longitude": (slice(2, 4),)}

    def test_longitude_cell_index_global_180(self):
        lon_index = LongitudeCellIndex(
            pd.IntervalIndex.from_breaks([-180, -1.0, 0.0, 1.0, 180], closed="left"),
            "longitude",
        )
        assert lon_index.is_global

        result = lon_index.sel({"longitude": slice(0, 1)})
        assert result.dim_indexers == {"longitude": (slice(2, 4),)}

        result = lon_index.sel({"longitude": slice(-185, 1)})
        assert result.dim_indexers == {"longitude": (slice(3, 4), slice(0, 3))}

        result = lon_index.sel({"longitude": slice(-220, -190)})
        assert result.dim_indexers == {"longitude": (slice(3, 4),)}

        result = lon_index.sel({"longitude": slice(190, 220)})
        assert result.dim_indexers == {"longitude": (slice(0, 1),)}

        result = lon_index.sel({"longitude": slice(150, 220)})
        assert result.dim_indexers == {"longitude": (slice(3, 4), slice(0, 1))}

    def test_longitude_cell_index_global_360(self):
        edges = [0, 90, 180, 270, 360]
        lon_index = LongitudeCellIndex(
            pd.IntervalIndex.from_breaks(edges, closed="left"), "longitude"
        )
        assert lon_index.is_global
        assert len(lon_index) == len(edges) - 1

        result = lon_index.sel({"longitude": slice(90, 220)})
        assert result.dim_indexers == {"longitude": (slice(1, 3),)}

        result = lon_index.sel({"longitude": slice(-90, 0)})
        assert result.dim_indexers == {"longitude": (slice(3, 4), slice(0, 1))}

        result = lon_index.sel({"longitude": slice(275, 365)})
        assert result.dim_indexers == {"longitude": (slice(3, 4), slice(0, 1))}

        result = lon_index.sel({"longitude": slice(-30, -10)})
        assert result.dim_indexers == {"longitude": (slice(3, 4),)}

        result = lon_index.sel({"longitude": slice(380, 420)})
        assert result.dim_indexers == {"longitude": (slice(0, 1),)}


class TestFixCoordinateDiscontinuities:
    """Test coordinate discontinuity fixing functionality."""

    def test_wrap_around_360_to_0_geographic(self):
        """Test fixing discontinuity when geographic coordinates wrap from 360 to 0 in 4326->4326 transform."""
        # This is the actual problematic array from the issue
        # fmt: off
        coords = np.array(
            [
                176.4, 180.0, 183.6, 187.2, 190.8, 194.4, 198.0, 201.6, 205.2, 208.8, 212.4, 216.0, 219.6, 223.2, 226.8, 230.4, 234.0, 237.6, 241.2, 244.8, 248.4, 252.0, 255.6, 259.2, 262.8, 266.4,
                270.0, 273.6, 277.2, 280.8, 284.4, 288.0, 291.6, 295.2, 298.8, 302.4, 306.0, 309.6, 313.2, 316.8, 320.4, 324.0, 327.6, 331.2, 334.8, 338.4, 342.0, 345.6, 349.2, 352.8, 356.4,
                0.0, 3.6, 7.2, 10.8, 14.4, 18.0, 21.6, 25.2, 28.8, 32.4, 36.0, 39.6, 43.2, 46.8, 50.4, 54.0, 57.6, 61.2, 64.8, 68.4, 72.0, 75.6, 79.2, 82.8, 86.4,
                90.0, 93.6, 97.2, 100.8, 104.4, 108.0, 111.6, 115.2, 118.8, 122.4, 126.0, 129.6, 133.2, 136.8, 140.4, 144.0, 147.6, 151.2, 154.8, 158.4, 162.0, 165.6, 169.2, 172.8, 176.4, 180.0,
            ]
        )
        # fmt: on
        expected = np.arange(-183.6, 183.6, 3.6)
        transformer = transformer_from_crs("EPSG:4326", "EPSG:4326", always_xy=True)
        transformed_x, _ = transformer.transform(coords, np.zeros_like(coords))
        bbox = BBox(west=-180, east=180, south=-90, north=90)
        fixed = fix_coordinate_discontinuities(
            transformed_x, transformer, axis=0, bbox=bbox
        )
        npt.assert_array_almost_equal(fixed, expected)

    def test_wrap_around_web_mercator(self):
        """Test fixing discontinuity in Web Mercator transformed coordinates."""
        coords = np.array([170, 175, 180, 185, 190, 195])
        transformer = transformer_from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        transformed_x, _ = transformer.transform(coords, np.zeros_like(coords))
        bbox = BBox(west=170, east=-170, south=-90, north=90)
        fixed = fix_coordinate_discontinuities(
            transformed_x, transformer, axis=0, bbox=bbox
        )
        WIDTH = 40075016.68557849  # SHOULD BE 20037508.34 * 2
        expected = transformed_x.copy()
        expected[:3] -= WIDTH
        npt.assert_array_almost_equal(fixed, expected)

    def test_wrap_around_180_to_minus_180(self):
        """Test fixing discontinuity when coordinates wrap from 180 to -180."""
        coords = np.array([170, 175, 180, -175, -170, -165])
        expected = np.array([170, 175, 180, 185, 190, 195])

        transformer = transformer_from_crs("EPSG:4326", "EPSG:4326", always_xy=True)
        transformed_x, _ = transformer.transform(coords, np.zeros_like(coords))
        bbox = BBox(west=170, east=180, south=-90, north=90)
        fixed = fix_coordinate_discontinuities(
            transformed_x, transformer, axis=0, bbox=bbox
        )
        npt.assert_array_equal(fixed, expected)

    def test_no_discontinuity(self):
        """Test that coordinates without discontinuity are not modified."""
        coords = np.array([0, 10, 20, 30, 40, 50])
        transformer = transformer_from_crs("EPSG:4326", "EPSG:4326", always_xy=True)
        transformed_x, _ = transformer.transform(coords, np.zeros_like(coords))
        bbox = BBox(west=-10, east=60, south=-90, north=90)
        fixed = fix_coordinate_discontinuities(
            transformed_x, transformer, axis=0, bbox=bbox
        )
        # Should not modify coordinates that don't have discontinuities
        npt.assert_array_equal(coords, fixed)

    def test_small_array(self):
        """Test with very small array."""
        coords = np.array([350, 0, 10])
        expected = np.array([-10, 0, 10])
        transformer = transformer_from_crs("EPSG:4326", "EPSG:4326", always_xy=True)
        transformed_x, _ = transformer.transform(coords, np.zeros_like(coords))
        bbox = BBox(west=-10, east=20, south=-90, north=90)
        fixed = fix_coordinate_discontinuities(
            transformed_x, transformer, axis=0, bbox=bbox
        )
        npt.assert_array_equal(fixed, expected)


def test_prevent_slice_overlap():
    """Test _prevent_slice_overlap function with realistic array index scenarios."""
    # Test single slice (no overlap possible)
    single = [slice(0, 10)]
    assert _prevent_slice_overlap(single) == single

    # Test empty list
    empty = []
    assert _prevent_slice_overlap(empty) == empty

    # Test typical longitude wrapping pattern (360-element array)
    # First slice: indices 300-359, Second slice: indices 0-59 (no overlap)
    longitude_wrap = [slice(300, 360), slice(0, 60)]
    result = _prevent_slice_overlap(longitude_wrap)
    # No adjustment needed since 60 < 300
    expected = [slice(300, 360), slice(0, 60)]
    assert result == expected

    # Test case where second slice erroneously extends into first slice's range
    overlap_case = [slice(300, 360), slice(0, 320)]
    result = _prevent_slice_overlap(overlap_case)
    # Second slice: stop=320 >= previous_start=300, so stop becomes 300
    expected = [slice(300, 360), slice(0, 300)]
    assert result == expected

    multiple = [slice(100, 200), slice(50, 150)]
    result = _prevent_slice_overlap(multiple)
    # First: slice(100, 200) - unchanged
    # Second: slice(50, 150) - stop=150 >= previous_start=100, so stop becomes 100 -> slice(50, 100)
    expected = [slice(100, 200), slice(50, 100)]
    assert result == expected

    # Test with step parameter (should be preserved)
    with_step = [slice(200, 300, 2), slice(100, 250, 3)]
    result = _prevent_slice_overlap(with_step)
    # Second slice: stop=250 >= previous_start=200, so stop becomes 200
    expected = [slice(200, 300, 2), slice(100, 200, 3)]
    assert result == expected
