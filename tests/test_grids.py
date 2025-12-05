from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal, cast
from unittest.mock import patch

import cf_xarray as cfxr  # noqa: F401
import morecantile
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import rasterix
from affine import Affine
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pyproj import CRS
from pyproj.aoi import BBox

import xarray as xr
from xpublish_tiles.config import config
from xpublish_tiles.grids import (
    X_COORD_PATTERN,
    Y_COORD_PATTERN,
    Curvilinear,
    CurvilinearCellIndex,
    GridSystem,
    GridSystem2D,
    LongitudeCellIndex,
    RasterAffine,
    Rectilinear,
    Triangular,
    UgridIndexer,
    guess_grid_system,
)
from xpublish_tiles.lib import _prevent_slice_overlap, transformer_from_crs
from xpublish_tiles.pipeline import apply_slicers, fix_coordinate_discontinuities
from xpublish_tiles.testing.datasets import (
    CURVILINEAR,
    ERA5,
    EU3035,
    EU3035_HIRES,
    FORECAST,
    HRRR,
    HRRR_CRS_WKT,
    HRRR_MULTIPLE,
    IFS,
    PARA_HIRES,
    POPDS,
    REDGAUSS_N320,
    UTM33S_HIRES,
    UTM50S_HIRES,
    Dataset,
)
from xpublish_tiles.testing.tiles import TILES
from xpublish_tiles.tiles_lib import get_max_zoom, get_min_zoom
from xpublish_tiles.types import ContinuousData

TRIANGULAR_SENTINEL = 1


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
                        width=3011,
                        height=3011,
                    ),
                ),
            ),
            id="eu3035",
        ),
        pytest.param(
            REDGAUSS_N320.create(),
            "foo",
            TRIANGULAR_SENTINEL,
            id="redgauss_n320",
        ),
    ],
)
def test_grid_detection(ds: xr.Dataset, array_name, expected: GridSystem) -> None:
    actual = guess_grid_system(ds, array_name)
    if expected is TRIANGULAR_SENTINEL:
        # too hard to construct for a test
        assert isinstance(actual, Triangular)
        assert actual.dim == "point"
        assert actual.bbox == BBox(west=-180, east=180, south=-89.784877, north=89.784877)
        assert actual.crs == CRS.from_epsg(4326)
        assert actual.lon_spans_globe
        assert len(actual.indexes) == 1
    else:
        assert expected == actual


@pytest.mark.parametrize(
    "dataset, minzoom, maxzoom",
    (
        pytest.param(IFS, 0, 3, id="ifs"),
        pytest.param(HRRR, 0, 6, id="hrrr"),
        pytest.param(REDGAUSS_N320, 0, 24, id="redgauss_n320"),
        # data spacing: 120m; Zoom level 10: 152m spacing @ eq
        pytest.param(EU3035_HIRES, 5, 10, id="eu3035_hires"),
        # data spacing: 30m; Zoom level 13: 38m spacing @ eq
        pytest.param(PARA_HIRES, 7, 13, id="para_hires"),
        # data spacing: 0.5m; Zoom level 19: 0.3m spacing @ eq
        pytest.param(UTM33S_HIRES, 13, 19, id="utm33s_hires"),
        # data spacing: 1m; Zoom level 18: 0.6m spacing @ eq
        pytest.param(UTM50S_HIRES, 12, 18, id="utm50s_hires"),
    ),
)
def test_unit_minmax_zoom_level(dataset: Dataset, minzoom, maxzoom):
    ds = dataset.create()
    grid = guess_grid_system(ds, "foo")
    tms = morecantile.tms.get("WebMercatorQuad")
    assert get_min_zoom(grid, tms, ds["foo"]) == minzoom
    if minzoom == 0:
        assert ds.foo.size < config.get("max_renderable_size")
    assert get_max_zoom(grid, tms) == maxzoom


def test_multiple_grid_mappings_detection() -> None:
    """Test detection of datasets with multiple grid mappings that create alternates."""
    ds = HRRR_MULTIPLE.create()
    grid = guess_grid_system(ds, "foo")

    # Should be a RasterAffine grid system (HRRR's native Lambert Conformal Conic projection)
    assert isinstance(grid, RasterAffine)

    # Should have 2 alternates (since 3 total grid mappings, first becomes primary)
    assert len(grid.alternates) == 2

    # Alternates are now GridMetadata objects with grid_cls field
    # We expect at least one with Curvilinear grid_cls (for geographic coordinates)
    assert any(alt.grid_cls == Curvilinear for alt in grid.alternates)

    # Check that we have the expected CRS systems
    # Grid should be a GridSystem2D which has crs, X, Y attributes
    assert isinstance(grid, GridSystem2D)
    if TYPE_CHECKING:
        grid = cast(GridSystem2D, grid)

    all_crs = [grid.crs] + [alt.crs for alt in grid.alternates]
    assert {crs.to_epsg() for crs in all_crs} == {None, 4326, 27700}

    # Check coordinate variables are different for each grid system
    coord_pairs = [(grid.X, grid.Y)] + [(alt.X, alt.Y) for alt in grid.alternates]

    # Should have geographic coordinates and projected coordinates
    assert ("longitude", "latitude") in coord_pairs  # Geographic coordinates
    # Should also have projected coordinates (x, y for various projections)
    assert ("x", "y") in coord_pairs  # Projected coordinates


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

    slicers = grid.sel(bbox=bbox_geo)
    if isinstance(grid, Triangular):
        assert isinstance(slicers["point"], list)
        assert len(slicers["point"]) == 1
        slicer = next(iter(slicers["point"]))
        assert isinstance(slicer, UgridIndexer)
    else:
        assert isinstance(slicers["latitude"], list)
        assert isinstance(slicers["longitude"], list)
        assert len(slicers["latitude"]) == 1  # Y dimension should always have one slice

    # Check that coordinates are within expected bounds (exact matching with controlled grid)
    actual = await apply_slicers(
        ds.foo,
        grid=grid,
        alternate=grid.to_metadata(),
        slicers=slicers,
        datatype=ContinuousData(valid_min=0, valid_max=1),
    )
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
        assert result.dim_indexers == {"longitude": [slice(2, 4)]}

    def test_longitude_cell_index_global_180(self):
        lon_index = LongitudeCellIndex(
            pd.IntervalIndex.from_breaks([-180, -1.0, 0.0, 1.0, 180], closed="left"),
            "longitude",
        )
        assert lon_index.is_global

        result = lon_index.sel({"longitude": slice(0, 1)})
        assert result.dim_indexers == {"longitude": [slice(2, 4)]}

        result = lon_index.sel({"longitude": slice(-185, 1)})
        assert result.dim_indexers == {"longitude": [slice(3, 4), slice(0, 3)]}

        result = lon_index.sel({"longitude": slice(-220, -190)})
        assert result.dim_indexers == {"longitude": [slice(3, 4)]}

        result = lon_index.sel({"longitude": slice(190, 220)})
        assert result.dim_indexers == {"longitude": [slice(0, 1)]}

        result = lon_index.sel({"longitude": slice(150, 220)})
        assert result.dim_indexers == {"longitude": [slice(3, 4), slice(0, 1)]}

    def test_longitude_cell_index_global_360(self):
        edges = [0, 90, 180, 270, 360]
        lon_index = LongitudeCellIndex(
            pd.IntervalIndex.from_breaks(edges, closed="left"), "longitude"
        )
        assert lon_index.is_global
        assert len(lon_index) == len(edges) - 1

        result = lon_index.sel({"longitude": slice(90, 220)})
        assert result.dim_indexers == {"longitude": [slice(1, 3)]}

        result = lon_index.sel({"longitude": slice(-90, 0)})
        assert result.dim_indexers == {"longitude": [slice(3, 4), slice(0, 1)]}

        result = lon_index.sel({"longitude": slice(275, 365)})
        assert result.dim_indexers == {"longitude": [slice(3, 4), slice(0, 1)]}

        result = lon_index.sel({"longitude": slice(-30, -10)})
        assert result.dim_indexers == {"longitude": [slice(3, 4)]}

        result = lon_index.sel({"longitude": slice(380, 420)})
        assert result.dim_indexers == {"longitude": [slice(0, 1)]}


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


class TestGridMinimumSpacing:
    """Test dXmin and dYmin calculation for different grid types."""

    def test_rectilinear_uniform_spacing(self):
        """Test dXmin/dYmin for Rectilinear grid with uniform spacing."""
        x = np.linspace(0, 10, 11)  # spacing = 1.0
        y = np.linspace(0, 5, 6)  # spacing = 1.0
        ds = xr.Dataset(
            {
                "data": xr.DataArray(np.zeros((6, 11)), dims=["y", "x"]),
                "x": xr.DataArray(x, dims="x"),
                "y": xr.DataArray(y, dims="y"),
            }
        )
        crs = CRS.from_epsg(4326)
        grid = Rectilinear.from_dataset(ds, crs, "x", "y")
        assert grid.dXmin == 1.0
        assert grid.dYmin == 1.0

    def test_rectilinear_nonuniform_spacing(self):
        """Test dXmin/dYmin for Rectilinear grid with non-uniform spacing."""
        x = np.array([0, 1, 3, 6, 10])  # original spacing = [1, 2, 3, 4]
        y = np.array([0, 0.5, 1.5, 3, 5])  # original spacing = [0.5, 1, 1.5, 2]
        ds = xr.Dataset(
            {
                "data": xr.DataArray(np.zeros((5, 5)), dims=["y", "x"]),
                "x": xr.DataArray(x, dims="x"),
                "y": xr.DataArray(y, dims="y"),
            }
        )
        crs = CRS.from_epsg(4326)
        grid = Rectilinear.from_dataset(ds, crs, "x", "y")

        assert grid.dXmin == pytest.approx(1)
        assert grid.dYmin == pytest.approx(0.5)

    def test_raster_affine_spacing(self):
        """Test dXmin/dYmin for RasterAffine grid."""
        nx, ny = 100, 50
        x_spacing = 0.25
        y_spacing = 0.5
        x = np.arange(nx) * x_spacing
        y = np.arange(ny) * y_spacing
        ds = xr.Dataset(
            {
                "data": xr.DataArray(np.zeros((ny, nx)), dims=["y", "x"]),
                "x": xr.DataArray(x, dims="x"),
                "y": xr.DataArray(y, dims="y"),
            }
        )
        ds = rasterix.assign_index(ds, x_dim="x", y_dim="y")
        crs = CRS.from_epsg(4326)
        grid = RasterAffine.from_dataset(ds, crs, "x", "y")
        assert grid.dXmin == x_spacing
        assert grid.dYmin == y_spacing

    def test_curvilinear_spacing(self):
        """Test dXmin/dYmin for Curvilinear grid with non-uniform spacing."""
        ni, nj = 10, 8
        lon_1d = np.array([0, 0.5, 1.2, 2.1, 3.3, 4.8, 6.5, 8.3, 10.0, 12.0])
        lat_1d = np.array([0, 0.3, 0.8, 1.5, 2.4, 3.5, 4.8, 6.3])
        lon_2d, lat_2d = np.broadcast_arrays(lon_1d[None, :], lat_1d[:, None])

        # Add some small perturbations to make it truly curvilinear (not just rectilinear)
        # This simulates a rotated or distorted grid
        lon_2d = lon_2d + 0.1 * np.sin(lat_2d * np.pi / 6)
        lat_2d = lat_2d + 0.05 * np.cos(lon_2d * np.pi / 12)

        ds = xr.Dataset(
            {
                "data": xr.DataArray(np.zeros((nj, ni)), dims=["j", "i"]),
                "lon": xr.DataArray(
                    lon_2d, dims=["j", "i"], attrs={"standard_name": "longitude"}
                ),
                "lat": xr.DataArray(
                    lat_2d, dims=["j", "i"], attrs={"standard_name": "latitude"}
                ),
            }
        )
        crs = CRS.from_epsg(4326)
        grid = Curvilinear.from_dataset(ds, crs, "lon", "lat")

        # Expected minimum spacings based on the smallest differences
        # lon: smallest diff is 0.5 (between first two points)
        # lat: smallest diff is 0.3 (between first two points)
        # Due to the perturbations, actual values might be slightly different
        assert grid.dXmin == pytest.approx(0.5, rel=0.1)
        assert grid.dYmin == pytest.approx(0.3, rel=0.1)


def _create_test_dataset(
    grid_type: str,
    tms,
    *,
    target_zoom: int,
    array_size: int,
    target_zoom_type: Literal["min", "max"] = "max",
):
    assert array_size > 1, f"array_size must be > 1, got {array_size}"
    if target_zoom_type == "min":
        # For min_zoom tests, create tile-specific coordinates
        tms_bbox = tms.bbox
        center_lon = (tms_bbox.left + tms_bbox.right) / 2
        center_lat = (tms_bbox.bottom + tms_bbox.top) / 2
        tile = tms.tile(center_lon, center_lat, target_zoom)
        tile_bbox = tms.bounds(tile)
        lon_coords = np.linspace(tile_bbox.left, tile_bbox.right, array_size)
        lat_coords = np.linspace(tile_bbox.bottom, tile_bbox.top, array_size)
    else:
        # For max_zoom tests, create resolution-based coordinates
        resolution = tms.tileMatrices[target_zoom].cellSize
        transformer = transformer_from_crs(
            CRS.from_wkt(tms.crs.to_wkt()), CRS.from_epsg(4326), always_xy=True
        )
        _, _, east, north = transformer.transform_bounds(0, 0, resolution, resolution)
        resolution = max(east, north)
        lon_coords = np.linspace(0.0, resolution * array_size, array_size)
        lat_coords = np.linspace(0.0, resolution * array_size, array_size)

    data = np.random.rand(array_size, array_size)
    if grid_type == "rectilinear":
        ds = xr.Dataset(
            {"temp": (["lat", "lon"], data)},
            coords={"lat": lat_coords, "lon": lon_coords},
        )
        grid = Rectilinear.from_dataset(ds, CRS.from_epsg(4326), "lon", "lat")
    elif grid_type == "curvilinear":
        lon, lat = np.meshgrid(lon_coords, lat_coords)
        ds = xr.Dataset(
            {
                "temp": (["y", "x"], data),
                "lon": (["y", "x"], lon),
                "lat": (["y", "x"], lat),
            },
            coords={"x": np.arange(array_size), "y": np.arange(array_size)},
        )
        grid = Curvilinear.from_dataset(ds, CRS.from_epsg(4326), "lon", "lat")
    elif grid_type == "raster_affine":
        pixel_size_x = (
            (lon_coords[-1] - lon_coords[0]) / (array_size - 1) if array_size > 1 else 1.0
        )
        pixel_size_y = (
            (lat_coords[-1] - lat_coords[0]) / (array_size - 1) if array_size > 1 else 1.0
        )
        transform = Affine.translation(lon_coords[0], lat_coords[-1]) * Affine.scale(
            pixel_size_x, -pixel_size_y
        )
        ds = xr.Dataset({"temp": (["y", "x"], data)})
        ds.coords["spatial_ref"] = (
            (),
            4326,
            CRS.from_epsg(4326).to_cf()
            | {"GeoTransform": " ".join(map(str, transform.to_gdal()))},
        )
        grid = RasterAffine.from_dataset(ds, CRS.from_epsg(4326), "x", "y")
    else:
        raise ValueError(f"Unknown grid_type: {grid_type}")
    return ds["temp"], grid


class TestGridZoomMethods:
    """Test get_min_zoom and get_max_zoom methods."""

    @pytest.mark.parametrize(
        "shape, expected",
        (
            ((10, 20), 0),
            ((30000, 15000), 2),
        ),
    )
    def test_get_min_zoom(self, shape, expected):
        ny, nx = shape
        x = np.linspace(-180, 180, nx)
        y = np.linspace(-90, 90, ny)
        data = np.random.rand(*shape)
        ds = xr.Dataset({"temp": (["lat", "lon"], data)}, coords={"lat": y, "lon": x})
        grid = Rectilinear.from_dataset(ds, CRS.from_epsg(4326), "lon", "lat")
        tms = morecantile.tms.get("WebMercatorQuad")
        assert get_min_zoom(grid, tms, ds["temp"]) == expected

    def test_get_max_zoom_basic(self):
        x = np.linspace(-180, 180, 360)
        y = np.linspace(-90, 90, 180)
        data = np.random.rand(180, 360)
        ds = xr.Dataset({"temp": (["lat", "lon"], data)}, coords={"lat": y, "lon": x})
        grid = Rectilinear.from_dataset(ds, CRS.from_epsg(4326), "lon", "lat")
        tms = morecantile.tms.get("WebMercatorQuad")
        max_zoom = get_max_zoom(grid, tms)
        assert isinstance(max_zoom, int)
        assert 0 <= max_zoom <= tms.maxzoom

    @pytest.mark.parametrize("tms_id", morecantile.tms.list())
    def test_min_max_zoom_relationship(self, tms_id):
        x = np.linspace(-180, 180, 100)
        y = np.linspace(-90, 90, 50)
        data = np.random.rand(50, 100)
        ds = xr.Dataset({"temp": (["lat", "lon"], data)}, coords={"lat": y, "lon": x})
        grid = Rectilinear.from_dataset(ds, CRS.from_epsg(4326), "lon", "lat")
        tms = morecantile.tms.get(tms_id)
        min_zoom = get_min_zoom(grid, tms, ds["temp"])
        max_zoom = get_max_zoom(grid, tms)
        assert min_zoom <= max_zoom, (
            f"min_zoom ({min_zoom}) > max_zoom ({max_zoom}) for TMS {tms_id}"
        )

    @pytest.mark.parametrize(
        "tms_id", ["WebMercatorQuad", "WGS1984Quad", "WorldCRS84Quad"]
    )
    @pytest.mark.parametrize("grid_type", ["rectilinear", "curvilinear", "raster_affine"])
    @given(data=st.data())
    @settings(deadline=None)
    def test_max_zoom_matches_dataset_resolution(self, tms_id, grid_type, data):
        """
        Property test:
        Construct a synthetic dataset whose spacing exactly matches tile spacing for zoom level Z.
        Inferred zoom level should be Z.
        """
        tms = morecantile.tms.get(tms_id)
        target_zoom = data.draw(st.integers(min_value=tms.minzoom, max_value=tms.maxzoom))
        _, grid = _create_test_dataset(
            grid_type, tms, target_zoom=target_zoom, array_size=100
        )
        calculated_zoom = get_max_zoom(grid, tms)
        assert calculated_zoom == target_zoom, (
            f"Expected {target_zoom}, got {calculated_zoom} for {tms_id} {grid_type}"
        )

    @pytest.mark.parametrize(
        "tms_id", ["WebMercatorQuad", "WGS1984Quad", "WorldCRS84Quad"]
    )
    @pytest.mark.parametrize("grid_type", ["rectilinear", "curvilinear", "raster_affine"])
    @given(data=st.data())
    @settings(deadline=None)
    def test_min_zoom_matches_renderable_size_limit(self, tms_id, grid_type, data):
        """
        Property test:
        Construct a synthetic dataset that is just slightly too big for a chosen zoom level Z.
        Inferred zoom level should be Z+1
        """
        tms = morecantile.tms.get(tms_id)
        target_zoom = data.draw(
            st.integers(min_value=max(tms.minzoom, 2), max_value=min(tms.maxzoom, 8))
        )
        assume(tms.minzoom <= target_zoom <= tms.maxzoom)

        pixels_per_tile = 1001
        da, grid = _create_test_dataset(
            grid_type,
            tms,
            target_zoom=target_zoom,
            array_size=pixels_per_tile,
            target_zoom_type="min",
        )
        with config.set(
            {"max_renderable_size": da.dtype.itemsize * (pixels_per_tile - 1) ** 2}
        ):
            actual = get_min_zoom(grid, tms, da)
        expected = target_zoom + 1
        assert expected == actual, (
            f"Expected {expected}, got {actual} for {tms_id} {grid_type} at zoom {target_zoom}"
        )


def test_grid_detection_thread_lock():
    """Test that concurrent calls to guess_grid_system only call constructor once."""
    ds = IFS.create()
    # avoid clashing with other tests
    ds.attrs["_xpublish_id"] = "ifs_thread_lock_test"

    original_from_dataset = Rectilinear.from_dataset

    with patch.object(
        Rectilinear, "from_dataset", wraps=original_from_dataset
    ) as mock_from_dataset:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(guess_grid_system, ds, "foo") for _ in range(4)]
            results = [future.result() for future in futures]

    assert all(isinstance(r, Rectilinear) for r in results)
    # Constructor should only be called once due to thread lock and caching
    assert mock_from_dataset.call_count == 1


def test_qhull_error():
    ds = REDGAUSS_N320.create()
    # make sure it works
    guess_grid_system(ds, "foo")

    ds.attrs["_xpublish_id"] = "n320_with_nans"
    ds["latitude"][-1] = np.nan
    with pytest.raises(ValueError):
        guess_grid_system(ds, "foo")

    ds.attrs["_xpublish_id"] = "n320_with_zeros"
    ds["latitude"][:] = 0
    with pytest.raises(ValueError):
        guess_grid_system(ds, "foo")
