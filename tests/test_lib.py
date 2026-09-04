import asyncio
import math
from typing import cast
from unittest.mock import patch

import matplotlib as mpl
import morecantile
import numpy as np
import pandas as pd
import pyproj
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst
from pyproj.aoi import BBox

import xarray as xr
from tests import NUMERIC_DTYPES
from xpublish_tiles.config import config
from xpublish_tiles.grids import guess_grid_system
from xpublish_tiles.lib import (
    InvalidCoordinateValues,
    _chunk_aligned_count,
    _chunk_sizes,
    adversarial_slicers,
    anynotfinite,
    apply_default_pad,
    apply_range_colors,
    check_data_is_renderable_size,
    coarsen_mean_pad,
    decompressed_size_bytes,
    normalize_slicers,
    timedelta_to_iso8601,
    transform_chunk,
    transform_coordinates,
)
from xpublish_tiles.pipeline import _parse_timedelta, fix_coordinate_discontinuities
from xpublish_tiles.projections import (
    epsg4326to3857,
    transformer_from_crs,
)
from xpublish_tiles.testing.datasets import IFS, PARA
from xpublish_tiles.tiles_lib import get_min_zoom, grid_overlaps_tms


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


def test_transform_chunk_inplace():
    """Test that transform_chunk inplace option works correctly."""
    # Use EPSG:3035 to EPSG:4326 transformation
    transformer = pyproj.Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

    # Create a simple 4x4 coordinate grid with EPSG:3035 coordinates
    x_grid = np.array(
        [
            [2635840.0, 2735840.0, 2835840.0, 2935840.0],
            [2635840.0, 2735840.0, 2835840.0, 2935840.0],
            [2635840.0, 2735840.0, 2835840.0, 2935840.0],
            [2635840.0, 2735840.0, 2835840.0, 2935840.0],
        ],
        dtype=np.float64,
        order="C",
    )
    y_grid = np.array(
        [
            [5415940.0, 5415940.0, 5415940.0, 5415940.0],
            [5315940.0, 5315940.0, 5315940.0, 5315940.0],
            [5215940.0, 5215940.0, 5215940.0, 5215940.0],
            [5115940.0, 5115940.0, 5115940.0, 5115940.0],
        ],
        dtype=np.float64,
        order="C",
    )

    # Save original values
    x_original = x_grid.copy()
    y_original = y_grid.copy()

    # Test inplace=False - should not modify input arrays
    x_out_noninplace = np.empty_like(x_grid)
    y_out_noninplace = np.empty_like(y_grid)
    slices = (slice(0, 4), slice(0, 4))

    transform_chunk(
        x_grid,
        y_grid,
        slices,
        transformer,
        x_out_noninplace,
        y_out_noninplace,
        inplace=False,
    )

    # Original arrays should be unchanged
    np.testing.assert_array_equal(x_grid, x_original)
    np.testing.assert_array_equal(y_grid, y_original)

    # Test inplace=True - should modify input arrays
    x_grid_inplace = x_original.copy(order="C")
    y_grid_inplace = y_original.copy(order="C")

    # For inplace, the output arrays are the same as the input arrays
    transform_chunk(
        x_grid_inplace,
        y_grid_inplace,
        slices,
        transformer,
        x_grid_inplace,
        y_grid_inplace,
        inplace=True,
    )

    # Input arrays should be modified
    assert not np.array_equal(x_grid_inplace, x_original)
    assert not np.array_equal(y_grid_inplace, y_original)

    # Both methods should produce the same transformed results
    np.testing.assert_allclose(x_grid_inplace, x_out_noninplace)
    np.testing.assert_allclose(y_grid_inplace, y_out_noninplace)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_transform_coordinates_with_dtypes(dtype):
    """Test transform_coordinates with different coordinate dtypes.

    This tests the full pipeline which converts coordinates to float64 for
    efficient transformations with pyproj's inplace optimization.
    """
    # Use EPSG:4326 to EPSG:3857 transformation
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    with config.set(transform_chunk_size=50):
        ny, nx = 300, 400
        lon_1d = np.linspace(-10, 10, nx, dtype=dtype)
        lat_1d = np.linspace(40, 60, ny, dtype=dtype)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

        # Add some curvature
        lon_2d = lon_2d + 0.1 * np.sin(
            2 * np.pi * np.arange(ny, dtype=dtype)[:, None] / ny
        )
        lat_2d = lat_2d + 0.1 * np.cos(
            2 * np.pi * np.arange(nx, dtype=dtype)[None, :] / nx
        )

        # Create a DataArray with coordinates of the specified dtype
        data = np.zeros((ny, nx))
        da = xr.DataArray(
            data,
            coords={
                "lon": (("y", "x"), lon_2d.astype(dtype)),
                "lat": (("y", "x"), lat_2d.astype(dtype)),
            },
            dims=["y", "x"],
        )

        # Verify input dtype
        assert da.coords["lon"].dtype == dtype
        assert da.coords["lat"].dtype == dtype

        # Transform coordinates
        with patch(
            "xpublish_tiles.lib.transform_chunk", wraps=transform_chunk
        ) as mock_transform_chunk:
            x_transformed, y_transformed = asyncio.run(
                transform_coordinates(da, "lon", "lat", transformer)
            )

            # Verify blocked transformation was used
            assert mock_transform_chunk.call_count > 0, (
                "transform_chunk should be called for large 2D grids"
            )

        # Verify output shape
        assert x_transformed.shape == (ny, nx)
        assert y_transformed.shape == (ny, nx)

        # Verify the data was transformed (should be in EPSG:3857 meter units now)
        # Original is in degrees (-10 to 10), transformed should be much larger (meters)
        assert np.abs(x_transformed.data).max() > 1e6, "Should be transformed to meters"
        assert np.abs(y_transformed.data).max() > 4e6, "Should be transformed to meters"

        # transform_coordinates converts to float64 via np.asarray for efficient transforms
        assert x_transformed.data.dtype == np.float64
        assert y_transformed.data.dtype == np.float64


def test_transform_coordinates_large_broadcast():
    """Test transform_coordinates with 1D inputs that trigger blocked transformation."""
    # Use EPSG:3035 (ETRS89-extended / LAEA Europe) to EPSG:4326
    # This avoids the fast path for 4326->3857 with 1D coords
    transformer = pyproj.Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

    # Create 1D coordinates large enough to trigger blocked transformation
    # Using 2000x2000 = 4,000,000 elements which is larger than chunk_size product
    # EPSG:3035 uses meters, typical European extent
    x_values = np.linspace(2635840.0, 3874240.0, 500)
    y_values = np.linspace(5415940.0, 2042740.0, 500)

    data = np.zeros((x_values.size, y_values.size))
    da = xr.DataArray(
        data,
        coords={
            "x": ("x", x_values),
            "y": ("y", y_values),
        },
        dims=["y", "x"],
    )

    with patch(
        "xpublish_tiles.lib.transform_chunk", wraps=transform_chunk
    ) as mock_transform_chunk:
        with config.set(transform_chunk_size=50):
            x_transformed, y_transformed = asyncio.run(
                transform_coordinates(da, "x", "y", transformer)
            )

        assert mock_transform_chunk.call_count > 0

    assert x_transformed.shape == da.shape
    assert y_transformed.shape == da.shape

    # Verify the coordinates were transformed (not equal to original 1D inputs)
    # Since x_values is 1D and x_transformed is 2D, we compare the first row
    assert not np.array_equal(x_transformed.data[0, :], x_values)
    assert not np.array_equal(y_transformed.data[:, 0], y_values)


def _get_over_color(cmap):
    """Get the over color from a colormap (for testing)."""
    return cmap._rgba_over  # type: ignore[attr-defined]


def _get_under_color(cmap):
    """Get the under color from a colormap (for testing)."""
    return cmap._rgba_under  # type: ignore[attr-defined]


class TestApplyRangeColors:
    """Tests for the apply_range_colors function."""

    def test_apply_range_colors_sets_over_color(self):
        """Test that apply_range_colors sets the over color correctly."""
        cmap = mpl.colormaps.get_cmap("viridis")
        result = apply_range_colors(cmap, "#ff0000", None)

        # Over color should be set to red
        assert _get_over_color(result) == pytest.approx((1.0, 0.0, 0.0, 1.0), rel=0.01)
        # Under color should be unchanged from original
        assert _get_under_color(result) == _get_under_color(cmap)

    def test_apply_range_colors_sets_under_color(self):
        """Test that apply_range_colors sets the under color correctly."""
        cmap = mpl.colormaps.get_cmap("viridis")
        result = apply_range_colors(cmap, None, "#0000ff")

        # Under color should be set to blue
        assert _get_under_color(result) == pytest.approx((0.0, 0.0, 1.0, 1.0), rel=0.01)
        # Over color should be unchanged from original
        assert _get_over_color(result) == _get_over_color(cmap)

    def test_apply_range_colors_sets_both_colors(self):
        """Test that apply_range_colors sets both over and under colors."""
        cmap = mpl.colormaps.get_cmap("viridis")
        result = apply_range_colors(cmap, "#ff0000", "#0000ff")

        assert _get_over_color(result) == pytest.approx((1.0, 0.0, 0.0, 1.0), rel=0.01)
        assert _get_under_color(result) == pytest.approx((0.0, 0.0, 1.0, 1.0), rel=0.01)

    def test_apply_range_colors_transparent(self):
        """Test that apply_range_colors handles transparent correctly."""
        cmap = mpl.colormaps.get_cmap("viridis")
        result = apply_range_colors(cmap, "transparent", "transparent")

        assert _get_over_color(result) == (0, 0, 0, 0)
        assert _get_under_color(result) == (0, 0, 0, 0)

    def test_apply_range_colors_extend_leaves_unchanged(self):
        """Test that apply_range_colors with 'extend' leaves colors unchanged."""
        cmap = mpl.colormaps.get_cmap("viridis")
        original_over = _get_over_color(cmap)
        original_under = _get_under_color(cmap)

        result = apply_range_colors(cmap, "extend", "extend")

        assert _get_over_color(result) == original_over
        assert _get_under_color(result) == original_under

    def test_apply_range_colors_none_leaves_unchanged(self):
        """Test that apply_range_colors with None leaves colors unchanged."""
        cmap = mpl.colormaps.get_cmap("viridis")
        original_over = _get_over_color(cmap)
        original_under = _get_under_color(cmap)

        result = apply_range_colors(cmap, None, None)

        assert _get_over_color(result) == original_over
        assert _get_under_color(result) == original_under

    def test_apply_range_colors_does_not_modify_original(self):
        """Test that apply_range_colors returns a copy and doesn't modify original."""
        cmap = mpl.colormaps.get_cmap("viridis")
        original_over = _get_over_color(cmap)
        original_under = _get_under_color(cmap)

        result = apply_range_colors(cmap, "#ff0000", "#0000ff")

        # Original should be unchanged
        assert _get_over_color(cmap) == original_over
        assert _get_under_color(cmap) == original_under
        # Result should be different
        assert result is not cmap

    def test_apply_range_colors_named_colors(self):
        """Test that apply_range_colors handles named colors correctly."""
        cmap = mpl.colormaps.get_cmap("viridis")
        result = apply_range_colors(cmap, "red", "blue")

        assert _get_over_color(result) == pytest.approx((1.0, 0.0, 0.0, 1.0), rel=0.01)
        assert _get_under_color(result) == pytest.approx((0.0, 0.0, 1.0, 1.0), rel=0.01)


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_coarsen_mean_pad_dtypes(dtype):
    """Test that coarsen_mean_pad handles all numeric dtypes."""
    arr = np.arange(12, dtype=dtype).reshape(4, 3)
    da = xr.DataArray(arr, dims=["y", "x"])

    result = coarsen_mean_pad(da, {"y": 2, "x": 2})

    # Row 0-1, Col 0-1: mean(0,1,3,4) = 2.0
    # Row 0-1, Col 2:   mean(2,5) = 3.5
    # Row 2-3, Col 0-1: mean(6,7,9,10) = 8.0
    # Row 2-3, Col 2:   mean(8,11) = 9.5
    expected = np.array([[2.0, 3.5], [8.0, 9.5]])
    np.testing.assert_allclose(result.values, expected)
    assert result.dims == ("y", "x")
    assert result.dtype == np.float64


@pytest.mark.parametrize(
    "value, expected",
    [
        ("0ns", "PT0S"),
        ("15D", "P15D"),
        ("353h", "P14DT17H"),
        ("100h", "P4DT4H"),
        ("90min", "PT1H30M"),
        ("1.5s", "PT1.5S"),
        ("1ns", "PT0.000000001S"),
        ("-6h", "-PT6H"),
    ],
)
def test_timedelta_to_iso8601(value, expected):
    delta = cast(pd.Timedelta, pd.Timedelta(value))
    assert timedelta_to_iso8601(delta.to_timedelta64()) == expected
    assert timedelta_to_iso8601(delta) == expected
    # Whatever we advertise must round-trip through the selector parser.
    assert pd.Timedelta(_parse_timedelta(expected)) == delta


@pytest.mark.parametrize(
    "start, stop, chunk, expected",
    [
        (0, 100, 100, 100),  # aligned
        (0, 101, 100, 200),  # spills into second chunk
        (50, 150, 100, 200),  # straddles a boundary
        (99, 100, 100, 100),  # single element, whole chunk
        (10, 10, 100, 0),  # empty
    ],
)
def test_chunk_aligned_count(start, stop, chunk, expected):
    assert _chunk_aligned_count(start, stop, chunk) == expected


def test_chunk_sizes_from_preferred_chunks():
    da = xr.DataArray(np.zeros((4, 5)), dims=("y", "x"))
    da.encoding["preferred_chunks"] = {"y": 2, "x": 3}
    assert _chunk_sizes(da) == {"y": 2, "x": 3}


def test_chunk_sizes_ignores_positional_encoding():
    # positional chunks are indexed by the array's original dims, so they
    # cannot be zipped against da.dims once a dim has been isel'd away
    da = xr.DataArray(np.zeros((4, 5)), dims=("y", "x"))
    da.encoding["chunks"] = (2, 3)
    assert _chunk_sizes(da) == {}


def test_chunk_sizes_from_zarr_array(tmp_path):
    store = tmp_path / "s.zarr"
    xr.Dataset(
        {"foo": (("time", "y", "x"), np.zeros((20, 40, 60), dtype=np.float32))}
    ).to_zarr(
        store,
        zarr_format=3,
        encoding={"foo": {"chunks": (10, 10, 10), "shards": (20, 20, 30)}},
    )
    ds = xr.open_zarr(store, chunks=None)

    # poison the encoding so only the zarr array can give the right answer
    ds.foo.encoding["preferred_chunks"] = {"time": 1, "y": 1, "x": 1}

    # inner chunk, not the shard: the shard is the IO unit, not the
    # decompression unit
    assert _chunk_sizes(ds.foo) == {"time": 10, "y": 10, "x": 10}

    # once indexed, zarr's positional chunks no longer line up with da.dims, so
    # we refuse the array and fall back (hence capture before the collapse, in
    # ValidatedArray.chunks)
    assert _chunk_sizes(ds.foo.isel(time=-1)) == {"time": 1, "y": 1, "x": 1}


def test_decompressed_size_bytes():
    ds = IFS.create()
    grid = guess_grid_system(ds, "foo")
    da = ds["foo"].isel(time=-1, step=-1)
    ny, nx = da.sizes[grid.Ydim], da.sizes[grid.Xdim]
    slicers = normalize_slicers(
        {grid.Ydim: [slice(0, 1)], grid.Xdim: [slice(0, 1)]}, dict(da.sizes)
    )

    da.encoding.clear()
    assert decompressed_size_bytes(slicers, da, grid) == da.dtype.itemsize

    chunks = {grid.Ydim: 10, grid.Xdim: 10}
    assert (
        decompressed_size_bytes(slicers, da, grid, chunks=chunks)
        == 100 * da.dtype.itemsize
    )

    # a dim already collapsed by isel still costs its full chunk extent
    chunks = {grid.Ydim: 10, grid.Xdim: 10, "time": 648}
    assert (
        decompressed_size_bytes(slicers, da, grid, chunks=chunks)
        == 648 * 100 * da.dtype.itemsize
    )

    # single chunk spanning the array: any tile costs the whole thing
    chunks = {grid.Ydim: ny, grid.Xdim: nx}
    assert (
        decompressed_size_bytes(slicers, da, grid, chunks=chunks)
        == ny * nx * da.dtype.itemsize
    )


@pytest.mark.parametrize(
    "tms_id, tropical",
    [
        # a conic or polar TMS reports a near-global tms.bbox, so only its CRS's
        # area of use rules it out over the tropics
        ("WebMercatorQuad", True),
        ("WorldCRS84Quad", True),
        ("CanadianNAD83_LCC", False),
        ("UPSArcticWGS84Quad", False),
        ("LINZAntarticaMapTilegrid", False),
        ("UTM31WGS84Quad", False),
    ],
)
def test_grid_overlaps_tms(tms_id, tropical):
    """A TMS whose projection isn't defined over the data cannot tile it."""
    tms = morecantile.tms.get(tms_id)
    assert grid_overlaps_tms(guess_grid_system(PARA.create(), "foo"), tms) is tropical
    # a global grid reaches into every TMS's area of use
    assert grid_overlaps_tms(guess_grid_system(IFS.create(), "foo"), tms)


def test_adversarial_slicers():
    ds = IFS.create()
    grid = guess_grid_system(ds, "foo")
    da = ds["foo"].isel(time=-1, step=-1)
    ny, nx = da.sizes[grid.Ydim], da.sizes[grid.Xdim]
    da.encoding.clear()
    chunks = {grid.Ydim: 10, grid.Xdim: 10}

    def bytes_for(y, x, *, worst_case):
        slicers = normalize_slicers({grid.Ydim: [y], grid.Xdim: [x]}, dict(da.sizes))
        if worst_case:
            slicers = adversarial_slicers(slicers, da, grid, chunks=chunks)
        return decompressed_size_bytes(slicers, da, grid, chunks=chunks)

    # a chunk-aligned span costs 1 chunk where it sits, but 2 at the worst offset
    assert (
        bytes_for(slice(0, 10), slice(0, 10), worst_case=False) == 100 * da.dtype.itemsize
    )
    assert (
        bytes_for(slice(0, 10), slice(0, 10), worst_case=True) == 400 * da.dtype.itemsize
    )
    # the same span shifted off the boundary already pays that cost
    assert (
        bytes_for(slice(5, 15), slice(5, 15), worst_case=False) == 400 * da.dtype.itemsize
    )
    assert (
        bytes_for(slice(5, 15), slice(5, 15), worst_case=True) == 400 * da.dtype.itemsize
    )
    # never more than reading the whole dim
    assert (
        bytes_for(slice(0, ny), slice(0, nx), worst_case=True)
        == math.ceil(ny / 10) * 10 * math.ceil(nx / 10) * 10 * da.dtype.itemsize
    )
    # worst case is never below what the selection actually costs
    for y, x in [(slice(0, 3), slice(7, 44)), (slice(19, 20), slice(0, 100))]:
        assert bytes_for(y, x, worst_case=True) >= bytes_for(y, x, worst_case=False)


def test_min_zoom_accounts_for_chunks():
    ds = IFS.create()
    ds.attrs["_xpublish_id"] = "chunk_minzoom"
    grid = guess_grid_system(ds, "foo")
    tms = morecantile.tms.get("WebMercatorQuad")
    da = ds["foo"].isel(time=-1, step=-1)

    with config.set({"max_renderable_size": 50_000 * 8}):
        da.encoding["preferred_chunks"] = dict.fromkeys(da.dims, 1)
        fine = get_min_zoom(grid=grid, tms=tms, da=da, style="raster")
        da.encoding["preferred_chunks"] = dict.fromkeys(da.dims, 64)
        coarse = get_min_zoom(grid=grid, tms=tms, da=da, style="raster")
    assert coarse > fine


def test_min_zoom_covers_interior_tiles():
    """minzoom must hold for every tile at that zoom.

    Chunk-aligned cost depends on where a tile falls, and the tiles the search
    samples sit on the grid boundary, where slicers get clipped. Interior tiles
    straddle more chunks and cost up to 4x more.
    """
    ds = PARA.create()
    grid = guess_grid_system(ds, "foo")
    da = ds["foo"]
    da.encoding["preferred_chunks"] = {"x": 250, "y": 250, "time": 1}
    tms = morecantile.tms.get("WebMercatorQuad")
    tms_crs = pyproj.CRS.from_wkt(tms.crs.to_wkt())
    alternate = grid.pick_alternate_grid(tms_crs, coarsen_factors={})
    transformer = transformer_from_crs(tms_crs, grid.crs)

    def num_over_limit(zoom: int) -> int:
        tiles = tms.tiles(
            grid.bbox.west, grid.bbox.south, grid.bbox.east, grid.bbox.north, [zoom]
        )
        over = 0
        for tile in tiles:
            bounds = tms.xy_bounds(tile)
            west, south, east, north = transformer.transform_bounds(
                bounds.left, bounds.bottom, bounds.right, bounds.top
            )
            slicers = grid.sel(bbox=BBox(west=west, south=south, east=east, north=north))
            slicers = normalize_slicers(
                apply_default_pad(slicers, da, grid), dict(da.sizes)
            )
            over += not check_data_is_renderable_size(
                slicers, da, grid, alternate, style="raster"
            )
        return over

    with config.set({"max_renderable_size": 2_000_000}):
        minzoom = get_min_zoom(grid=grid, tms=tms, da=da, style="raster")
        assert num_over_limit(minzoom) == 0
        # and no higher than it has to be
        assert num_over_limit(minzoom - 1) > 0


@pytest.mark.parametrize(
    "array, expected",
    [
        (np.linspace(-180.0, 180.0, 32).reshape(4, 8), False),
        (np.array([[1.0, np.nan], [3.0, 4.0]]), True),
        (np.array([[1.0, np.inf], [3.0, 4.0]]), True),
        (np.array([[1.0, -np.inf], [3.0, 4.0]]), True),
        # Last element: the short-circuit must not stop early and miss it.
        (np.array([[1.0, 2.0], [3.0, np.nan]]), True),
        (np.array([], dtype=np.float64), False),
        (np.zeros((3, 3), dtype=np.float32), False),
        # Integers cannot hold NaN/inf — must not be rejected.
        (np.arange(12).reshape(3, 4), False),
    ],
)
def test_anynotfinite(array, expected):
    assert anynotfinite(array) is expected


@pytest.fixture
def webmerc_transformer() -> pyproj.Transformer:
    return pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


@pytest.mark.parametrize("axis", [None, 1])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_fix_coordinate_discontinuities_rejects_nonfinite(webmerc_transformer, axis, bad):
    """``axis=None`` hangs in ``unwrap_phase``; with an axis ``round(nan)`` raises.

    Reached by HYCOM/RTOFS: fold-row lons past 360° map to ``inf`` in Web Mercator.
    """
    bbox = BBox(west=-180, south=-85, east=180, north=85)
    poisoned = np.array([[0.0, 1.0e7], [2.0e7, bad]])

    with pytest.raises(InvalidCoordinateValues, match="NaN/inf"):
        fix_coordinate_discontinuities(
            poisoned, webmerc_transformer, bbox=bbox, axis=axis
        )


@pytest.mark.parametrize("axis", [None, 1])
def test_fix_coordinate_discontinuities_allows_finite(webmerc_transformer, axis):
    """The guard must not reject ordinary all-finite coordinates."""
    bbox = BBox(west=-180, south=-85, east=180, north=85)
    coords = np.array([[0.0, 1.0e7], [2.0e7, -1.0e7]])

    result = fix_coordinate_discontinuities(
        coords, webmerc_transformer, bbox=bbox, axis=axis
    )
    assert np.isfinite(result).all()
    assert result.shape == coords.shape
