import hypothesis.strategies as st
import morecantile
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import DrawFn
from morecantile import Tile, TileMatrixSet

import xarray as xr
from tests import create_query_params
from xpublish_tiles.lib import check_transparent_pixels
from xpublish_tiles.pipeline import pipeline
from xpublish_tiles.testing.datasets import Dim, uniform_grid
from xpublish_tiles.testing.lib import visualize_tile


@st.composite
def global_datasets(draw: DrawFn, allow_categorical: bool = True) -> xr.Dataset:
    """Strategy that generates global datasets using uniform_grid with random parameters."""
    # Generate dimensions between 100 and 1000 to ensure sufficient coverage
    # Smaller datasets may have gaps when projected
    nlat = draw(st.integers(min_value=100, max_value=1000))
    nlon = draw(st.integers(min_value=100, max_value=1000))

    # Generate latitude ordering
    lat_ascending = draw(st.booleans())
    lats = np.linspace(-90, 90, nlat)
    if not lat_ascending:
        lats = lats[::-1]

    # Generate longitude ordering
    lon_0_360 = draw(st.booleans())
    if lon_0_360:
        lons = np.linspace(0, 360, nlon)
    else:
        lons = np.linspace(-180, 180, nlon)

    # Use full size as chunk size (single chunk)
    dims = (
        Dim(
            name="latitude",
            size=nlat,
            chunk_size=nlat,
            data=lats,
            attrs={"units": "degrees_north", "axis": "Y"},
        ),
        Dim(
            name="longitude",
            size=nlon,
            chunk_size=nlon,
            data=lons,
            attrs={"units": "degrees_east", "axis": "X"},
        ),
    )

    is_categorical = allow_categorical and draw(st.booleans())

    if is_categorical:
        # Generate categorical data with flag_values
        num_categories = draw(st.integers(min_value=2, max_value=12))
        flag_values = list(range(num_categories))

        flag_meanings = " ".join([f"category_{i}" for i in flag_values])

        attrs = {
            "long_name": "Test categorical data",
            "flag_meanings": flag_meanings,
            "flag_values": flag_values,
        }
        dtype = np.uint8
    else:
        attrs = {
            "long_name": "Test continuous data",
            "valid_min": -1,
            "valid_max": 1,
        }
        dtype = np.float32

    ds = uniform_grid(dims=dims, dtype=dtype, attrs=attrs)
    return ds


@st.composite
def tile_matrix_sets(draw: DrawFn) -> str:
    """Strategy that returns standard TileMatrixSet names from morecantile."""
    tms_name = draw(st.sampled_from(["WebMercatorQuad", "WorldCRS84Quad"]))
    return tms_name


@st.composite
def tiles(
    draw: DrawFn,
    tile_matrix_sets: st.SearchStrategy[str] = tile_matrix_sets(),  # noqa: B008
) -> Tile:
    """Strategy that returns morecantile.Tile objects based on a TileMatrixSet."""
    tms_name: str = draw(tile_matrix_sets)
    tms: TileMatrixSet = morecantile.tms.get(tms_name)
    zoom = draw(st.integers(min_value=0, max_value=len(tms.tileMatrices) - 1))
    minmax = tms.minmax(zoom)
    x = draw(st.integers(min_value=minmax["x"]["min"], max_value=minmax["x"]["max"]))
    y = draw(st.integers(min_value=minmax["y"]["min"], max_value=minmax["y"]["max"]))
    return Tile(x=x, y=y, z=zoom)


@st.composite
def tile_and_tms(
    draw: DrawFn,
    *,
    tile_matrix_sets: st.SearchStrategy[str] = tile_matrix_sets(),  # noqa: B008
) -> tuple[Tile, TileMatrixSet]:
    """Strategy that returns a tile and its corresponding TileMatrixSet."""
    tms_name: str = draw(tile_matrix_sets)
    tms: TileMatrixSet = morecantile.tms.get(tms_name)
    zoom = draw(st.integers(min_value=0, max_value=len(tms.tileMatrices) - 1))
    minmax = tms.minmax(zoom)
    x = draw(st.integers(min_value=minmax["x"]["min"], max_value=minmax["x"]["max"]))
    y = draw(st.integers(min_value=minmax["y"]["min"], max_value=minmax["y"]["max"]))
    tile = Tile(x=x, y=y, z=zoom)
    return tile, tms


@pytest.mark.asyncio
@settings(deadline=None)
@given(tile_tms=tile_and_tms(), ds=global_datasets(allow_categorical=False))
async def test_property_global_render_no_transparent_tile(
    tile_tms: tuple[Tile, TileMatrixSet],
    ds: xr.Dataset,
    pytestconfig,
):
    """Property test that global datasets should never produce transparent pixels."""
    tile, tms = tile_tms
    query_params = create_query_params(tile, tms)
    result = await pipeline(ds, query_params)
    transparent_percent = check_transparent_pixels(result.getvalue())
    if pytestconfig.getoption("--visualize"):
        visualize_tile(result, tile)
    assert (
        transparent_percent == 0
    ), f"Found {transparent_percent:.1f}% transparent pixels in tile {tile}"
