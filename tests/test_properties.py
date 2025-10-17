import io
import uuid

import hypothesis.strategies as st
import morecantile
import numpy as np
import pytest
from hypothesis import (
    assume,
    given,
    reproduce_failure,  # noqa: F401
    settings,
)
from hypothesis.strategies import DrawFn
from morecantile import Tile, TileMatrixSet
from PIL import Image

import xarray as xr
from tests import create_query_params
from xpublish_tiles import config
from xpublish_tiles.lib import TileTooBigError, check_transparent_pixels
from xpublish_tiles.pipeline import pipeline
from xpublish_tiles.testing.datasets import (
    EU3035_HIRES,
    HRRR,
    HRRR_MULTIPLE,
    REDGAUSS_N320,
    Dim,
    uniform_grid,
)
from xpublish_tiles.testing.lib import (
    compare_image_buffers_with_debug,
    visualize_tile,
)


@st.composite
def global_datasets(
    draw: DrawFn,
    allow_decreasing_lat: bool = True,
    allow_categorical: bool = True,
    maxsize: int = 2000,
    perturb: bool = True,
) -> xr.Dataset:
    """Strategy that generates global datasets using uniform_grid with random parameters.

    Parameters
    ----------
    allow_decreasing_lat: bool
    allow_categorical: bool
    maxsize: int
        Max size of either dimension
    perturb: bool
        Whether to add tiny perturbations to the bounds of each axis, for robustness testing
    """
    # Generate dimensions between 100 and 1000 to ensure sufficient coverage
    # Smaller datasets may have gaps when projected
    # Prioritize sizes that exercise coarsening
    size_st = st.one_of(
        st.sampled_from([i for i in [256, 512, 1024, 2048] if i < maxsize]),
        st.integers(min_value=100, max_value=maxsize),
    )
    nlat = draw(size_st)
    nlon = draw(size_st)

    # Generate latitude ordering
    lat_ascending = not allow_decreasing_lat or draw(st.booleans())
    delta_lat = 0 if not perturb else draw(st.floats(-1e-3, 1e-3))
    delta_lon = 0 if not perturb else draw(st.floats(-1e-3, 1e-3))
    lats = np.linspace(-90 - delta_lat, 90 + delta_lat, nlat)
    if not lat_ascending:
        lats = lats[::-1]

    # Generate longitude ordering
    lon_0_360 = draw(st.booleans())
    if lon_0_360:
        lons = np.linspace(0 - delta_lon, 360 + delta_lon, nlon, endpoint=False)
    else:
        lons = np.linspace(-180 - delta_lon, 180 + delta_lon, nlon, endpoint=False)

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
def global_unstructured_datasets(draw: DrawFn) -> xr.Dataset:
    """Strategy that returns global unstructured grid datasets.

    Currently returns REDGAUSS_N320 (Reduced Gaussian Grid N320).
    """
    # copy to avoid interfering with other tests!!!
    ds = REDGAUSS_N320.create().copy(deep=True)

    # this is yuck; but things are too slow without caching the grid object
    attr = ds.attrs["_xpublish_id"] + "_proptest"

    # Rescale latitude to full -90 to 90 range, this means we can keep the present property test
    lat = ds.latitude.values
    lat_min, lat_max = lat.min(), lat.max()
    lat_scaled = -90 + (lat - lat_min) / (lat_max - lat_min) * 180
    # Occasionally reverse the latitude vector
    if draw(st.booleans()):
        lat_scaled = lat_scaled[::-1]
        attr += "_reversed_lat"

    ds = ds.assign_coords(latitude=("point", lat_scaled))

    # Occasionally convert longitude from 0-360 to -180-180 convention
    if draw(st.booleans()):
        lon = ds.longitude.values
        lon_converted = np.where(lon > 180, lon - 360, lon)
        ds = ds.assign_coords(longitude=("point", lon_converted))
        attr += "_converted_lon"

    ds.attrs["_xpublish_id"] = attr
    return ds


# Combine both regular and unstructured global datasets
all_global_datasets = st.one_of(
    global_datasets(allow_categorical=False), global_unstructured_datasets()
)


@st.composite
def tile_matrix_sets(draw: DrawFn) -> str:
    """Strategy that returns standard TileMatrixSet names from morecantile."""
    tms_name = draw(st.sampled_from(["WebMercatorQuad", "WorldCRS84Quad"]))
    return tms_name


@st.composite
def tiles(
    draw: DrawFn,
    tile_matrix_sets: st.SearchStrategy[str] = tile_matrix_sets(),
) -> Tile:
    """Strategy that returns morecantile.Tile objects based on a TileMatrixSet."""
    tms_name: str = draw(tile_matrix_sets)
    tms: TileMatrixSet = morecantile.tms.get(tms_name)
    # Sample uniformly from available zoom levels
    zoom_levels = list(range(len(tms.tileMatrices)))
    zoom = draw(st.sampled_from(zoom_levels))
    minmax = tms.minmax(zoom)
    x = draw(st.integers(min_value=minmax["x"]["min"], max_value=minmax["x"]["max"]))
    y = draw(st.integers(min_value=minmax["y"]["min"], max_value=minmax["y"]["max"]))
    return Tile(x=x, y=y, z=zoom)


@st.composite
def tile_and_tms(
    draw: DrawFn,
    *,
    tile_matrix_sets: st.SearchStrategy[str] = tile_matrix_sets(),
    bbox=None,
) -> tuple[Tile, TileMatrixSet]:
    """Strategy that returns a tile and its corresponding TileMatrixSet.

    Args:
        tile_matrix_sets: Strategy for selecting TileMatrixSet names
        bbox: Optional bounding box to constrain tiles to overlap with this area
    """
    tms_name: str = draw(tile_matrix_sets)
    tms: TileMatrixSet = morecantile.tms.get(tms_name)
    # Sample uniformly from available zoom levels
    zoom_levels = list(range(len(tms.tileMatrices)))
    zoom = draw(st.sampled_from(zoom_levels))

    if bbox is not None:
        # Get the tiles at the four corners of the bounding box to define the range
        # This is much more efficient than listing all tiles at high zoom levels
        try:
            # Get tiles for the four corners
            sw_tile = tms.tile(bbox.west, bbox.south, zoom)  # Southwest corner
            se_tile = tms.tile(bbox.east, bbox.south, zoom)  # Southeast corner
            nw_tile = tms.tile(bbox.west, bbox.north, zoom)  # Northwest corner
            ne_tile = tms.tile(bbox.east, bbox.north, zoom)  # Northeast corner

            # Determine the x and y ranges from the corner tiles
            min_x = min(sw_tile.x, se_tile.x, nw_tile.x, ne_tile.x)
            max_x = max(sw_tile.x, se_tile.x, nw_tile.x, ne_tile.x)
            min_y = min(sw_tile.y, se_tile.y, nw_tile.y, ne_tile.y)
            max_y = max(sw_tile.y, se_tile.y, nw_tile.y, ne_tile.y)

            # Choose a random tile within the range
            x = draw(st.integers(min_value=min_x, max_value=max_x))
            y = draw(st.integers(min_value=min_y, max_value=max_y))
            tile = Tile(x=x, y=y, z=zoom)
        except Exception:
            # If we can't get tiles for the bbox (e.g., bbox outside TMS bounds),
            # fall back to any valid tile
            minmax = tms.minmax(zoom)
            x = draw(
                st.integers(min_value=minmax["x"]["min"], max_value=minmax["x"]["max"])
            )
            y = draw(
                st.integers(min_value=minmax["y"]["min"], max_value=minmax["y"]["max"])
            )
            tile = Tile(x=x, y=y, z=zoom)
    else:
        # Original behavior - any valid tile for this zoom level
        minmax = tms.minmax(zoom)
        x = draw(st.integers(min_value=minmax["x"]["min"], max_value=minmax["x"]["max"]))
        y = draw(st.integers(min_value=minmax["y"]["min"], max_value=minmax["y"]["max"]))
        tile = Tile(x=x, y=y, z=zoom)

    return tile, tms


@pytest.mark.asyncio
@settings(max_examples=500)
@given(
    tile_tms=tile_and_tms(),
    ds=all_global_datasets,
    data=st.data(),
)
async def test_property_global_render_no_transparent_tile(
    tile_tms: tuple[Tile, TileMatrixSet],
    ds: xr.Dataset,
    data: st.DataObject,
    pytestconfig,
):
    """Property test that global datasets should never produce transparent pixels."""
    tile, tms = tile_tms
    query_params = create_query_params(
        tile, tms, size=2 ** data.draw(st.sampled_from(np.arange(6, 11)))
    )
    with config.set(max_pixel_factor=2):
        result = await pipeline(ds, query_params)
    transparent_percent = check_transparent_pixels(result.getvalue())
    if pytestconfig.getoption("--visualize"):
        visualize_tile(result, tile)
    assert transparent_percent == 0, (
        f"Found {transparent_percent:.1f}% transparent pixels in tile {tile}"
    )


@pytest.mark.asyncio
@given(data=st.data(), rect=global_datasets(allow_categorical=False))
@settings(max_examples=50)
async def test_property_equivalent_grids_render_equivalently(
    rect: xr.Dataset, data: st.DataObject, pytestconfig
):
    """
    Result from
    1. rectilinear grid
    2. curvilinear grid constructed from broadcasting out rectilinear grid
    3. unstructured grid constructed from stacking rectilinear grid
    must be preceptually very very similar.

    Note that this test receives new datasets and repeatedly triangulating the grid is slow;
    so we run that comparison in a different test with fewer datasets, and more tiles per dataset.
    """

    curvi = rect.rename(latitude="nlat", longitude="nlon")
    newlat, newlon = np.meshgrid(curvi.nlat.data, curvi.nlon.data, indexing="ij")
    curvi = curvi.assign_coords(
        longitude=(("nlon", "nlat"), newlon.T, {"standard_name": "longitude"}),
        latitude=(("nlon", "nlat"), newlat.T, {"standard_name": "latitude"}),
    )
    curvi["foo"].attrs["coordinates"] = "longitude latitude"

    # rectilinear = guess_grid_system(ds, "foo")
    # curvilinear = guess_grid_system(ds, "foo")

    # Check that grid indexers are the same
    # TODO: this is a hard invariant to maintain!
    #       because of rounding errors determining the bounds :(
    # bounds = tms.bounds(tile)
    # bbox = round_bbox(
    #     BBox(west=bounds.left, east=bounds.right, south=bounds.bottom, north=bounds.top),
    # )
    # npt.assert_array_equal(
    #     rectilinear.sel(rect_ds.foo, bbox=bbox).data,
    #     curvilinear.sel(ds.foo, bbox=bbox).data,
    # )

    lon, lat = curvi.longitude, curvi.latitude
    transposed = curvi.assign_coords(
        longitude=lon.transpose() if data.draw(st.booleans()) else curvi.longitude,
        latitude=lat.transpose() if data.draw(st.booleans()) else curvi.latitude,
    )

    # Compare images with optional debug visualization using perceptual comparison
    compare = lambda buffer1, buffer2, tile, tms: compare_image_buffers_with_debug(
        buffer1,
        buffer2,
        test_name="grid_equivalency",
        tile_info=(tile, tms),
        debug_visual=pytestconfig.getoption("--debug-visual", default=False),
        debug_visual_save=pytestconfig.getoption("--debug-visual-save", default=False),
        mode="perceptual",
        perceptual_threshold=0.9,  # 90% similarity threshold
    )

    for _ in range(10):
        tile, tms = data.draw(tile_and_tms())
        query = create_query_params(tile, tms)

        with config.set(transform_chunk_size=256, detect_approx_rectilinear=False):
            rectilinear_result = await pipeline(rect, query)

            curvilinear_result = await pipeline(curvi, query)
            images_similar, ssim_score = compare(
                rectilinear_result, curvilinear_result, tile, tms
            )
            assert images_similar, (
                f"Rectilinear and curvilinear results differ for tile {tile} (SSIM: {ssim_score:.4f})"
            )

            transposed_result = await pipeline(transposed, query)
            images_similar, ssim_score = compare(
                rectilinear_result, transposed_result, tile, tms
            )
            assert images_similar, (
                f"Rectilinear and *transposed* curvilinear results differ for tile {tile} (SSIM: {ssim_score:.4f})"
            )


@pytest.mark.asyncio
@given(
    data=st.data(),
    # disable perturbing the edges because for the triangular grid
    # we treat these as cell vertices, not centers
    rect=global_datasets(allow_categorical=False, maxsize=720, perturb=False),
)
@settings(max_examples=20)
async def test_rectilinear_triangular_equivalency(data, rect, pytestconfig):
    stacked = rect.load().stack(point=("latitude", "longitude"), create_index=False)
    stacked.attrs["_xpublish_id"] = str(uuid.uuid4())

    for _ in range(20):
        tile, tms = data.draw(tile_and_tms())
        query = create_query_params(tile, tms)

        with config.set(transform_chunk_size=256, detect_approx_rectilinear=False):
            rectilinear_result = await pipeline(rect, query)

            triangular_result = await pipeline(stacked, query)
            images_similar, ssim_score = compare_image_buffers_with_debug(
                rectilinear_result,
                triangular_result,
                test_name="rectilinear_triangular_equivalency",
                tile_info=(tile, tms),
                debug_visual=pytestconfig.getoption("--debug-visual", default=False),
                debug_visual_save=pytestconfig.getoption(
                    "--debug-visual-save", default=False
                ),
                mode="perceptual",
                perceptual_threshold=0.9,  # 90% similarity threshold
            )
            assert images_similar, (
                f"Rectilinear and triangular results differ for tile {tile} (SSIM: {ssim_score:.4f})"
            )


@pytest.mark.asyncio
@given(dataset=st.sampled_from([HRRR_MULTIPLE, EU3035_HIRES, HRRR]), data=st.data())
async def test_projected_coordinate_succeeds(dataset, data, pytestconfig):
    """Test that projected coordinate datasets can successfully render tiles within their bbox."""
    ds = dataset.create()

    # Use the strategy to generate a tile and TMS that overlaps with dataset bbox
    bbox = ds.attrs["bbox"]
    tile, tms = data.draw(tile_and_tms(bbox=bbox))

    # Create query parameters and render the tile
    query_params = create_query_params(tile, tms)

    try:
        result = await pipeline(ds, query_params)
        # Basic validation - ensure we got a result
        assert result is not None
        result_bytes = result.getvalue()
        assert len(result_bytes) > 0

        # Verify it's a valid PNG
        # PNG files start with an 8-byte signature
        png_signature = b"\x89PNG\r\n\x1a\n"
        assert result_bytes[:8] == png_signature, (
            f"Result does not have valid PNG signature, got {result_bytes[:8]!r}"
        )

        if pytestconfig.getoption("--visualize"):
            visualize_tile(result, tile)
    except TileTooBigError:
        assume(False)


@pytest.mark.asyncio
@given(tile_tms=tile_and_tms(), ds=all_global_datasets, data=st.data())
@settings(max_examples=50)
# @reproduce_failure('6.138.3', b'AXicc2R0ZECFDBoMUKBhH5Dke+njyj8MjqyO3I4MAI0xCBQ=')
async def test_zoom_in_doesnt_change_rendering(tile_tms, ds, data, pytestconfig) -> None:
    """Property test that zooming in doesn't change rendering.

    For a quadtree TMS, rendering a tile at a large size should produce the same
    result as rendering a sub-tile at a higher zoom level that covers the same sub-area.

    We render a parent tile at 2048x2048, then test 20 randomly generated sub-tiles
    at higher zoom levels (up to 6 levels deeper, which gives us 32x32 minimum tile size).

    Conceptual diagram (example with zoom delta = 2):

        Parent tile (x, y, z) @ 2048×2048:
        ┌──────────────────────────┐
        │▓▓▓▓▓▓│      │      │     │  Child tiles at zoom z+2
        │▓▓▓▓▓▓│      │      │     │  (4×4 grid, each 512×512)
        │▓▓▓▓▓▓│      │      │     │
        ├──────┼──────┼──────┼─────┤  Shaded tile (▓▓) is at
        │      │      │      │     │  top-left corner, aligned
        │      │      │      │     │  with parent boundary
        │      │      │      │     │
        ├──────┼──────┼──────┼─────┤
        │      │      │      │     │
        │      │      │      │     │
        │      │      │      │     │
        ├──────┼──────┼──────┼─────┤
        │      │      │      │     │
        │      │      │      │     │
        │      │      │      │     │
        └──────┴──────┴──────┴─────┘

        We render the child tile (▓▓) independently at 512×512,
        then compare it to the corresponding top-left 512×512
        region extracted from the parent rendering.

    In a quadtree, tile (x, y, z) contains 2^n × 2^n child tiles at zoom z+n:
        Child tile coordinates: (2^n·x + i, 2^n·y + j, z+n)
        where i, j ∈ [0, 2^n - 1]

    TODO:
    1. Support coarsening, this might require symmetric padding
    2. Support higher perceptual threshold, even without coarsening; This might require constructing appropriate datasets
       with shapes and spacings that match the parent tile properly.
    """
    tile, tms = tile_tms

    # Assert that TMS is a quadtree
    # WebMercatorQuad and WorldCRS84Quad are both quadtrees
    assert tms.id in [
        "WebMercatorQuad",
        "WorldCRS84Quad",
    ], f"TMS {tms.id} is not a known quadtree"

    # Render parent tile at 2048x2048
    parent_size = 2048
    parent_query = create_query_params(tile, tms, size=parent_size)

    with config.set(max_pixel_factor=20000):
        parent_result = await pipeline(ds, parent_query)

    # Convert parent PNG to numpy array
    parent_result.seek(0)
    parent_img = Image.open(parent_result)
    parent_array = np.array(parent_img)

    # Pick a child zoom level (absolute)
    # We choose minimum tile size of 32x32 (2048/2^6 = 32), so max delta is 6
    # We use 4, because at higher zoom levels, pixels can move around a bit.
    max_child_zoom = min(tile.z + 4, tms.maxzoom)
    assume(max_child_zoom > tile.z)  # Must be able to zoom in at least one level

    # Test 10 randomly generated child tiles
    for _ in range(10):
        child_zoom = data.draw(
            st.integers(min_value=tile.z + 1, max_value=max_child_zoom)
        )

        # Calculate the zoom delta between parent and child
        zoom_delta = child_zoom - tile.z

        # At zoom level child_zoom, the parent tile (x, y, tile.z) corresponds to
        # 2^delta × 2^delta child tiles with coordinates ranging from:
        # x: [2^delta * parent_x, 2^delta * parent_x + 2^delta - 1]
        # y: [2^delta * parent_y, 2^delta * parent_y + 2^delta - 1]
        num_tiles_per_side = 2**zoom_delta

        # Pick a random child tile within this range
        child_x_offset = data.draw(
            st.integers(min_value=0, max_value=num_tiles_per_side - 1)
        )
        child_y_offset = data.draw(
            st.integers(min_value=0, max_value=num_tiles_per_side - 1)
        )

        child_tile = Tile(
            x=num_tiles_per_side * tile.x + child_x_offset,
            y=num_tiles_per_side * tile.y + child_y_offset,
            z=child_zoom,
        )

        # Calculate the size to render the child tile
        # Since parent is 2048x2048 and covers 2^zoom_delta × 2^zoom_delta child tiles,
        # each child tile corresponds to (2048 / 2^zoom_delta)^2 pixels
        child_size = parent_size // num_tiles_per_side

        # Render the child tile
        child_query = create_query_params(child_tile, tms, size=child_size)
        # Turn off coarsening
        with config.set(max_pixel_factor=20000):
            child_result = await pipeline(ds, child_query)

        # Extract corresponding region from parent
        pixel_x_start = child_x_offset * child_size
        pixel_y_start = child_y_offset * child_size
        pixel_x_end = (child_x_offset + 1) * child_size
        pixel_y_end = (child_y_offset + 1) * child_size

        parent_region = parent_array[pixel_y_start:pixel_y_end, pixel_x_start:pixel_x_end]

        # Convert parent region to PNG buffer for comparison
        parent_region_buffer = io.BytesIO()
        Image.fromarray(parent_region).save(parent_region_buffer, format="PNG")
        parent_region_buffer.seek(0)
        child_result.seek(0)

        # Compare with perceptual similarity
        images_similar, ssim_score = compare_image_buffers_with_debug(
            buffer1=parent_region_buffer,  # expected (parent region)
            buffer2=child_result,  # actual (child tile)
            test_name=f"zoom_consistency_z{tile.z}_{tile.x}_{tile.y}_to_z{child_tile.z}_{child_tile.x}_{child_tile.y}",
            tile_info=(child_tile, tms),
            debug_visual=pytestconfig.getoption("--debug-visual", default=False),
            debug_visual_save=pytestconfig.getoption(
                "--debug-visual-save", default=False
            ),
            mode="perceptual",
            perceptual_threshold=0.98,
        )

        assert images_similar, (
            f"Child tile {child_tile} (zoom {child_zoom}, delta +{zoom_delta}) doesn't match parent tile {tile} (zoom {tile.z}) region (SSIM: {ssim_score:.4f})"
        )
