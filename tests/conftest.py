import io
from itertools import product

import arraylake as al
import numpy as np
import pytest
from PIL import Image
from pyproj.aoi import BBox

import icechunk
import xarray as xr
from xpublish_tiles.testing.datasets import EU3035, HRRR, create_global_dataset
from xpublish_tiles.testing.fixtures import png_snapshot  # noqa: F401
from xpublish_tiles.testing.tiles import ETRS89_TILES, HRRR_TILES

ARRAYLAKE_REPO = "earthmover-integration/tiles-datasets-develop"
IS_SNAPSHOT_UPDATE = False


def compare_image_buffers(buffer1: io.BytesIO, buffer2: io.BytesIO) -> bool:
    """Compare two image BytesIO buffers by converting them to numpy arrays."""
    buffer1.seek(0)
    buffer2.seek(0)

    # Convert both images to numpy arrays
    img1 = Image.open(buffer1)
    img2 = Image.open(buffer2)

    array1 = np.array(img1)
    array2 = np.array(img2)

    # Compare arrays using numpy array equality
    return np.array_equal(array1, array2)


def pytest_addoption(parser):
    parser.addoption(
        "--where",
        action="store",
        choices=["local", "arraylake"],
        default="local",
        help="Storage backend: 'local' for local filesystem or 'arraylake' for Arraylake (default: local)",
    )
    parser.addoption(
        "--prefix",
        action="store",
        help="Prefix for the repository/storage path (defaults: local=/tmp/tiles-icechunk/, arraylake=earthmover-integration/tiles-icechunk/)",
    )
    parser.addoption("--setup", action="store_true", help="Run setup tests (test_create)")
    parser.addoption(
        "--debug-visual",
        action="store_true",
        help="Show visual difference plots in matplotlib window when PNG snapshots don't match (automatically disables parallelization)",
    )
    parser.addoption(
        "--debug-visual-save",
        action="store_true",
        help="Save visual difference plots to PNG files and auto-open them (automatically disables parallelization)",
    )
    parser.addoption(
        "--visualize",
        action="store_true",
        help="Show matplotlib visualization windows during tests",
    )


def pytest_configure(config):
    """Configure pytest settings based on command line options."""
    # Disable parallelization when debug visual options are used
    if config.getoption("--debug-visual") or config.getoption("--debug-visual-save"):
        # Check if pytest-xdist is being used and disable it
        if hasattr(config.option, "numprocesses") and config.option.numprocesses != 0:
            config.option.numprocesses = 0
            print(
                "🔍 Debug visual mode enabled - disabling parallel execution for better visualization"
            )

        # Also disable dist mode completely
        if hasattr(config.option, "dist") and config.option.dist:
            config.option.dist = "no"


@pytest.fixture(scope="session")
def air_dataset():
    ds = xr.tutorial.load_dataset("air_temperature")
    ds.air.attrs["valid_min"] = 271
    ds.air.attrs["valid_max"] = 317.4
    return ds


@pytest.fixture(scope="session")
def where(request):
    return request.config.getoption("--where")


@pytest.fixture(scope="session")
def prefix(request, where):
    provided_prefix = request.config.getoption("--prefix")
    if provided_prefix:
        return provided_prefix

    # Use defaults based on storage backend
    if where == "local":
        return "/tmp/tiles-icechunk/"
    elif where == "arraylake":
        return "earthmover-integration/tiles-icechunk/"
    else:
        raise ValueError(f"No default prefix available for storage backend: {where}")


def generate_repo(where: str, prefix: str):
    """Generate an icechunk Repository based on storage backend choice.

    Args:
        where: Storage backend - 'local' or 'arraylake'
        prefix: Prefix for the repository/storage path

    Returns:
        icechunk.Repository: Repository object for the specified backend
    """
    if where == "local":
        storage = icechunk.local_filesystem_storage(prefix)
        try:
            # Try to open existing repository
            return icechunk.Repository.open(storage)
        except Exception:
            # Create new repository if it doesn't exist
            return icechunk.Repository.create(storage)
    elif where == "arraylake":
        client = al.Client()
        repo = client.get_or_create_repo(ARRAYLAKE_REPO)
        return repo
    else:
        raise ValueError(f"Unsupported storage backend: {where}")


@pytest.fixture
def repo(where, prefix):
    return generate_repo(where, prefix)


@pytest.fixture(
    params=tuple(map(",".join, product(["-90->90", "90->-90"], ["-180->180", "0->360"])))
)
def global_datasets(request):
    param = request.param

    # Parse parameters to determine coordinate ordering
    lat_ascending = "-90->90" in param
    lon_0_360 = "0->360" in param

    yield create_global_dataset(lat_ascending=lat_ascending, lon_0_360=lon_0_360)


# Create the product of datasets and their appropriate tiles
def _get_projected_dataset_tile_params():
    params = []
    for dataset_class, tiles in [
        (EU3035, ETRS89_TILES),
        (HRRR, HRRR_TILES),
    ]:
        for tile_param in tiles:
            tile, tms = tile_param.values
            param_id = f"{dataset_class.name}_{tile_param.id}"
            params.append(pytest.param((dataset_class, tile, tms), id=param_id))
    return params


@pytest.fixture(params=_get_projected_dataset_tile_params())
def projected_dataset_and_tile(request):
    dataset_class, tile, tms = request.param
    ds = dataset_class.create()

    # Validate that tile overlaps with dataset bounding box
    dataset_bbox = ds.attrs["bbox"]
    tile_bounds = tms.bounds(tile)
    tile_bbox = BBox(
        west=tile_bounds.left,
        south=tile_bounds.bottom,
        east=tile_bounds.right,
        north=tile_bounds.top,
    )

    # Check if dataset bbox intersects with tile bounds
    if not dataset_bbox.intersects(tile_bbox):
        pytest.skip(f"Tile {tile} does not overlap with dataset bbox {dataset_bbox}")

    return (ds, tile, tms)
