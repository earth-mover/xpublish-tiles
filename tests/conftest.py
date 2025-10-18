import logging
from itertools import product

import pytest
from hypothesis import Verbosity, settings

import icechunk
import xarray as xr
from xpublish_tiles.testing.datasets import (
    CURVILINEAR,
    EU3035,
    EU3035_HIRES,
    HRRR,
    REDGAUSS_N320,
    UTM33S,
    create_global_dataset,
)
from xpublish_tiles.testing.lib import compare_image_buffers, png_snapshot  # noqa: F401
from xpublish_tiles.testing.tiles import CURVILINEAR_TILES

# Disable numba, datashader, and PIL debug logs
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("datashader").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

IS_SNAPSHOT_UPDATE = False

settings.register_profile(
    "ci",
    deadline=None,
    # suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    # verbosity=Verbosity.verbose,
    print_blob=True,
)

settings.register_profile(
    "default",
    deadline=None,
    # suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    verbosity=Verbosity.verbose,
    print_blob=True,
)


def pytest_addoption(parser):
    parser.addoption(
        "--setup", action="store_true", help="Run setup tests (test_create_local_dataset)"
    )
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


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    if config.getoption("--setup"):
        # Filter to only include test_create_local_dataset function from test_create_datasets.py when --setup is specified
        setup_items = [
            item
            for item in items
            if item.name.startswith("test_create_local_dataset")
            and "test_create_datasets.py" in str(item.fspath)
        ]
        items[:] = setup_items


@pytest.fixture(scope="session")
def air_dataset():
    ds = xr.tutorial.load_dataset("air_temperature")
    ds.air.attrs["valid_min"] = 271
    ds.air.attrs["valid_max"] = 317.4
    return ds


@pytest.fixture
def repo(pytestconfig):
    """Generate an icechunk Repository for local filesystem storage."""
    if not pytestconfig.getoption("--setup", default=False):
        pytest.skip("repo fixture only available when --setup flag is provided")

    prefix = "/tmp/tiles-icechunk/"
    storage = icechunk.local_filesystem_storage(prefix)
    try:
        # Try to open existing repository
        return icechunk.Repository.open(storage)
    except Exception:
        # Create new repository if it doesn't exist
        return icechunk.Repository.create(storage)


@pytest.fixture(
    params=tuple(map(",".join, product(["-90->90", "90->-90"], ["-180->180", "0->360"])))
    + ("reduced_gaussian_n320",)
)
def global_datasets(request):
    param = request.param

    # Parse parameters to determine coordinate ordering
    lat_ascending = "-90->90" in param
    lon_0_360 = "0->360" in param

    if param == "reduced_gaussian_n320":
        ds = REDGAUSS_N320.create()
    else:
        ds = create_global_dataset(lat_ascending=lat_ascending, lon_0_360=lon_0_360)
    ds.attrs["name"] = param
    yield ds


# Create the product of datasets and their appropriate tiles
def _get_projected_dataset_tile_params():
    params = []
    for dataset_class in [UTM33S, EU3035, EU3035_HIRES, HRRR]:
        # Use the tiles directly from the dataset class
        for tile_param in dataset_class.tiles:
            tile, tms = tile_param.values
            param_id = f"{dataset_class.name}_{tile_param.id}"
            params.append(pytest.param((dataset_class, tile, tms), id=param_id))
    return params


@pytest.fixture(params=_get_projected_dataset_tile_params())
def projected_dataset_and_tile(request):
    dataset_class, tile, tms = request.param
    ds = dataset_class.create()
    return (ds, tile, tms)


@pytest.fixture(params=CURVILINEAR_TILES)
def curvilinear_dataset_and_tile(request):
    tile, tms = request.param
    return (CURVILINEAR.create(), tile, tms)
