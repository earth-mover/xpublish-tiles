from itertools import product

import arraylake as al
import numpy as np
import pytest
from syrupy.extensions.image import PNGImageSnapshotExtension

import icechunk
import xarray as xr
from xpublish_tiles.datasets import Dim, uniform_grid

ARRAYLAKE_REPO = "earthmover-integration/tiles-datasets-develop"


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


@pytest.fixture(scope="session")
def air_dataset():
    return xr.tutorial.load_dataset("air_temperature")


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
    dims = []

    nlat, nlon = 720, 1441
    lats = np.linspace(-90, 90, nlat)
    if "90->-90" in param:
        lats = lats[::-1]

    if "-180->180" in param:
        lons = np.linspace(-180, 180, nlon)
    else:
        lons = np.linspace(0, 360, nlon)

    dims = [
        Dim(
            name="latitude",
            size=nlat,
            chunk_size=nlat,
            data=lats,
            attrs={"standard_name": "latitude"},
        ),
        Dim(
            name="longitude",
            size=nlon,
            chunk_size=nlon,
            data=lons,
            attrs={"standard_name": "longitude"},
        ),
    ]
    yield uniform_grid(dims=tuple(dims), dtype=np.float32, attrs={})


@pytest.fixture
def png_snapshot(snapshot):
    """PNG snapshot with custom numpy array comparison for robustness."""
    import io

    import numpy as np
    from PIL import Image

    class RobustPNGSnapshotExtension(PNGImageSnapshotExtension):
        def matches(self, *, serialized_data: bytes, snapshot_data: bytes) -> bool:
            """
            Compare PNG images as numpy arrays instead of raw bytes.
            This is more robust against compression differences and platform variations.
            """
            try:
                # Convert both images to numpy arrays
                actual_img = Image.open(io.BytesIO(serialized_data))
                expected_img = Image.open(io.BytesIO(snapshot_data))

                actual_array = np.array(actual_img)
                expected_array = np.array(expected_img)

                # Use numpy array equality comparison
                return np.array_equal(actual_array, expected_array)

            except Exception:
                # Fallback to byte comparison if image parsing fails
                return super().matches(
                    serialized_data=serialized_data, snapshot_data=snapshot_data
                )

    return snapshot.use_extension(RobustPNGSnapshotExtension)
