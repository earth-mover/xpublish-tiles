import arraylake as al
import pytest

import icechunk
import xarray as xr

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
