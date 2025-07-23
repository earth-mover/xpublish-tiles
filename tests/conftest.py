import pytest

import xarray as xr


@pytest.fixture(scope="session")
def air_dataset():
    return xr.tutorial.load_dataset("air_temperature")
