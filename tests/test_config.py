"""Test configuration management with donfig."""

import numpy as np

import xarray as xr
from xpublish_tiles.config import config
from xpublish_tiles.lib import get_transform_chunk_size


def test_default_config_values():
    """Test that default configuration values are set correctly."""
    assert config.get("num_threads") == 8
    assert config.get("transform_chunk_size") == 1024
    assert config.get("detect_approx_rectilinear") is True
    assert config.get("default_pad") == 2
    assert config.get("max_renderable_size") == 1024**3


def test_config_with_context_manager():
    """Test that configuration can be modified with context manager."""
    # Check defaults
    assert config.get("num_threads") == 8
    assert config.get("transform_chunk_size") == 1024

    # Use context manager to temporarily change values
    with config.set(num_threads=16, transform_chunk_size=512):
        assert config.get("num_threads") == 16
        assert config.get("transform_chunk_size") == 512

    # Values should revert after context manager exits
    assert config.get("num_threads") == 8
    assert config.get("transform_chunk_size") == 1024


def test_get_transform_chunk_size():
    """Test that the dynamic config functions work correctly."""
    # Test defaults
    da = xr.DataArray(np.ones((1024, 1024)), dims=("x", "y"))
    assert get_transform_chunk_size(da) == (1024, 1024)

    # Test with context manager
    with config.set(transform_chunk_size=256):
        assert get_transform_chunk_size(da) == (64, 1024)

    da = xr.DataArray(np.ones((2048, 2048)), dims=("x", "y"))
    assert get_transform_chunk_size(da) == (512, 2048)


def test_detect_approx_rectilinear_config():
    """Test that detect_approx_rectilinear configuration works correctly."""
    # Check default is True
    assert config.get("detect_approx_rectilinear") is True

    # Test disabling approximate rectilinear detection
    with config.set(detect_approx_rectilinear=False):
        assert config.get("detect_approx_rectilinear") is False

    # Verify it reverts to True after context manager
    assert config.get("detect_approx_rectilinear") is True


def test_max_renderable_size_config():
    """Test that max_renderable_size configuration works correctly."""
    # Check default value (10,000 * 10,000 pixels)
    assert config.get("max_renderable_size") == 1024**3

    # Test changing the value with context manager
    with config.set(max_renderable_size=50_000_000):
        assert config.get("max_renderable_size") == 50_000_000

    # Verify it reverts to default after context manager
    assert config.get("max_renderable_size") == 1024**3
