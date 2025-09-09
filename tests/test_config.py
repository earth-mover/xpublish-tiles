"""Test configuration management with donfig."""

from xpublish_tiles.config import config
from xpublish_tiles.lib import get_transform_chunk_size


def test_default_config_values():
    """Test that default configuration values are set correctly."""
    assert config.get("num_threads") == 8
    assert config.get("rectilinear_check_subsample_step") == 2
    assert config.get("transform_chunk_size") == 1024
    assert config.get("detect_approx_rectilinear") is True
    assert config.get("default_pad") == 2


def test_config_with_context_manager():
    """Test that configuration can be modified with context manager."""
    # Check defaults
    assert config.get("num_threads") == 8
    assert config.get("rectilinear_check_subsample_step") == 2
    assert config.get("transform_chunk_size") == 1024

    # Use context manager to temporarily change values
    with config.set(
        num_threads=16, rectilinear_check_subsample_step=5, transform_chunk_size=512
    ):
        assert config.get("num_threads") == 16
        assert config.get("rectilinear_check_subsample_step") == 5
        assert config.get("transform_chunk_size") == 512

    # Values should revert after context manager exits
    assert config.get("num_threads") == 8
    assert config.get("rectilinear_check_subsample_step") == 2
    assert config.get("transform_chunk_size") == 1024


def test_dynamic_config_in_lib():
    """Test that the dynamic config functions work correctly."""
    # Test defaults
    assert get_transform_chunk_size() == (1024, 1024)

    # Test subsample step directly from config
    assert config.get("rectilinear_check_subsample_step") == 2

    # Test with context manager
    with config.set(transform_chunk_size=256, rectilinear_check_subsample_step=5):
        assert get_transform_chunk_size() == (256, 256)
        assert config.get("rectilinear_check_subsample_step") == 5


def test_detect_approx_rectilinear_config():
    """Test that detect_approx_rectilinear configuration works correctly."""
    # Check default is True
    assert config.get("detect_approx_rectilinear") is True

    # Test disabling approximate rectilinear detection
    with config.set(detect_approx_rectilinear=False):
        assert config.get("detect_approx_rectilinear") is False

    # Verify it reverts to True after context manager
    assert config.get("detect_approx_rectilinear") is True
