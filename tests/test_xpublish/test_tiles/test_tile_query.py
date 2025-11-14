"""Tests for TileQuery model validation."""

from xpublish_tiles.types import ImageFormat
from xpublish_tiles.xpublish.tiles.types import TileQuery


class TestTileQueryColormap:
    """Test TileQuery model validation for colormap parameter."""

    def test_tile_query_colormap_validation_valid(self):
        """Test that valid colormap input is accepted with raster/custom style."""
        colormap = {"0": "#ffffff", "128": "#808080", "255": "#000000"}
        query = TileQuery(
            variables=["air"],
            colormap=colormap,
            width=256,
            height=256,
            colorscalerange=None,
            style="raster/custom",  # type: ignore  # Pydantic converts string to tuple
            f=ImageFormat.PNG,
            render_errors=False,
        )
        assert query.colormap == colormap
        assert query.style == ("raster", "custom")

    def test_tile_query_colormap_validation_none(self):
        """Test that None colormap is handled correctly."""
        query = TileQuery(
            variables=["air"],
            width=256,
            height=256,
            colorscalerange=None,
            style=None,
            colormap=None,
            f=ImageFormat.PNG,
            render_errors=False,
        )
        assert query.colormap is None

    def test_tile_query_colormap_requires_custom_style(self):
        """Test that colormap requires raster/custom style."""
        import pytest
        from pydantic import ValidationError

        colormap = {"0": "#ffffff", "255": "#000000"}

        # Test that raster/custom works
        query = TileQuery(
            variables=["air"],
            colormap=colormap,
            style="raster/custom",  # type: ignore  # Pydantic converts string to tuple
            width=256,
            height=256,
            colorscalerange=None,
            f=ImageFormat.PNG,
            render_errors=False,
        )
        assert query.colormap == colormap
        assert query.style == ("raster", "custom")

        # Test that raster/default fails
        with pytest.raises(ValidationError, match="must be 'raster/custom'"):
            TileQuery(
                variables=["air"],
                colormap=colormap,
                style="raster/default",  # type: ignore
                width=256,
                height=256,
                colorscalerange=None,
                f=ImageFormat.PNG,
                render_errors=False,
            )

        # Test that raster/viridis fails
        with pytest.raises(ValidationError, match="must be 'raster/custom'"):
            TileQuery(
                variables=["air"],
                colormap=colormap,
                style="raster/viridis",  # type: ignore
                width=256,
                height=256,
                colorscalerange=None,
                f=ImageFormat.PNG,
                render_errors=False,
            )
