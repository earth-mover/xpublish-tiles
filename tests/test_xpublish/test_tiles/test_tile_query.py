"""Tests for TileQuery model validation."""

from xpublish_tiles.types import ImageFormat
from xpublish_tiles.xpublish.tiles.types import TileQuery


class TestTileQueryColormap:
    """Test TileQuery model validation for colormap parameter."""

    def test_tile_query_colormap_validation_valid(self):
        """Test that valid colormap input is accepted."""
        colormap = {"0": "#ffffff", "128": "#808080", "255": "#000000"}
        query = TileQuery(
            variables=["air"],
            colormap=colormap,
            width=256,
            height=256,
            colorscalerange=None,
            style=None,
            f=ImageFormat.PNG,
            render_errors=False,
        )
        assert query.colormap == colormap

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

    def test_tile_query_colormap_with_style_succeeds(self):
        """Test that colormap with style is allowed - colormap overrides style colormap."""
        colormap = {"0": "#ffffff", "255": "#000000"}

        # Test with raster/default
        query = TileQuery(
            variables=["air"],
            colormap=colormap,
            style="raster/default",  # type: ignore  # Pydantic converts string to tuple
            width=256,
            height=256,
            colorscalerange=None,
            f=ImageFormat.PNG,
            render_errors=False,
        )
        assert query.colormap == colormap
        assert query.style == ("raster", "default")

        # Test with raster/viridis
        query = TileQuery(
            variables=["air"],
            colormap=colormap,
            style="raster/viridis",  # type: ignore  # Pydantic converts string to tuple
            width=256,
            height=256,
            colorscalerange=None,
            f=ImageFormat.PNG,
            render_errors=False,
        )
        assert query.colormap == colormap
        assert query.style == ("raster", "viridis")
