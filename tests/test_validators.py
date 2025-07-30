import pytest

from xpublish_tiles.types import ImageFormat, Style
from xpublish_tiles.validators import (
    validate_colorscalerange,
    validate_image_format,
    validate_style,
)


class TestValidateColorscalerange:
    def test_valid_colorscalerange(self):
        result = validate_colorscalerange("0.0,1.0")
        assert result == (0.0, 1.0)

    def test_valid_colorscalerange_negative(self):
        result = validate_colorscalerange("-10.5,20.3")
        assert result == (-10.5, 20.3)

    def test_valid_colorscalerange_integers(self):
        result = validate_colorscalerange("0,100")
        assert result == (0.0, 100.0)

    def test_none_input(self):
        result = validate_colorscalerange(None)
        assert result is None

    def test_invalid_format_single_value(self):
        with pytest.raises(
            ValueError, match="colorscalerange must be in the format 'min,max'"
        ):
            validate_colorscalerange("1.0")

    def test_invalid_format_three_values(self):
        with pytest.raises(
            ValueError, match="colorscalerange must be in the format 'min,max'"
        ):
            validate_colorscalerange("1.0,2.0,3.0")

    def test_invalid_format_empty_string(self):
        with pytest.raises(
            ValueError, match="colorscalerange must be in the format 'min,max'"
        ):
            validate_colorscalerange("")

    def test_invalid_float_first_value(self):
        with pytest.raises(
            ValueError,
            match="colorscalerange must be in the format 'min,max' where min and max are valid floats",
        ):
            validate_colorscalerange("invalid,1.0")

    def test_invalid_float_second_value(self):
        with pytest.raises(
            ValueError,
            match="colorscalerange must be in the format 'min,max' where min and max are valid floats",
        ):
            validate_colorscalerange("1.0,invalid")

    def test_invalid_float_both_values(self):
        with pytest.raises(
            ValueError,
            match="colorscalerange must be in the format 'min,max' where min and max are valid floats",
        ):
            validate_colorscalerange("invalid,also_invalid")


class TestValidateImageFormat:
    def test_valid_png_format(self):
        result = validate_image_format("png")
        assert result == ImageFormat.PNG

    def test_valid_jpeg_format(self):
        result = validate_image_format("jpeg")
        assert result == ImageFormat.JPEG

    def test_valid_png_format_uppercase(self):
        result = validate_image_format("PNG")
        assert result == ImageFormat.PNG

    def test_valid_jpeg_format_uppercase(self):
        result = validate_image_format("JPEG")
        assert result == ImageFormat.JPEG

    def test_valid_format_with_mime_type(self):
        result = validate_image_format("image/png")
        assert result == ImageFormat.PNG

    def test_valid_format_with_mime_type_jpeg(self):
        result = validate_image_format("image/jpeg")
        assert result == ImageFormat.JPEG

    def test_none_input(self):
        result = validate_image_format(None)
        assert result is None

    def test_invalid_format(self):
        with pytest.raises(
            ValueError, match="image format gif is not valid. Options are: PNG, JPEG"
        ):
            validate_image_format("gif")

    def test_invalid_format_with_mime_type(self):
        with pytest.raises(
            ValueError, match="image format gif is not valid. Options are: PNG, JPEG"
        ):
            validate_image_format("image/gif")


class TestValidateStyle:
    def test_valid_raster_style(self):
        result = validate_style("raster/default")
        assert result == (Style.RASTER, "default")

    def test_valid_quiver_style(self):
        result = validate_style("quiver/arrows")
        assert result == (Style.QUIVER, "arrows")

    def test_valid_numpy_style(self):
        result = validate_style("numpy/colormap")
        assert result == (Style.NUMPY, "colormap")

    def test_valid_vector_style(self):
        result = validate_style("vector/lines")
        assert result == (Style.VECTOR, "lines")

    def test_valid_style_lowercase(self):
        result = validate_style("raster/default")
        assert result == (Style.RASTER, "default")

    def test_valid_style_mixed_case(self):
        result = validate_style("RaStEr/default")
        assert result == (Style.RASTER, "default")

    def test_none_input(self):
        result = validate_style(None)
        assert result is None

    def test_invalid_format_single_value(self):
        with pytest.raises(
            ValueError,
            match="style must be in the format 'stylename/palettename'. A common default for this is 'raster/default'",
        ):
            validate_style("raster")

    def test_invalid_format_three_values(self):
        with pytest.raises(
            ValueError,
            match="style must be in the format 'stylename/palettename'. A common default for this is 'raster/default'",
        ):
            validate_style("raster/default/extra")

    def test_invalid_format_empty_string(self):
        with pytest.raises(
            ValueError,
            match="style must be in the format 'stylename/palettename'. A common default for this is 'raster/default'",
        ):
            validate_style("")

    def test_invalid_style_name(self):
        with pytest.raises(
            ValueError,
            match="style invalid is not valid. Options are: RASTER, QUIVER, NUMPY, VECTOR",
        ):
            validate_style("invalid/default")
