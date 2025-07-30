import pytest

from xpublish_tiles.types import ImageFormat
from xpublish_tiles.validators import validate_colorscalerange, validate_image_format


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
