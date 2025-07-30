import pytest

from xpublish_tiles.validators import validate_colorscalerange


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
