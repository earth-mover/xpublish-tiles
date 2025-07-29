import pytest

from xpublish_tiles.utils import parse_colorscalerange


class TestUnpackColorscalerange:
    """Test cases for the unpack_colorscalerange function"""

    def test_valid_positive_range(self):
        """Test unpacking valid positive range"""
        result = parse_colorscalerange("0.5,10.2")
        assert result == (0.5, 10.2)

    def test_valid_negative_range(self):
        """Test unpacking valid negative range"""
        result = parse_colorscalerange("-5.0,-1.5")
        assert result == (-5.0, -1.5)

    def test_valid_mixed_range(self):
        """Test unpacking valid mixed positive/negative range"""
        result = parse_colorscalerange("-2.5,7.8")
        assert result == (-2.5, 7.8)

    def test_valid_integer_values(self):
        """Test unpacking integer values"""
        result = parse_colorscalerange("0,100")
        assert result == (0.0, 100.0)

    def test_valid_zero_values(self):
        """Test unpacking with zero values"""
        result = parse_colorscalerange("0,0")
        assert result == (0.0, 0.0)

    def test_valid_scientific_notation(self):
        """Test unpacking scientific notation values"""
        result = parse_colorscalerange("1e-3,1.5e2")
        assert result == (0.001, 150.0)

    def test_valid_with_whitespace(self):
        """Test unpacking with whitespace around values"""
        result = parse_colorscalerange(" 1.0 , 2.0 ")
        assert result == (1.0, 2.0)

    def test_invalid_single_value(self):
        """Test error handling for single value"""
        with pytest.raises(ValueError, match="Invalid color scale range format: 5.0"):
            parse_colorscalerange("5.0")

    def test_invalid_three_values(self):
        """Test error handling for three values"""
        with pytest.raises(ValueError, match="Invalid color scale range format: 1,2,3"):
            parse_colorscalerange("1,2,3")

    def test_invalid_empty_string(self):
        """Test error handling for empty string"""
        with pytest.raises(ValueError, match="Invalid color scale range format: "):
            parse_colorscalerange("")

    def test_invalid_non_numeric_values(self):
        """Test error handling for non-numeric values"""
        with pytest.raises(ValueError, match="Invalid color scale range format: a,b"):
            parse_colorscalerange("a,b")

    def test_invalid_partial_numeric(self):
        """Test error handling for partially numeric values"""
        with pytest.raises(ValueError, match="Invalid color scale range format: 1.0,abc"):
            parse_colorscalerange("1.0,abc")

    def test_invalid_comma_only(self):
        """Test error handling for comma only"""
        with pytest.raises(ValueError, match="Invalid color scale range format: ,"):
            parse_colorscalerange(",")

    def test_invalid_no_comma(self):
        """Test error handling for no comma separator"""
        with pytest.raises(ValueError, match="Invalid color scale range format: 1.0 2.0"):
            parse_colorscalerange("1.0 2.0")

    def test_invalid_multiple_commas(self):
        """Test error handling for multiple commas"""
        with pytest.raises(ValueError, match="Invalid color scale range format: 1,,2"):
            parse_colorscalerange("1,,2")

    def test_edge_case_very_large_numbers(self):
        """Test with very large numbers"""
        result = parse_colorscalerange("1e10,9.99e20")
        assert result == (1e10, 9.99e20)

    def test_edge_case_very_small_numbers(self):
        """Test with very small numbers"""
        result = parse_colorscalerange("1e-10,9.99e-20")
        assert result == (1e-10, 9.99e-20)

    def test_return_type(self):
        """Test that function returns tuple of floats"""
        result = parse_colorscalerange("1,2")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)
