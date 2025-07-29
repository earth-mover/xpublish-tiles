import pytest

from xpublish_tiles.types import Style
from xpublish_tiles.utils import parse_colorscalerange, parse_style


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


class TestParseStyle:
    """Test cases for the parse_style function"""

    def test_valid_raster_style(self):
        """Test parsing valid raster style"""
        result = parse_style("raster/viridis")
        assert result == (Style.RASTER, "viridis")

    def test_valid_quiver_style(self):
        """Test parsing valid quiver style"""
        result = parse_style("quiver/plasma")
        assert result == (Style.QUIVER, "plasma")

    def test_valid_numpy_style(self):
        """Test parsing valid numpy style"""
        result = parse_style("numpy/coolwarm")
        assert result == (Style.NUMPY, "coolwarm")

    def test_valid_vector_style(self):
        """Test parsing valid vector style"""
        result = parse_style("vector/jet")
        assert result == (Style.VECTOR, "jet")

    def test_valid_uppercase_style(self):
        """Test parsing uppercase style names"""
        result = parse_style("RASTER/viridis")
        assert result == (Style.RASTER, "viridis")

    def test_valid_mixed_case_style(self):
        """Test parsing mixed case style names"""
        result = parse_style("Raster/viridis")
        assert result == (Style.RASTER, "viridis")

    def test_valid_complex_colormap_name(self):
        """Test parsing with complex colormap names"""
        result = parse_style("raster/RdYlBu_r")
        assert result == (Style.RASTER, "RdYlBu_r")

    def test_valid_colormap_with_numbers(self):
        """Test parsing colormap names with numbers"""
        result = parse_style("raster/Set1")
        assert result == (Style.RASTER, "Set1")

    def test_invalid_style_name(self):
        """Test error handling for invalid style name"""
        with pytest.raises(ValueError, match="Invalid style format: invalid/viridis"):
            parse_style("invalid/viridis")

    def test_invalid_no_separator(self):
        """Test error handling for missing separator"""
        with pytest.raises(ValueError, match="Invalid style format: raster"):
            parse_style("raster")

    def test_invalid_empty_string(self):
        """Test error handling for empty string"""
        with pytest.raises(ValueError, match="Invalid style format: "):
            parse_style("")

    def test_invalid_only_separator(self):
        """Test error handling for only separator"""
        with pytest.raises(ValueError, match="Invalid style format: /"):
            parse_style("/")

    def test_invalid_multiple_separators(self):
        """Test parsing with multiple separators (only first split is used)"""
        result = parse_style("raster/viridis/extra")
        assert result == (Style.RASTER, "viridis/extra")

    def test_invalid_empty_style(self):
        """Test error handling for empty style part"""
        with pytest.raises(ValueError, match="Invalid style format: /viridis"):
            parse_style("/viridis")

    def test_invalid_empty_colormap(self):
        """Test parsing with empty colormap part"""
        result = parse_style("raster/")
        assert result == (Style.RASTER, "")

    def test_valid_whitespace_in_colormap(self):
        """Test parsing colormap with whitespace"""
        result = parse_style("raster/color map")
        assert result == (Style.RASTER, "color map")

    def test_return_types(self):
        """Test that function returns correct types"""
        result = parse_style("raster/viridis")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], Style)
        assert isinstance(result[1], str)

    def test_all_valid_enum_values(self):
        """Test all valid Style enum values"""
        test_cases = [
            ("raster", Style.RASTER),
            ("quiver", Style.QUIVER),
            ("numpy", Style.NUMPY),
            ("vector", Style.VECTOR),
        ]

        for style_str, expected_enum in test_cases:
            result = parse_style(f"{style_str}/test_colormap")
            assert result[0] == expected_enum
            assert result[1] == "test_colormap"
