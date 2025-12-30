import pytest
from pyproj import CRS

from xpublish_tiles.lib import create_listed_colormap_from_dict
from xpublish_tiles.types import ImageFormat
from xpublish_tiles.validators import (
    validate_colormap,
    validate_colorscalerange,
    validate_crs,
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
            ValueError, match=r"image format gif is not valid. Options are: PNG, JPEG"
        ):
            validate_image_format("gif")

    def test_invalid_format_with_mime_type(self):
        with pytest.raises(
            ValueError, match=r"image format gif is not valid. Options are: PNG, JPEG"
        ):
            validate_image_format("image/gif")


class TestValidateStyle:
    def test_valid_raster_style(self):
        result = validate_style("raster/default")
        assert result == ("raster", "default")

    def test_valid_raster_style_with_colormap(self):
        result = validate_style("raster/viridis")
        assert result == ("raster", "viridis")

    def test_valid_raster_style_custom(self):
        result = validate_style("raster/custom")
        assert result == ("raster", "custom")

    @pytest.mark.skip()
    def test_valid_quiver_style(self):
        result = validate_style("quiver/arrows")
        assert result == ("quiver", "arrows")

    @pytest.mark.skip()
    def test_valid_quiver_style_default(self):
        result = validate_style("quiver/default")
        assert result == ("quiver", "default")

    def test_valid_style_lowercase(self):
        result = validate_style("raster/default")
        assert result == ("raster", "default")

    def test_valid_style_mixed_case(self):
        result = validate_style("RaStEr/default")
        assert result == ("raster", "default")

    def test_none_input(self):
        result = validate_style(None)
        assert result is None

    def test_empty_string(self):
        result = validate_style("")
        assert result is None

    def test_invalid_format_single_value(self):
        with pytest.raises(
            ValueError,
            match=r"style must be in the format 'stylename/palettename'. A common default for this is 'raster/default'",
        ):
            validate_style("raster")

    def test_invalid_format_three_values(self):
        with pytest.raises(
            ValueError,
            match=r"style must be in the format 'stylename/palettename'. A common default for this is 'raster/default'",
        ):
            validate_style("raster/default/extra")

    def test_invalid_style_name(self):
        with pytest.raises(
            ValueError,
            match=r"style 'invalid' is not valid. Available styles are:",
        ):
            validate_style("invalid/default")

    def test_invalid_variant_for_raster(self):
        with pytest.raises(
            ValueError,
            match="variant 'invalid_variant' is not supported for style 'raster'",
        ):
            validate_style("raster/invalid_variant")

    @pytest.mark.skip()
    def test_invalid_variant_for_quiver(self):
        with pytest.raises(
            ValueError,
            match="variant 'invalid_variant' is not supported for style 'quiver'",
        ):
            validate_style("quiver/invalid_variant")


class TestValidateCrs:
    def test_valid_epsg_code(self):
        result = validate_crs("EPSG:4326")
        assert isinstance(result, CRS)
        assert result.to_epsg() == 4326

    def test_valid_epsg_code_numeric(self):
        result = validate_crs("4326")
        assert isinstance(result, CRS)
        assert result.to_epsg() == 4326

    def test_valid_proj_string(self):
        result = validate_crs("+proj=longlat +datum=WGS84 +no_defs")
        assert isinstance(result, CRS)
        assert result.to_epsg() == 4326

    def test_valid_wkt_string(self):
        wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
        result = validate_crs(wkt)
        assert isinstance(result, CRS)
        assert result.to_epsg() == 4326

    def test_none_input(self):
        result = validate_crs(None)
        assert result is None

    def test_invalid_crs_string(self):
        with pytest.raises(ValueError, match="crs invalid_crs is not valid"):
            validate_crs("invalid_crs")

    def test_invalid_epsg_code(self):
        with pytest.raises(ValueError, match="crs EPSG:999999 is not valid"):
            validate_crs("EPSG:999999")


class TestValidateColormap:
    def test_none_input(self):
        result = validate_colormap(None)
        assert result is None

    def test_valid_dict_input(self):
        colormap = {"0": "#ffffff", "255": "#000000"}
        result = validate_colormap(colormap)
        assert result == {"0": "#ffffff", "255": "#000000"}

    def test_valid_json_string_input(self):
        colormap_json = '{"0": "#ffffff", "128": "#808080", "255": "#000000"}'
        result = validate_colormap(colormap_json)
        assert result == {"0": "#ffffff", "128": "#808080", "255": "#000000"}

    def test_numeric_keys_converted_to_strings(self):
        colormap = {0: "#ffffff", 255: "#000000"}
        result = validate_colormap(colormap)
        assert result == {"0": "#ffffff", "255": "#000000"}

    def test_valid_range_0_to_255(self):
        colormap = {"0": "#ffffff", "100": "#808080", "255": "#000000"}
        result = validate_colormap(colormap)
        assert result == {"0": "#ffffff", "100": "#808080", "255": "#000000"}

    def test_invalid_json_string(self):
        with pytest.raises(
            ValueError, match="colormap must be a valid JSON-encoded dictionary"
        ):
            validate_colormap('{"invalid": json}')

    def test_invalid_key_out_of_range_high(self):
        with pytest.raises(
            ValueError, match="colormap keys must be integers between 0 and 255, got 256"
        ):
            validate_colormap({"256": "#ffffff"})

    def test_invalid_key_out_of_range_low(self):
        with pytest.raises(
            ValueError, match="colormap keys must be integers between 0 and 255, got -1"
        ):
            validate_colormap({"-1": "#ffffff"})

    def test_invalid_key_range(self):
        with pytest.raises(
            ValueError,
            match="colormap keys must include 0 and 255 as minimum and maximum",
        ):
            validate_colormap({"0": "#000000", "1": "#123123"})

        with pytest.raises(
            ValueError,
            match="colormap keys must include 0 and 255 as minimum and maximum",
        ):
            validate_colormap({"1": "#000000", "255": "#123123"})

        with pytest.raises(
            ValueError,
            match="colormap keys must include 0 and 255 as minimum and maximum",
        ):
            validate_colormap({"1": "#000000", "254": "#123123"})

    def test_invalid_key_non_numeric(self):
        with pytest.raises(ValueError, match="colormap keys must be numeric, got 'abc'"):
            validate_colormap({"abc": "#ffffff"})

    def test_invalid_value_non_string(self):
        with pytest.raises(
            ValueError, match="colormap values must be strings, got int for key 0"
        ):
            validate_colormap({"0": 255})

    def test_invalid_color_format(self):
        with pytest.raises(
            ValueError,
            match="colormap value 'invalid' for key 0 must be a hex color \\(#RRGGBB\\)",
        ):
            validate_colormap({"0": "invalid"})

    def test_invalid_named_colors(self):
        with pytest.raises(
            ValueError,
            match="colormap value 'white' for key 0 must be a hex color \\(#RRGGBB\\)",
        ):
            validate_colormap({"0": "white", "255": "black"})

    def test_valid_hex_colors(self):
        colormap = {"0": "#FFFFFF", "255": "#000000"}
        result = validate_colormap(colormap)
        assert result == {"0": "#FFFFFF", "255": "#000000"}

    def test_non_dict_json_input(self):
        with pytest.raises(ValueError, match="colormap must be a dictionary"):
            validate_colormap(["not", "a", "dict"])  # type: ignore  # this doesn't validate with the type checker


class TestCategoricalColormap:
    def test_create_listed_colormap_valid(self):
        """Test creating a color_key dictionary with valid categorical colormap."""
        colormap_dict = {
            "0": "#ff0000",
            "1": "#00ff00",
            "2": "#0000ff",
        }
        flag_values = [0, 1, 2]
        color_key = create_listed_colormap_from_dict(colormap_dict, flag_values)
        assert len(color_key) == 3
        assert color_key == {0: "#ff0000", 1: "#00ff00", 2: "#0000ff"}

    def test_create_listed_colormap_missing_flag_value(self):
        """Test that missing flag_value raises error."""
        colormap_dict = {
            "0": "#ff0000",
            "1": "#00ff00",
        }
        flag_values = [0, 1, 2, 3, 4]
        with pytest.raises(
            ValueError,
            match="colormap is missing entries for flag_values",
        ):
            create_listed_colormap_from_dict(colormap_dict, flag_values)

    def test_create_listed_colormap_invalid_key(self):
        """Test that keys not in flag_values raise error."""
        colormap_dict = {
            "0": "#ff0000",
            "1": "#00ff00",
            "99": "#0000ff",  # 99 is not in flag_values
        }
        flag_values = [0, 1, 2]
        with pytest.raises(
            ValueError,
            match="colormap contains keys not in flag_values: \\['99'\\]",
        ):
            create_listed_colormap_from_dict(colormap_dict, flag_values)

    def test_create_listed_colormap_non_consecutive_flag_values(self):
        """Test that non-consecutive flag_values work correctly."""
        # flag_values don't have to be consecutive
        colormap_dict = {
            "0": "#ff0000",
            "5": "#00ff00",  # skipping 1-4
            "10": "#0000ff",
        }
        flag_values = [0, 5, 10]
        color_key = create_listed_colormap_from_dict(colormap_dict, flag_values)
        assert len(color_key) == 3
        assert color_key == {0: "#ff0000", 5: "#00ff00", 10: "#0000ff"}
