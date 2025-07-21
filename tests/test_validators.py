import pytest
from pyproj import CRS

from xpublish_tiles.types import ImageFormat
from xpublish_tiles.validators import (
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
        assert result == ("raster", "default")

    def test_valid_raster_style_with_colormap(self):
        result = validate_style("raster/viridis")
        assert result == ("raster", "viridis")

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
            match="style must be in the format 'stylename/palettename'. A common default for this is 'raster/default'",
        ):
            validate_style("raster")

    def test_invalid_format_three_values(self):
        with pytest.raises(
            ValueError,
            match="style must be in the format 'stylename/palettename'. A common default for this is 'raster/default'",
        ):
            validate_style("raster/default/extra")

    def test_invalid_style_name(self):
        with pytest.raises(
            ValueError,
            match="style 'invalid' is not valid. Available styles are:",
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
