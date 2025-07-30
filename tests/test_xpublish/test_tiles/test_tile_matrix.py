import pyproj
import pytest
from pyproj.aoi import BBox

from xpublish_tiles.xpublish.tiles.tile_matrix import (
    extract_tile_bbox_and_crs,
    get_web_mercator_quad,
    get_web_mercator_quad_summary,
)


class TestTileMatrixFunctions:
    def test_get_web_mercator_quad(self):
        """Test WebMercatorQuad tile matrix set creation"""
        tms = get_web_mercator_quad()

        assert tms.id == "WebMercatorQuad"
        assert tms.title == "Web Mercator Quad"
        assert tms.crs == "http://www.opengis.net/def/crs/EPSG/0/3857"
        assert len(tms.tileMatrices) == 22  # Zoom levels 0-21

        # Test zoom level 0
        zoom_0 = tms.tileMatrices[0]
        assert zoom_0.id == "0"
        assert zoom_0.matrixWidth == 1
        assert zoom_0.matrixHeight == 1
        assert zoom_0.tileWidth == 256
        assert zoom_0.tileHeight == 256

        # Test zoom level 10
        zoom_10 = tms.tileMatrices[10]
        assert zoom_10.id == "10"
        assert zoom_10.matrixWidth == 1024  # 2^10
        assert zoom_10.matrixHeight == 1024

    def test_get_web_mercator_quad_summary(self):
        """Test WebMercatorQuad summary creation"""
        summary = get_web_mercator_quad_summary()

        assert summary.id == "WebMercatorQuad"
        assert summary.title == "Web Mercator Quad"
        assert summary.crs == "http://www.opengis.net/def/crs/EPSG/0/3857"
        assert len(summary.links) == 1
        assert summary.links[0].href == "/tiles/tileMatrixSets/WebMercatorQuad"


class TestExtractTileBboxAndCrs:
    def test_extract_tile_bbox_zoom_0(self):
        """Test bbox extraction for zoom level 0 (single tile covering world)"""
        bbox, crs = extract_tile_bbox_and_crs("WebMercatorQuad", 0, 0, 0)

        assert isinstance(crs, pyproj.CRS)
        assert crs.to_epsg() == 3857  # Web Mercator EPSG code
        expected_min_x = -20037508.3428
        expected_max_x = 20037508.3428
        expected_min_y = -20037508.3428
        expected_max_y = 20037508.3428

        assert abs(bbox.west - expected_min_x) < 1  # minX
        assert abs(bbox.south - expected_min_y) < 1  # minY
        assert abs(bbox.east - expected_max_x) < 1  # maxX
        assert abs(bbox.north - expected_max_y) < 1  # maxY

    def test_extract_tile_bbox_zoom_1(self):
        """Test bbox extraction for zoom level 1"""
        # Test top-left tile (0,0)
        bbox, crs = extract_tile_bbox_and_crs("WebMercatorQuad", 1, 0, 0)

        assert isinstance(crs, pyproj.CRS)
        assert (
            crs.to_epsg() == 3857
        )  # At zoom 1, we have 2x2 tiles, each covering half the world extent
        expected_min_x = -20037508.3428
        expected_max_x = 0.0
        expected_min_y = 0.0
        expected_max_y = 20037508.3428

        assert abs(bbox.west - expected_min_x) < 1
        assert abs(bbox.south - expected_min_y) < 1
        assert abs(bbox.east - expected_max_x) < 1
        assert abs(bbox.north - expected_max_y) < 1

        # Test bottom-right tile (1,1)
        bbox, crs = extract_tile_bbox_and_crs("WebMercatorQuad", 1, 1, 1)
        expected_min_x = 0.0
        expected_max_x = 20037508.3428
        expected_min_y = -20037508.3428
        expected_max_y = 0.0

        assert abs(bbox.west - expected_min_x) < 1
        assert abs(bbox.south - expected_min_y) < 1
        assert abs(bbox.east - expected_max_x) < 1
        assert abs(bbox.north - expected_max_y) < 1

    def test_extract_tile_bbox_higher_zoom(self):
        """Test bbox extraction for higher zoom level"""
        bbox, crs = extract_tile_bbox_and_crs("WebMercatorQuad", 5, 10, 15)

        assert isinstance(crs, pyproj.CRS)
        assert crs.to_epsg() == 3857

        # Verify bbox format [minX, minY, maxX, maxY]
        assert bbox.west < bbox.east  # minX < maxX
        assert bbox.south < bbox.north  # minY < maxY

        # At zoom 5, tiles should be much smaller than zoom 0
        tile_width = bbox.east - bbox.west
        tile_height = bbox.north - bbox.south

        # Should be 1/32 of the world extent (2^5 = 32)
        world_extent = 20037508.3428 * 2
        expected_tile_size = world_extent / 32

        assert abs(tile_width - expected_tile_size) < 100
        assert abs(tile_height - expected_tile_size) < 100

    def test_extract_tile_bbox_invalid_matrix_set(self):
        """Test error handling for invalid tile matrix set"""
        with pytest.raises(ValueError, match="Tile matrix set 'InvalidSet' not found"):
            extract_tile_bbox_and_crs("InvalidSet", 0, 0, 0)

    def test_extract_tile_bbox_invalid_zoom_level(self):
        """Test error handling for invalid zoom level"""
        with pytest.raises(ValueError, match="Tile matrix '25' not found"):
            extract_tile_bbox_and_crs("WebMercatorQuad", 25, 0, 0)

    def test_bbox_consistency_across_tiles(self):
        """Test that adjacent tiles have consistent bounding boxes"""
        # Get two adjacent tiles at zoom 2
        bbox1, _ = extract_tile_bbox_and_crs("WebMercatorQuad", 2, 0, 0)
        bbox2, _ = extract_tile_bbox_and_crs("WebMercatorQuad", 2, 0, 1)

        # Adjacent tiles should share a boundary
        # bbox1's maxX should equal bbox2's minX
        assert abs(bbox1.east - bbox2.west) < 0.1

        # Y coordinates should be the same for horizontally adjacent tiles
        assert abs(bbox1.south - bbox2.south) < 0.1  # minY
        assert abs(bbox1.north - bbox2.north) < 0.1  # maxY

    def test_bbox_format(self):
        """Test that bbox is returned in correct format [minX, minY, maxX, maxY]"""
        bbox, crs = extract_tile_bbox_and_crs("WebMercatorQuad", 3, 2, 4)

        assert isinstance(bbox, BBox)
        assert isinstance(crs, pyproj.CRS)

        # Verify coordinate order
        assert bbox.west < bbox.east
        assert bbox.south < bbox.north

        # All coordinates should be within Web Mercator bounds
        web_mercator_bound = 20037508.3428
        assert -web_mercator_bound <= bbox.west <= web_mercator_bound
        assert -web_mercator_bound <= bbox.south <= web_mercator_bound
        assert -web_mercator_bound <= bbox.east <= web_mercator_bound
        assert -web_mercator_bound <= bbox.north <= web_mercator_bound
