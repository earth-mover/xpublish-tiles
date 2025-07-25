import pytest

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
        
        assert crs == "http://www.opengis.net/def/crs/EPSG/0/3857"        # Zoom 0 should cover the entire Web Mercator extent
        expected_min_x = -20037508.3428
        expected_max_x = 20037508.3428
        expected_min_y = -20037508.3428
        expected_max_y = 20037508.3428

        assert abs(bbox[0] - expected_min_x) < 1  # minX
        assert abs(bbox[1] - expected_min_y) < 1  # minY
        assert abs(bbox[2] - expected_max_x) < 1  # maxX
        assert abs(bbox[3] - expected_max_y) < 1  # maxY

    def test_extract_tile_bbox_zoom_1(self):
        """Test bbox extraction for zoom level 1"""
        # Test top-left tile (0,0)
        bbox, crs = extract_tile_bbox_and_crs("WebMercatorQuad", 1, 0, 0)
        
        assert crs == "http://www.opengis.net/def/crs/EPSG/0/3857"        # At zoom 1, we have 2x2 tiles, each covering half the world extent
        expected_min_x = -20037508.3428
        expected_max_x = 0.0
        expected_min_y = 0.0
        expected_max_y = 20037508.3428

        assert abs(bbox[0] - expected_min_x) < 1
        assert abs(bbox[1] - expected_min_y) < 1
        assert abs(bbox[2] - expected_max_x) < 1
        assert abs(bbox[3] - expected_max_y) < 1

        # Test bottom-right tile (1,1)
        bbox, crs = extract_tile_bbox_and_crs("WebMercatorQuad", 1, 1, 1)
        expected_min_x = 0.0
        expected_max_x = 20037508.3428
        expected_min_y = -20037508.3428
        expected_max_y = 0.0

        assert abs(bbox[0] - expected_min_x) < 1
        assert abs(bbox[1] - expected_min_y) < 1
        assert abs(bbox[2] - expected_max_x) < 1
        assert abs(bbox[3] - expected_max_y) < 1

    def test_extract_tile_bbox_higher_zoom(self):
        """Test bbox extraction for higher zoom level"""
        bbox, crs = extract_tile_bbox_and_crs("WebMercatorQuad", 5, 10, 15)
        
        assert crs == "http://www.opengis.net/def/crs/EPSG/0/3857"        assert len(bbox) == 4

        # Verify bbox format [minX, minY, maxX, maxY]
        assert bbox[0] < bbox[2]  # minX < maxX
        assert bbox[1] < bbox[3]  # minY < maxY

        # At zoom 5, tiles should be much smaller than zoom 0
        tile_width = bbox[2] - bbox[0]
        tile_height = bbox[3] - bbox[1]

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
        assert abs(bbox1[2] - bbox2[0]) < 0.1

        # Y coordinates should be the same for horizontally adjacent tiles
        assert abs(bbox1[1] - bbox2[1]) < 0.1  # minY
        assert abs(bbox1[3] - bbox2[3]) < 0.1  # maxY

    def test_bbox_format(self):
        """Test that bbox is returned in correct format [minX, minY, maxX, maxY]"""
        bbox, crs = extract_tile_bbox_and_crs("WebMercatorQuad", 3, 2, 4)

        assert isinstance(bbox, list)
        assert len(bbox) == 4
        assert isinstance(crs, str)

        # Verify coordinate order
        min_x, min_y, max_x, max_y = bbox
        assert min_x < max_x
        assert min_y < max_y

        # All coordinates should be within Web Mercator bounds
        web_mercator_bound = 20037508.3428
        assert -web_mercator_bound <= min_x <= web_mercator_bound
        assert -web_mercator_bound <= min_y <= web_mercator_bound
        assert -web_mercator_bound <= max_x <= web_mercator_bound
        assert -web_mercator_bound <= max_y <= web_mercator_bound
