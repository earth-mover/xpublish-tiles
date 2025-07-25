"""Tests for OGC Tiles API data models"""

import pytest
from pydantic import ValidationError

from xpublish_tiles.xpublish.tiles.models import CRSType, MD_ReferenceSystem, BoundingBox


class TestCRSType:
    """Test CRSType model and conversion functionality"""

    def test_uri_crs(self):
        """Test CRS with URI reference"""
        crs = CRSType(uri="http://www.opengis.net/def/crs/EPSG/0/4326")
        assert crs.uri == "http://www.opengis.net/def/crs/EPSG/0/4326"
        assert crs.wkt is None
        assert crs.referenceSystem is None

    def test_wkt_crs(self):
        """Test CRS with WKT definition"""
        wkt_string = 'GEOGCS["WGS 84",DATUM["WGS_1984"]]'
        crs = CRSType(wkt=wkt_string)
        assert crs.wkt == wkt_string
        assert crs.uri is None
        assert crs.referenceSystem is None

    def test_reference_system_crs(self):
        """Test CRS with ISO 19115 MD_ReferenceSystem"""
        ref_sys = MD_ReferenceSystem(code="4326", codeSpace="EPSG", version="8.5")
        crs = CRSType(referenceSystem=ref_sys)
        assert crs.referenceSystem == ref_sys
        assert crs.uri is None
        assert crs.wkt is None

    def test_empty_crs(self):
        """Test CRS with no values"""
        crs = CRSType()
        assert crs.uri is None
        assert crs.wkt is None
        assert crs.referenceSystem is None

    def test_uri_validation_error(self):
        """Test URI validation with invalid type"""
        with pytest.raises(ValidationError):
            CRSType(uri=123)  # type: ignore[arg-type]

    def test_to_epsg_string_from_uri_epsg_colon(self):
        """Test EPSG string conversion from URI with colon format"""
        crs = CRSType(uri="http://www.opengis.net/def/crs/EPSG/0/4326")
        assert crs.to_epsg_string() == "EPSG:4326"

    def test_to_epsg_string_from_uri_epsg_slash(self):
        """Test EPSG string conversion from URI with slash format"""
        crs = CRSType(uri="epsg/3857")
        assert crs.to_epsg_string() == "EPSG:3857"

    def test_to_epsg_string_from_uri_epsg_case_insensitive(self):
        """Test EPSG string conversion is case insensitive"""
        crs = CRSType(uri="EPSG:4326")
        assert crs.to_epsg_string() == "EPSG:4326"

    def test_to_epsg_string_from_uri_non_epsg(self):
        """Test EPSG string conversion from non-EPSG URI"""
        crs = CRSType(uri="http://example.com/crs/custom")
        assert crs.to_epsg_string() == "http://example.com/crs/custom"

    def test_to_epsg_string_from_wkt(self):
        """Test EPSG string conversion from WKT"""
        wkt_string = 'GEOGCS["WGS 84",DATUM["WGS_1984"]]'
        crs = CRSType(wkt=wkt_string)
        assert crs.to_epsg_string() == wkt_string

    def test_to_epsg_string_from_reference_system_epsg(self):
        """Test EPSG string conversion from MD_ReferenceSystem with EPSG"""
        ref_sys = MD_ReferenceSystem(code="4326", codeSpace="EPSG")
        crs = CRSType(referenceSystem=ref_sys)
        assert crs.to_epsg_string() == "EPSG:4326"

    def test_to_epsg_string_from_reference_system_non_epsg(self):
        """Test EPSG string conversion from MD_ReferenceSystem without EPSG"""
        ref_sys = MD_ReferenceSystem(code="4326", codeSpace="OTHER")
        crs = CRSType(referenceSystem=ref_sys)
        assert crs.to_epsg_string() == "4326"

    def test_to_epsg_string_from_reference_system_no_code(self):
        """Test EPSG string conversion from MD_ReferenceSystem without code"""
        ref_sys = MD_ReferenceSystem(codeSpace="EPSG")
        crs = CRSType(referenceSystem=ref_sys)
        assert crs.to_epsg_string() is None

    def test_to_epsg_string_empty(self):
        """Test EPSG string conversion from empty CRS"""
        crs = CRSType()
        assert crs.to_epsg_string() is None


class TestMD_ReferenceSystem:
    """Test MD_ReferenceSystem model"""

    def test_full_reference_system(self):
        """Test complete MD_ReferenceSystem"""
        ref_sys = MD_ReferenceSystem(code="4326", codeSpace="EPSG", version="8.5")
        assert ref_sys.code == "4326"
        assert ref_sys.codeSpace == "EPSG"
        assert ref_sys.version == "8.5"

    def test_minimal_reference_system(self):
        """Test minimal MD_ReferenceSystem"""
        ref_sys = MD_ReferenceSystem()
        assert ref_sys.code is None
        assert ref_sys.codeSpace is None
        assert ref_sys.version is None


class TestBoundingBoxWithCRS:
    """Test BoundingBox with new CRS types"""

    def test_bounding_box_with_string_crs(self):
        """Test BoundingBox with string CRS (backward compatibility)"""
        bbox = BoundingBox(
            lowerLeft=[-180.0, -90.0], upperRight=[180.0, 90.0], crs="EPSG:4326"
        )
        assert bbox.crs == "EPSG:4326"

    def test_bounding_box_with_crs_type(self):
        """Test BoundingBox with CRSType"""
        crs_type = CRSType(uri="http://www.opengis.net/def/crs/EPSG/0/4326")
        bbox = BoundingBox(
            lowerLeft=[-180.0, -90.0], upperRight=[180.0, 90.0], crs=crs_type
        )
        assert bbox.crs == crs_type

    def test_bounding_box_no_crs(self):
        """Test BoundingBox without CRS"""
        bbox = BoundingBox(lowerLeft=[-180.0, -90.0], upperRight=[180.0, 90.0])
        assert bbox.crs is None
