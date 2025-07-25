"""OGC Tiles API data models"""

import re
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, field_validator


class MD_ReferenceSystem(BaseModel):
    """ISO 19115 MD_ReferenceSystem data structure"""

    code: Optional[str] = None
    codeSpace: Optional[str] = None
    version: Optional[str] = None


class CRSType(BaseModel):
    """CRS definition supporting URI, WKT2, or ISO 19115 MD_ReferenceSystem"""

    uri: Optional[str] = Field(None, description="A reference to a CRS, typically EPSG")
    wkt: Optional[Any] = Field(None, description="WKT2 CRS definition")
    referenceSystem: Optional[MD_ReferenceSystem] = Field(
        None, description="ISO 19115 reference system"
    )

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, v):
        if v is not None and not isinstance(v, str):
            raise ValueError("URI must be a string")
        return v

    def to_epsg_string(self) -> Optional[str]:
        """Convert CRS to EPSG string format for pyproj"""
        if self.uri:
            # Handle OGC URI format: http://www.opengis.net/def/crs/EPSG/0/4326
            ogc_match = re.search(r"/epsg/\d+/(\d+)$", self.uri.lower())
            if ogc_match:
                return f"EPSG:{ogc_match.group(1)}"
            # Handle simple EPSG format: epsg:4326 or epsg/4326
            epsg_match = re.search(r"epsg[:/](\d+)", self.uri.lower())
            if epsg_match:
                return f"EPSG:{epsg_match.group(1)}"
            return self.uri
        elif self.wkt:
            return str(self.wkt)
        elif self.referenceSystem and self.referenceSystem.code:
            if (
                self.referenceSystem.codeSpace
                and "epsg" in self.referenceSystem.codeSpace.lower()
            ):
                return f"EPSG:{self.referenceSystem.code}"
            return self.referenceSystem.code
        return None


class Link(BaseModel):
    """A link to another resource"""

    href: str
    rel: str
    type: Optional[str] = None
    title: Optional[str] = None


class ConformanceDeclaration(BaseModel):
    """OGC API conformance declaration"""

    conformsTo: list[str]


class BoundingBox(BaseModel):
    """Bounding box definition"""

    lowerLeft: list[float]  # [minX, minY]
    upperRight: list[float]  # [maxX, maxY]
    crs: Optional[Union[str, CRSType]] = None


class TileMatrix(BaseModel):
    """Definition of a tile matrix within a tile matrix set"""

    id: str
    scaleDenominator: float
    topLeftCorner: list[float]
    tileWidth: int
    tileHeight: int
    matrixWidth: int
    matrixHeight: int


class TileMatrixSet(BaseModel):
    """Complete tile matrix set definition"""

    id: str
    title: Optional[str] = None
    uri: Optional[str] = None
    crs: Union[str, CRSType]
    tileMatrices: list[TileMatrix]


class TileMatrixSetSummary(BaseModel):
    """Summary of a tile matrix set for listings"""

    id: str
    title: Optional[str] = None
    uri: Optional[str] = None
    crs: Union[str, CRSType]
    links: list[Link]


class TileMatrixSets(BaseModel):
    """Collection of tile matrix sets"""

    tileMatrixSets: list[TileMatrixSetSummary]


class TileSetMetadata(BaseModel):
    """Metadata for a tileset applied to a specific dataset"""

    title: Optional[str] = None
    tileMatrixSetURI: str
    crs: Union[str, CRSType]
    dataType: str  # "map", "vector", "coverage"
    links: list[Link]
    boundingBox: Optional[BoundingBox] = None


class TilesetSummary(BaseModel):
    """Summary of a tileset in a tilesets list"""

    title: Optional[str] = None
    tileMatrixSetURI: Optional[str] = None
    crs: Union[str, CRSType]
    dataType: str  # "map", "vector", "coverage"
    links: list[Link]


class TilesetsList(BaseModel):
    """List of available tilesets"""

    tilesets: list[TilesetSummary]
    links: Optional[list[Link]] = None


class TilesLandingPage(BaseModel):
    """Landing page for a dataset's tiles"""

    title: str
    description: Optional[str] = None
    links: list[Link]
