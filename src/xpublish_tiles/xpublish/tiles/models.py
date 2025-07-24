"""OGC Tiles API data models"""

from typing import List, Optional
from pydantic import BaseModel


class Link(BaseModel):
    """A link to another resource"""

    href: str
    rel: str
    type: Optional[str] = None
    title: Optional[str] = None


class ConformanceDeclaration(BaseModel):
    """OGC API conformance declaration"""

    conformsTo: List[str]


class BoundingBox(BaseModel):
    """Bounding box definition"""

    lowerLeft: List[float]  # [minX, minY]
    upperRight: List[float]  # [maxX, maxY]
    crs: Optional[str] = None


class TileMatrix(BaseModel):
    """Definition of a tile matrix within a tile matrix set"""

    id: str
    scaleDenominator: float
    topLeftCorner: List[float]
    tileWidth: int
    tileHeight: int
    matrixWidth: int
    matrixHeight: int


class TileMatrixSet(BaseModel):
    """Complete tile matrix set definition"""

    id: str
    title: Optional[str] = None
    uri: Optional[str] = None
    crs: str
    tileMatrices: List[TileMatrix]


class TileMatrixSetSummary(BaseModel):
    """Summary of a tile matrix set for listings"""

    id: str
    title: Optional[str] = None
    uri: Optional[str] = None
    crs: str
    links: List[Link]


class TileMatrixSets(BaseModel):
    """Collection of tile matrix sets"""

    tileMatrixSets: List[TileMatrixSetSummary]


class TileSetMetadata(BaseModel):
    """Metadata for a tileset applied to a specific dataset"""

    title: Optional[str] = None
    tileMatrixSetURI: str
    crs: str
    dataType: str  # "map", "vector", "coverage"
    links: List[Link]
    boundingBox: Optional[BoundingBox] = None


class TilesLandingPage(BaseModel):
    """Landing page for a dataset's tiles"""

    title: str
    description: Optional[str] = None
    links: List[Link]
