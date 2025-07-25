"""OGC Tiles API data models"""

import re
from enum import Enum
from typing import Any, Optional, Union

import pyproj
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

    def to_pyproj_crs(self) -> Optional[pyproj.CRS]:
        """Convert CRS to pyproj.CRS object for coordinate transformations

        Returns:
            pyproj.CRS object if conversion successful, None otherwise

        Raises:
            pyproj.exceptions.CRSError: If CRS string is invalid
        """
        epsg_string = self.to_epsg_string()
        if epsg_string is None:
            return None

        try:
            return pyproj.CRS.from_user_input(epsg_string)
        except Exception:
            # If pyproj can't parse the CRS string, return None
            # This allows the caller to handle the error appropriately
            return None


class Link(BaseModel):
    """A link to another resource"""

    href: str
    rel: str
    type: Optional[str] = None
    title: Optional[str] = None
    templated: Optional[bool] = None
    varBase: Optional[str] = None
    hreflang: Optional[str] = None
    length: Optional[int] = None


class ConformanceDeclaration(BaseModel):
    """OGC API conformance declaration"""

    conformsTo: list[str]


class BoundingBox(BaseModel):
    """Bounding box definition"""

    lowerLeft: list[float]  # [minX, minY]
    upperRight: list[float]  # [maxX, maxY]
    crs: Optional[Union[str, CRSType]] = None
    orderedAxes: Optional[list[str]] = None


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


class DataType(str, Enum):
    """Valid data types as defined in OGC Tiles specification"""

    MAP = "map"
    VECTOR = "vector"
    COVERAGE = "coverage"


class TileSetMetadata(BaseModel):
    """Metadata for a tileset applied to a specific dataset"""

    title: Optional[str] = None
    tileMatrixSetURI: str
    crs: Union[str, CRSType]
    dataType: Union[DataType, str]  # "map", "vector", "coverage"
    links: list[Link]
    boundingBox: Optional[BoundingBox] = None


class TileMatrixSetLimit(BaseModel):
    """Limits for a specific tile matrix"""

    tileMatrix: str
    minTileRow: int
    maxTileRow: int
    minTileCol: int
    maxTileCol: int


class Style(BaseModel):
    """Style definition"""

    id: str
    title: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[list[str]] = None
    links: Optional[list[Link]] = None


class PropertySchema(BaseModel):
    """Schema definition for a property"""

    title: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    enum: Optional[list[str]] = None
    format: Optional[str] = None
    contentMediaType: Optional[str] = None
    maximum: Optional[float] = None
    exclusiveMaximum: Optional[float] = None
    minimum: Optional[float] = None
    exclusiveMinimum: Optional[float] = None
    pattern: Optional[str] = None
    maxItems: Optional[int] = None
    minItems: Optional[int] = None
    observedProperty: Optional[str] = None
    observedPropertyURI: Optional[str] = None
    uom: Optional[str] = None
    uomURI: Optional[str] = None


class DimensionType(str, Enum):
    """Types of dimensions supported"""

    TEMPORAL = "temporal"
    VERTICAL = "vertical"
    CUSTOM = "custom"


class DimensionExtent(BaseModel):
    """Extent information for a dimension"""

    name: str
    type: DimensionType
    extent: list[Union[str, float, int]]  # [min, max] or list of discrete values
    values: Optional[list[Union[str, float, int]]] = None  # Available discrete values
    units: Optional[str] = None
    description: Optional[str] = None
    default: Optional[Union[str, float, int]] = None


class Layer(BaseModel):
    """Layer definition within a tileset"""

    id: str
    title: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[str] = None
    dataType: Optional[Union[DataType, str]] = None
    geometryDimension: Optional[int] = None
    featureType: Optional[str] = None
    attribution: Optional[str] = None
    license: Optional[str] = None
    pointOfContact: Optional[str] = None
    publisher: Optional[str] = None
    theme: Optional[str] = None
    crs: Optional[Union[str, CRSType]] = None
    epoch: Optional[float] = None
    minScaleDenominator: Optional[float] = None
    maxScaleDenominator: Optional[float] = None
    minCellSize: Optional[float] = None
    maxCellSize: Optional[float] = None
    maxTileMatrix: Optional[str] = None
    minTileMatrix: Optional[str] = None
    boundingBox: Optional[BoundingBox] = None
    created: Optional[str] = None
    updated: Optional[str] = None
    style: Optional[Style] = None
    geoDataClasses: Optional[list[str]] = None
    propertiesSchema: Optional[dict[str, PropertySchema]] = None
    dimensions: Optional[list[DimensionExtent]] = None
    links: Optional[list[Link]] = None


class CenterPoint(BaseModel):
    """Center point definition"""

    coordinates: list[float]
    crs: Optional[Union[str, CRSType]] = None
    tileMatrix: Optional[str] = None
    scaleDenominator: Optional[float] = None
    cellSize: Optional[float] = None


class TilesetSummary(BaseModel):
    """Summary of a tileset in a tilesets list"""

    title: Optional[str] = None
    description: Optional[str] = None
    dataType: Union[DataType, str]  # "map", "vector", "coverage"
    crs: Union[str, CRSType]
    tileMatrixSetURI: Optional[str] = None
    links: list[Link]
    tileMatrixSetLimits: Optional[list[TileMatrixSetLimit]] = None
    epoch: Optional[float] = None
    layers: Optional[list[Layer]] = None
    boundingBox: Optional[BoundingBox] = None
    centerPoint: Optional[CenterPoint] = None
    style: Optional[Style] = None
    attribution: Optional[str] = None
    license: Optional[str] = None
    accessConstraints: Optional[str] = None
    keywords: Optional[list[str]] = None
    version: Optional[str] = None
    created: Optional[str] = None
    updated: Optional[str] = None
    pointOfContact: Optional[str] = None
    mediaTypes: Optional[list[str]] = None


class TilesetsList(BaseModel):
    """List of available tilesets"""

    tilesets: list[TilesetSummary]
    links: Optional[list[Link]] = None


class TilesLandingPage(BaseModel):
    """Landing page for a dataset's tiles"""

    title: str
    description: Optional[str] = None
    links: list[Link]
