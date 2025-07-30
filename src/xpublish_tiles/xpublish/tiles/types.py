"""OGC Tiles API data models"""

import re
from enum import Enum
from typing import Annotated, Any, Optional, Union

import pyproj
from pydantic import BaseModel, Field, field_validator


class MD_ReferenceSystem(BaseModel):
    """ISO 19115 MD_ReferenceSystem data structure"""

    code: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Alphanumeric value identifying an instance in the namespace",
            }
        ),
    ] = None
    codeSpace: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Name or identifier of the person or organization responsible for namespace",
            }
        ),
    ] = None
    version: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Identifier of the version of the associated codeSpace or code",
            }
        ),
    ] = None


class CRSType(BaseModel):
    """CRS definition supporting URI, WKT2, or ISO 19115 MD_ReferenceSystem"""

    uri: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "A reference to a CRS, typically EPSG",
            }
        ),
    ] = None
    wkt: Annotated[
        Optional[Any],
        Field(
            json_schema_extra={
                "description": "WKT2 CRS definition",
            }
        ),
    ] = None
    referenceSystem: Annotated[
        Optional[MD_ReferenceSystem],
        Field(
            json_schema_extra={
                "description": "ISO 19115 reference system",
            }
        ),
    ] = None

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

    href: Annotated[
        str,
        Field(
            json_schema_extra={
                "description": "The URI of the linked resource",
            }
        ),
    ]
    rel: Annotated[
        str,
        Field(
            json_schema_extra={
                "description": "The relationship type of the linked resource",
            }
        ),
    ]
    type: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "The media type of the linked resource",
            }
        ),
    ] = None
    title: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "A human-readable title for the link",
            }
        ),
    ] = None
    templated: Annotated[
        Optional[bool],
        Field(
            json_schema_extra={
                "description": "Whether the href is a URI template",
            }
        ),
    ] = None
    varBase: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Base URI for template variable resolution",
            }
        ),
    ] = None
    hreflang: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Language of the linked resource",
            }
        ),
    ] = None
    length: Annotated[
        Optional[int],
        Field(
            json_schema_extra={
                "description": "Length of the linked resource in bytes",
            }
        ),
    ] = None


class ConformanceDeclaration(BaseModel):
    """OGC API conformance declaration"""

    conformsTo: Annotated[
        list[str],
        Field(
            json_schema_extra={
                "description": "List of conformance class URIs that this API conforms to",
            }
        ),
    ]


class BoundingBox(BaseModel):
    """Bounding box definition"""

    lowerLeft: Annotated[
        list[float],
        Field(
            json_schema_extra={
                "description": "Lower left corner coordinates [minX, minY]",
            }
        ),
    ]
    upperRight: Annotated[
        list[float],
        Field(
            json_schema_extra={
                "description": "Upper right corner coordinates [maxX, maxY]",
            }
        ),
    ]
    crs: Annotated[
        Optional[Union[str, CRSType]],
        Field(
            json_schema_extra={
                "description": "Coordinate reference system of the bounding box",
            }
        ),
    ] = None
    orderedAxes: Annotated[
        Optional[list[str]],
        Field(
            json_schema_extra={
                "description": "Ordered list of axis names for the CRS",
            }
        ),
    ] = None


class TileMatrix(BaseModel):
    """Definition of a tile matrix within a tile matrix set"""

    id: Annotated[
        str,
        Field(
            json_schema_extra={
                "description": "Identifier for this tile matrix",
            }
        ),
    ]
    scaleDenominator: Annotated[
        float,
        Field(
            json_schema_extra={
                "description": "Scale denominator for this tile matrix level",
            }
        ),
    ]
    topLeftCorner: Annotated[
        list[float],
        Field(
            json_schema_extra={
                "description": "Top-left corner coordinates of the tile matrix",
            }
        ),
    ]
    tileWidth: Annotated[
        int,
        Field(
            json_schema_extra={
                "description": "Width of each tile in pixels",
            }
        ),
    ]
    tileHeight: Annotated[
        int,
        Field(
            json_schema_extra={
                "description": "Height of each tile in pixels",
            }
        ),
    ]
    matrixWidth: Annotated[
        int,
        Field(
            json_schema_extra={
                "description": "Number of tiles in the horizontal direction",
            }
        ),
    ]
    matrixHeight: Annotated[
        int,
        Field(
            json_schema_extra={
                "description": "Number of tiles in the vertical direction",
            }
        ),
    ]


class TileMatrixSet(BaseModel):
    """Complete tile matrix set definition"""

    id: Annotated[
        str,
        Field(
            json_schema_extra={
                "description": "Identifier for this tile matrix set",
            }
        ),
    ]
    title: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Human-readable title for this tile matrix set",
            }
        ),
    ] = None
    uri: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "URI identifier for this tile matrix set",
            }
        ),
    ] = None
    crs: Annotated[
        Union[str, CRSType],
        Field(
            json_schema_extra={
                "description": "Coordinate reference system used by this tile matrix set",
            }
        ),
    ]
    tileMatrices: Annotated[
        list[TileMatrix],
        Field(
            json_schema_extra={
                "description": "List of tile matrices in this set",
            }
        ),
    ]


class TileMatrixSetSummary(BaseModel):
    """Summary of a tile matrix set for listings"""

    id: Annotated[
        str,
        Field(
            json_schema_extra={
                "description": "Identifier for this tile matrix set",
            }
        ),
    ]
    title: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Human-readable title for this tile matrix set",
            }
        ),
    ] = None
    uri: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "URI identifier for this tile matrix set",
            }
        ),
    ] = None
    crs: Annotated[
        Union[str, CRSType],
        Field(
            json_schema_extra={
                "description": "Coordinate reference system used by this tile matrix set",
            }
        ),
    ]
    links: Annotated[
        list[Link],
        Field(
            json_schema_extra={
                "description": "Links related to this tile matrix set",
            }
        ),
    ]


class TileMatrixSets(BaseModel):
    """Collection of tile matrix sets"""

    tileMatrixSets: Annotated[
        list[TileMatrixSetSummary],
        Field(
            json_schema_extra={
                "description": "List of available tile matrix sets",
            }
        ),
    ]


class DataType(str, Enum):
    """Valid data types as defined in OGC Tiles specification"""

    MAP = "map"
    VECTOR = "vector"
    COVERAGE = "coverage"


class TileSetMetadata(BaseModel):
    """Metadata for a tileset applied to a specific dataset"""

    title: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Human-readable title for this tileset",
            }
        ),
    ] = None
    tileMatrixSetURI: Annotated[
        str,
        Field(
            json_schema_extra={
                "description": "URI of the tile matrix set used by this tileset",
            }
        ),
    ]
    crs: Annotated[
        Union[str, CRSType],
        Field(
            json_schema_extra={
                "description": "Coordinate reference system used by this tileset",
            }
        ),
    ]
    dataType: Annotated[
        Union[DataType, str],
        Field(
            json_schema_extra={
                "description": "Type of data contained in the tiles (map, vector, coverage)",
            }
        ),
    ]
    links: Annotated[
        list[Link],
        Field(
            json_schema_extra={
                "description": "Links related to this tileset",
            }
        ),
    ]
    boundingBox: Annotated[
        Optional[BoundingBox],
        Field(
            json_schema_extra={
                "description": "Bounding box of the tileset data",
            }
        ),
    ] = None


class TileMatrixSetLimit(BaseModel):
    """Limits for a specific tile matrix"""

    tileMatrix: Annotated[
        str,
        Field(
            json_schema_extra={
                "description": "Identifier of the tile matrix these limits apply to",
            }
        ),
    ]
    minTileRow: Annotated[
        int,
        Field(
            json_schema_extra={
                "description": "Minimum tile row index",
            }
        ),
    ]
    maxTileRow: Annotated[
        int,
        Field(
            json_schema_extra={
                "description": "Maximum tile row index",
            }
        ),
    ]
    minTileCol: Annotated[
        int,
        Field(
            json_schema_extra={
                "description": "Minimum tile column index",
            }
        ),
    ]
    maxTileCol: Annotated[
        int,
        Field(
            json_schema_extra={
                "description": "Maximum tile column index",
            }
        ),
    ]


class Style(BaseModel):
    """Style definition"""

    id: Annotated[
        str,
        Field(
            json_schema_extra={
                "description": "Identifier for this style",
            }
        ),
    ]
    title: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Human-readable title for this style",
            }
        ),
    ] = None
    description: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Brief narrative description of this style",
            }
        ),
    ] = None
    keywords: Annotated[
        Optional[list[str]],
        Field(
            json_schema_extra={
                "description": "Keywords associated with this style",
            }
        ),
    ] = None
    links: Annotated[
        Optional[list[Link]],
        Field(
            json_schema_extra={
                "description": "Links related to this style",
            }
        ),
    ] = None


class PropertySchema(BaseModel):
    """Schema definition for a property"""

    title: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Human-readable title for this property",
            }
        ),
    ] = None
    description: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Description of this property",
            }
        ),
    ] = None
    type: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Data type of this property",
            }
        ),
    ] = None
    enum: Annotated[
        Optional[list[str]],
        Field(
            json_schema_extra={
                "description": "List of valid enumerated values",
            }
        ),
    ] = None
    format: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Format specification for this property",
            }
        ),
    ] = None
    contentMediaType: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Media type of the property content",
            }
        ),
    ] = None
    maximum: Annotated[
        Optional[float],
        Field(
            json_schema_extra={
                "description": "Maximum allowed value (inclusive)",
            }
        ),
    ] = None
    exclusiveMaximum: Annotated[
        Optional[float],
        Field(
            json_schema_extra={
                "description": "Maximum allowed value (exclusive)",
            }
        ),
    ] = None
    minimum: Annotated[
        Optional[float],
        Field(
            json_schema_extra={
                "description": "Minimum allowed value (inclusive)",
            }
        ),
    ] = None
    exclusiveMinimum: Annotated[
        Optional[float],
        Field(
            json_schema_extra={
                "description": "Minimum allowed value (exclusive)",
            }
        ),
    ] = None
    pattern: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Regular expression pattern for validation",
            }
        ),
    ] = None
    maxItems: Annotated[
        Optional[int],
        Field(
            json_schema_extra={
                "description": "Maximum number of items in array",
            }
        ),
    ] = None
    minItems: Annotated[
        Optional[int],
        Field(
            json_schema_extra={
                "description": "Minimum number of items in array",
            }
        ),
    ] = None
    observedProperty: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Name of the observed property",
            }
        ),
    ] = None
    observedPropertyURI: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "URI of the observed property definition",
            }
        ),
    ] = None
    uom: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Unit of measurement",
            }
        ),
    ] = None
    uomURI: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "URI of the unit of measurement definition",
            }
        ),
    ] = None


class DimensionType(str, Enum):
    """Types of dimensions supported"""

    TEMPORAL = "temporal"
    VERTICAL = "vertical"
    CUSTOM = "custom"


class DimensionExtent(BaseModel):
    """Extent information for a dimension"""

    name: Annotated[
        str,
        Field(
            json_schema_extra={
                "description": "Name of the dimension",
            }
        ),
    ]
    type: Annotated[
        DimensionType,
        Field(
            json_schema_extra={
                "description": "Type of dimension (temporal, vertical, or custom)",
            }
        ),
    ]
    extent: Annotated[
        list[Union[str, float, int]],
        Field(
            json_schema_extra={
                "description": "Extent as [min, max] or list of discrete values",
            }
        ),
    ]
    values: Annotated[
        Optional[list[Union[str, float, int]]],
        Field(
            json_schema_extra={
                "description": "Available discrete values for this dimension",
            }
        ),
    ] = None
    units: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Units of measurement for this dimension",
            }
        ),
    ] = None
    description: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Description of this dimension",
            }
        ),
    ] = None
    default: Annotated[
        Optional[Union[str, float, int]],
        Field(
            json_schema_extra={
                "description": "Default value for this dimension",
            }
        ),
    ] = None


class Layer(BaseModel):
    """Layer definition within a tileset"""

    id: Annotated[
        str,
        Field(
            json_schema_extra={
                "description": "Identifier for this layer",
            }
        ),
    ]
    title: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Human-readable title for this layer",
            }
        ),
    ] = None
    description: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Brief narrative description of this layer",
            }
        ),
    ] = None
    keywords: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Keywords associated with this layer",
            }
        ),
    ] = None
    dataType: Annotated[
        Optional[Union[DataType, str]],
        Field(
            json_schema_extra={
                "description": "Type of data in this layer (map, vector, coverage)",
            }
        ),
    ] = None
    geometryDimension: Annotated[
        Optional[int],
        Field(
            json_schema_extra={
                "description": "Dimension of the geometry (0=point, 1=line, 2=polygon, 3=volume)",
            }
        ),
    ] = None
    featureType: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Type of features in this layer",
            }
        ),
    ] = None
    attribution: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Attribution text for this layer",
            }
        ),
    ] = None
    license: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "License information for this layer",
            }
        ),
    ] = None
    pointOfContact: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Contact information for this layer",
            }
        ),
    ] = None
    publisher: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Publisher of this layer",
            }
        ),
    ] = None
    theme: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Theme or category of this layer",
            }
        ),
    ] = None
    crs: Annotated[
        Optional[Union[str, CRSType]],
        Field(
            json_schema_extra={
                "description": "Coordinate reference system for this layer",
            }
        ),
    ] = None
    epoch: Annotated[
        Optional[float],
        Field(
            json_schema_extra={
                "description": "Epoch for coordinate reference system",
            }
        ),
    ] = None
    minScaleDenominator: Annotated[
        Optional[float],
        Field(
            json_schema_extra={
                "description": "Minimum scale denominator for this layer",
            }
        ),
    ] = None
    maxScaleDenominator: Annotated[
        Optional[float],
        Field(
            json_schema_extra={
                "description": "Maximum scale denominator for this layer",
            }
        ),
    ] = None
    minCellSize: Annotated[
        Optional[float],
        Field(
            json_schema_extra={
                "description": "Minimum cell size for this layer",
            }
        ),
    ] = None
    maxCellSize: Annotated[
        Optional[float],
        Field(
            json_schema_extra={
                "description": "Maximum cell size for this layer",
            }
        ),
    ] = None
    maxTileMatrix: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Maximum tile matrix identifier for this layer",
            }
        ),
    ] = None
    minTileMatrix: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Minimum tile matrix identifier for this layer",
            }
        ),
    ] = None
    boundingBox: Annotated[
        Optional[BoundingBox],
        Field(
            json_schema_extra={
                "description": "Bounding box of this layer",
            }
        ),
    ] = None
    created: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Creation date of this layer",
            }
        ),
    ] = None
    updated: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Last update date of this layer",
            }
        ),
    ] = None
    style: Annotated[
        Optional[Style],
        Field(
            json_schema_extra={
                "description": "Default style for this layer",
            }
        ),
    ] = None
    geoDataClasses: Annotated[
        Optional[list[str]],
        Field(
            json_schema_extra={
                "description": "Geographic data classes for this layer",
            }
        ),
    ] = None
    propertiesSchema: Annotated[
        Optional[dict[str, PropertySchema]],
        Field(
            json_schema_extra={
                "description": "Schema definitions for layer properties",
            }
        ),
    ] = None
    dimensions: Annotated[
        Optional[list[DimensionExtent]],
        Field(
            json_schema_extra={
                "description": "Available dimensions for this layer",
            }
        ),
    ] = None
    links: Annotated[
        Optional[list[Link]],
        Field(
            json_schema_extra={
                "description": "Links related to this layer",
            }
        ),
    ] = None


class CenterPoint(BaseModel):
    """Center point definition"""

    coordinates: Annotated[
        list[float],
        Field(
            json_schema_extra={
                "description": "Coordinates of the center point",
            }
        ),
    ]
    crs: Annotated[
        Optional[Union[str, CRSType]],
        Field(
            json_schema_extra={
                "description": "Coordinate reference system for the center point",
            }
        ),
    ] = None
    tileMatrix: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Tile matrix identifier for the center point",
            }
        ),
    ] = None
    scaleDenominator: Annotated[
        Optional[float],
        Field(
            json_schema_extra={
                "description": "Scale denominator at the center point",
            }
        ),
    ] = None
    cellSize: Annotated[
        Optional[float],
        Field(
            json_schema_extra={
                "description": "Cell size at the center point",
            }
        ),
    ] = None


class TilesetSummary(BaseModel):
    """Summary of a tileset in a tilesets list"""

    title: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Human-readable title for this tileset",
            }
        ),
    ] = None
    description: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Brief narrative description of this tileset",
            }
        ),
    ] = None
    dataType: Annotated[
        Union[DataType, str],
        Field(
            json_schema_extra={
                "description": "Type of data contained in the tiles (map, vector, coverage)",
            }
        ),
    ]
    crs: Annotated[
        Union[str, CRSType],
        Field(
            json_schema_extra={
                "description": "Coordinate reference system used by this tileset",
            }
        ),
    ]
    tileMatrixSetURI: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "URI of the tile matrix set used by this tileset",
            }
        ),
    ] = None
    links: Annotated[
        list[Link],
        Field(
            json_schema_extra={
                "description": "Links related to this tileset",
            }
        ),
    ]
    tileMatrixSetLimits: Annotated[
        Optional[list[TileMatrixSetLimit]],
        Field(
            json_schema_extra={
                "description": "Limits for tile matrices in this tileset",
            }
        ),
    ] = None
    epoch: Annotated[
        Optional[float],
        Field(
            json_schema_extra={
                "description": "Epoch for coordinate reference system",
            }
        ),
    ] = None
    layers: Annotated[
        Optional[list[Layer]],
        Field(
            json_schema_extra={
                "description": "Layers contained in this tileset",
            }
        ),
    ] = None
    boundingBox: Annotated[
        Optional[BoundingBox],
        Field(
            json_schema_extra={
                "description": "Bounding box of the tileset data",
            }
        ),
    ] = None
    centerPoint: Annotated[
        Optional[CenterPoint],
        Field(
            json_schema_extra={
                "description": "Center point of the tileset",
            }
        ),
    ] = None
    style: Annotated[
        Optional[Style],
        Field(
            json_schema_extra={
                "description": "Default style for this tileset",
            }
        ),
    ] = None
    attribution: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Attribution text for this tileset",
            }
        ),
    ] = None
    license: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "License information for this tileset",
            }
        ),
    ] = None
    accessConstraints: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Access constraints for this tileset",
            }
        ),
    ] = None
    keywords: Annotated[
        Optional[list[str]],
        Field(
            json_schema_extra={
                "description": "Keywords associated with this tileset",
            }
        ),
    ] = None
    version: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Version of this tileset",
            }
        ),
    ] = None
    created: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Creation date of this tileset",
            }
        ),
    ] = None
    updated: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Last update date of this tileset",
            }
        ),
    ] = None
    pointOfContact: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Contact information for this tileset",
            }
        ),
    ] = None
    mediaTypes: Annotated[
        Optional[list[str]],
        Field(
            json_schema_extra={
                "description": "Supported media types for this tileset",
            }
        ),
    ] = None


class TilesetsList(BaseModel):
    """List of available tilesets"""

    tilesets: Annotated[
        list[TilesetSummary],
        Field(
            json_schema_extra={
                "description": "List of available tilesets",
            }
        ),
    ]
    links: Annotated[
        Optional[list[Link]],
        Field(
            json_schema_extra={
                "description": "Links related to this tilesets collection",
            }
        ),
    ] = None


class TilesLandingPage(BaseModel):
    """Landing page for a dataset's tiles"""

    title: Annotated[
        str,
        Field(
            json_schema_extra={
                "description": "Title of the tiles landing page",
            }
        ),
    ]
    description: Annotated[
        Optional[str],
        Field(
            json_schema_extra={
                "description": "Description of the tiles service",
            }
        ),
    ] = None
    links: Annotated[
        list[Link],
        Field(
            json_schema_extra={
                "description": "Links to tiles resources and metadata",
            }
        ),
    ]
