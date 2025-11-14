from typing import Any, Literal, Union, overload

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_validator,
)
from pydantic_xml import BaseXmlModel, attr, element
from pyproj import CRS
from pyproj.aoi import BBox

from xpublish_tiles.types import ImageFormat
from xpublish_tiles.validators import (
    validate_bbox,
    validate_colormap,
    validate_colorscalerange,
    validate_crs,
    validate_image_format,
    validate_style,
)


class WMSBaseQuery(BaseModel):
    service: Literal["WMS"] = Field(..., description="Service type. Must be WMS")
    version: Literal["1.1.1", "1.3.0"] = Field(
        "1.3.0",
        description="Version of the WMS service",
    )


class WMSGetCapabilitiesQuery(WMSBaseQuery):
    """WMS GetCapabilities query"""

    request: Literal["GetCapabilities"] = Field(..., description="Request type")


class WMSGetMapQuery(WMSBaseQuery):
    """WMS GetMap query"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    request: Literal["GetMap"] = Field(..., description="Request type")
    layers: str = Field(
        validation_alias=AliasChoices("layername", "layers", "query_layers"),
    )
    styles: tuple[str, str] | None = Field(
        ("raster", "default"),
        description="Style to use for the query. Defaults to raster/default. Default may be replaced by the name of any colormap available to matplotlibs",
    )
    crs: CRS = Field(
        CRS.from_epsg(4326),
        description="Coordinate reference system to use for the query. Default is EPSG:4326",
    )
    time: str | None = Field(
        None,
        description="Optional time to get map for in Y-m-dTH:M:SZ format. Only valid when the layer has a time dimension. When not specified, the default time is used",
    )
    elevation: str | None = Field(
        None,
        description="Optional elevation to get map for. Only valid when the layer has an elevation dimension. When not specified, the default elevation is used",
    )
    bbox: BBox = Field(
        ...,
        description="Bounding box to use for the query in the format 'minx,miny,maxx,maxy'",
    )
    width: int = Field(
        ...,
        description="The width of the image to return in pixels",
    )
    height: int = Field(
        ...,
        description="The height of the image to return in pixels",
    )
    colorscalerange: tuple[float, float] | None = Field(
        None,
        description="Color scale range to use for the query in the format 'min,max'. If not specified, the default color scale range is used or if none is available it is autoscaled",
    )
    colormap: dict[str, str] | None = Field(
        None,
        description="Custom colormap as JSON-encoded dictionary with numeric keys (0-255) and hex color values (#RRGGBB). When provided, overrides any colormap from the styles parameter.",
    )
    format: ImageFormat = Field(
        ImageFormat.PNG,
        description="The format of the image to return",
    )

    @field_validator("colorscalerange", mode="before")
    @classmethod
    def validate_colorscalerange(cls, v: str | None) -> tuple[float, float] | None:
        return validate_colorscalerange(v)

    @field_validator("colormap", mode="before")
    @classmethod
    def validate_colormap(cls, v: str | dict | None) -> dict[str, str] | None:
        return validate_colormap(v)

    @field_validator("bbox", mode="before")
    @classmethod
    def validate_bbox(cls, v: str | None) -> BBox | None:
        return validate_bbox(v)

    @field_validator("styles", mode="before")
    @classmethod
    def validate_style(cls, v: str | None) -> tuple[str, str] | None:
        valid_style = validate_style(v)
        if valid_style is None:
            return ("raster", "default")
        return valid_style

    @field_validator("crs", mode="before")
    @classmethod
    def validate_crs(cls, v: str | None) -> CRS | None:
        return validate_crs(v)

    @field_validator("format", mode="before")
    @classmethod
    def validate_format(cls, v: str | None) -> ImageFormat | None:
        return validate_image_format(v)

    @model_validator(mode="after")
    def validate_colormap_requires_custom_style(self):
        """Validate that colormap requires style to be raster/custom."""
        if self.colormap is not None:
            if self.styles is None or self.styles != ("raster", "custom"):
                style_str = (
                    f"{self.styles[0]}/{self.styles[1]}" if self.styles else "None"
                )
                raise ValueError(
                    f"When 'colormap' parameter is provided, 'styles' must be 'raster/custom'. Got styles='{style_str}' instead."
                )
        return self


class WMSGetFeatureInfoQuery(WMSBaseQuery):
    """WMS GetFeatureInfo query"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    request: Literal["GetFeatureInfo", "GetTimeseries", "GetVerticalProfile"] = Field(
        ...,
        description="Request type",
    )
    query_layers: str = Field(
        validation_alias=AliasChoices("layername", "layers", "query_layers"),
    )
    time: str | None = Field(
        None,
        description="Optional time to get feature info for in Y-m-dTH:M:SZ format. Only valid when the layer has a time dimension. To get a range of times, use 'start/end'",
    )
    elevation: str | None = Field(
        None,
        description="Optional elevation to get feature info for. Only valid when the layer has an elevation dimension. To get all elevations, use 'all', to get a range of elevations, use 'start/end'",
    )
    crs: CRS = Field(
        "EPSG:4326",
        description="Coordinate reference system to use for the query. Currently only EPSG:4326 is supported for this request",
    )
    bbox: BBox = Field(
        ...,
        description="Bounding box to use for the query in the format 'minx,miny,maxx,maxy'",
    )
    width: int = Field(
        ...,
        description="Width of the image to query against. This is the number of points between minx and maxx",
    )
    height: int = Field(
        ...,
        description="Height of the image to query against. This is the number of points between miny and maxy",
    )
    x: int = Field(
        ...,
        description="The x coordinate of the point to query. This is the index of the point in the x dimension",
    )
    y: int = Field(
        ...,
        description="The y coordinate of the point to query. This is the index of the point in the y dimension",
    )

    @field_validator("bbox", mode="before")
    @classmethod
    def validate_bbox(cls, v: str | None) -> BBox | None:
        return validate_bbox(v)

    @field_validator("crs", mode="before")
    @classmethod
    def validate_crs(cls, v: str | None) -> CRS | None:
        return validate_crs(v)


class WMSGetLegendGraphicQuery(WMSBaseQuery):
    """WMS GetLegendGraphic query"""

    request: Literal["GetLegendGraphic"] = Field(..., description="Request type")
    layer: str
    width: int = 100
    height: int = 100
    vertical: bool = False
    colorscalerange: tuple[float, float] | None = Field(
        None,
        description="Color scale range to use for the query in the format 'min,max'. If not provided, the default will be used or autoscaled if no default is available",
    )
    colormap: dict[str, str] | None = Field(
        None,
        description="Custom colormap as JSON-encoded dictionary with numeric keys (0-255) and hex color values (#RRGGBB). When provided, overrides any colormap from the styles parameter.",
    )
    styles: tuple[str, str] = Field(
        ("raster", "default"),
        description="Style to use for the query. Defaults to raster/default. Default may be replaced by the name of any colormap defined by matplotlibs defaults",
    )
    format: ImageFormat = Field(
        "image/png",
        description="Format to use for the query. Defaults to image/png",
    )

    @field_validator("colorscalerange", mode="before")
    @classmethod
    def validate_colorscalerange(cls, v: str | None) -> tuple[float, float] | None:
        return validate_colorscalerange(v)

    @field_validator("colormap", mode="before")
    @classmethod
    def validate_colormap(cls, v: str | dict | None) -> dict[str, str] | None:
        return validate_colormap(v)

    @field_validator("styles", mode="before")
    @classmethod
    def validate_style(cls, v: str | None) -> tuple[str, str] | None:
        return validate_style(v)

    @field_validator("format", mode="before")
    @classmethod
    def validate_format(cls, v: str | None) -> ImageFormat | None:
        return validate_image_format(v)

    @model_validator(mode="after")
    def validate_colormap_requires_custom_style(self):
        """Validate that colormap requires style to be raster/custom."""
        if self.colormap is not None:
            if self.styles is None or self.styles != ("raster", "custom"):
                style_str = (
                    f"{self.styles[0]}/{self.styles[1]}" if self.styles else "None"
                )
                raise ValueError(
                    f"When 'colormap' parameter is provided, 'styles' must be 'raster/custom'. Got styles='{style_str}' instead."
                )
        return self


WMSQueryType = Union[
    WMSGetCapabilitiesQuery,
    WMSGetMapQuery,
    WMSGetFeatureInfoQuery,
    WMSGetLegendGraphicQuery,
]


class WMSQuery(RootModel):
    root: WMSQueryType = Field(discriminator="request")

    @overload
    def __init__(
        self,
        *,
        service: Literal["WMS"],
        version: Literal["1.1.1", "1.3.0"],
        request: Literal["GetCapabilities"],
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        service: Literal["WMS"],
        version: Literal["1.1.1", "1.3.0"],
        request: Literal["GetMap"],
        layers: str,
        width: int,
        height: int,
        bbox: str | BBox | None = None,
        styles: str | tuple[str, str] = ("raster", "default"),
        crs: Literal["EPSG:4326", "EPSG:3857"] = "EPSG:4326",
        time: str | None = None,
        elevation: str | None = None,
        colorscalerange: str | tuple[float, float] | None = None,
        autoscale: bool = False,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        service: Literal["WMS"],
        version: Literal["1.1.1", "1.3.0"],
        request: Literal["GetFeatureInfo"],
        query_layers: str,
        bbox: str | BBox,
        width: int,
        height: int,
        x: int,
        y: int,
        crs: Literal["EPSG:4326"] = "EPSG:4326",
        time: str | None = None,
        elevation: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        service: Literal["WMS"],
        version: Literal["1.1.1", "1.3.0"],
        request: Literal["GetLegendGraphic"],
        layer: str,
        width: int,
        height: int,
        colorscalerange: str | tuple[float, float],
        vertical: bool = False,
        autoscale: bool = False,
        styles: str | tuple[str, str] = ("raster", "default"),
    ) -> None: ...

    def __init__(self, **data: Any) -> None:
        super().__init__(data)

    @model_validator(mode="before")
    def lower_case_dict(cls, values: Any) -> Any:
        if isinstance(values, dict):
            ret_dict = dict()
            for k, v in values.items():
                ret_k = k.lower()
                ret_v = v

                if isinstance(ret_v, str):
                    if ret_k == "item":
                        ret_v = ret_v.lower()
                    elif ret_k == "crs":
                        ret_v = ret_v.upper()

                ret_dict[ret_k] = ret_v
            return ret_dict
        return values


# These params are used for GetMap and GetFeatureInfo requests, and can be filtered out of the query params for any requests that are handled
WMS_FILTERED_QUERY_PARAMS = {
    "service",
    "version",
    "request",
    "layers",
    "layer",
    "layername",
    "query_layers",
    "styles",
    "crs",
    "time",
    "elevation",
    "bbox",
    "width",
    "height",
    "colorscalerange",
    "colormap",
    "autoscale",
    "vertical",
    "item",
    "day",
    "range",
    "x",
    "y",
}


class WMSOnlineResourceResponse(
    BaseXmlModel, tag="OnlineResource", nsmap={"xlink": "http://www.w3.org/1999/xlink"}
):
    href: str = attr(name="xlink:href")
    type: str = attr(name="xlink:type", default="simple")


class WMSContactInformationResponse(BaseXmlModel, tag="ContactInformation"):
    contact_person_primary: str | None = element(tag="ContactPersonPrimary", default=None)
    contact_position: str | None = element(tag="ContactPosition", default=None)
    contact_address: str | None = element(tag="ContactAddress", default=None)
    contact_voice_telephone: str | None = element(
        tag="ContactVoiceTelephone", default=None
    )
    contact_electronic_mail_address: str | None = element(
        tag="ContactElectronicMailAddress", default=None
    )


class WMSServiceResponse(BaseXmlModel, tag="Service"):
    name: str = element(tag="Name")
    title: str = element(tag="Title")
    abstract: str | None = element(tag="Abstract", default=None)
    keyword_list: list[str] | None = element(tag="KeywordList", default=None)
    online_resource: WMSOnlineResourceResponse = element(tag="OnlineResource")
    contact_information: WMSContactInformationResponse | None = element(
        tag="ContactInformation", default=None
    )
    fees: str | None = element(tag="Fees", default="none")
    access_constraints: str | None = element(tag="AccessConstraints", default="none")


class WMSBoundingBoxResponse(BaseXmlModel, tag="BoundingBox"):
    crs: str = attr(name="CRS")
    minx: float = attr(name="minx")
    miny: float = attr(name="miny")
    maxx: float = attr(name="maxx")
    maxy: float = attr(name="maxy")


class WMSGeographicBoundingBoxResponse(BaseXmlModel, tag="EX_GeographicBoundingBox"):
    west_bound_longitude: float = element(tag="westBoundLongitude")
    east_bound_longitude: float = element(tag="eastBoundLongitude")
    south_bound_latitude: float = element(tag="southBoundLatitude")
    north_bound_latitude: float = element(tag="northBoundLatitude")


class WMSDimensionResponse(BaseXmlModel, tag="Dimension"):
    name: str = attr(name="name")
    units: str = attr(name="units")
    unit_symbol: str | None = attr(name="unitSymbol", default=None)
    default: str | None = attr(name="default", default=None)
    multiple_values: bool = attr(name="multipleValues", default=False)
    nearest_value: bool = attr(name="nearestValue", default=False)
    current: bool = attr(name="current", default=False)
    values: str


class WMSStyleResponse(BaseXmlModel, tag="Style"):
    name: str = element(tag="Name")
    title: str = element(tag="Title")
    abstract: str | None = element(tag="Abstract", default=None)
    legend_url: WMSOnlineResourceResponse | None = element(tag="LegendURL", default=None)


class WMSAttributeResponse(BaseXmlModel, tag="Attribute"):
    """Custom attribute element for WMS layer metadata"""

    name: str = attr(name="name")
    value: str = attr(name="value")


class WMSLayerResponse(BaseXmlModel, tag="Layer"):
    queryable: bool = attr(name="queryable", default=True)
    cascaded: int | None = attr(name="cascaded", default=None)
    opaque: bool = attr(name="opaque", default=False)
    no_subsets: bool = attr(name="noSubsets", default=False)
    fixed_width: int | None = attr(name="fixedWidth", default=None)
    fixed_height: int | None = attr(name="fixedHeight", default=None)

    name: str | None = element(tag="Name", default=None)
    title: str = element(tag="Title")
    abstract: str | None = element(tag="Abstract", default=None)
    keyword_list: list[str] | None = element(tag="KeywordList", default=None)
    crs: list[str] | None = element(tag="CRS", default=None)
    ex_geographic_bounding_box: WMSGeographicBoundingBoxResponse | None = element(
        tag="EX_GeographicBoundingBox",
        default=None,
    )
    bounding_box: list[WMSBoundingBoxResponse] | None = element(
        tag="BoundingBox", default=[]
    )
    dimensions: list[WMSDimensionResponse] = element(tag="Dimension", default=[])
    attribution: str | None = element(tag="Attribution", default=None)
    authority_url: WMSOnlineResourceResponse | None = element(
        tag="AuthorityURL", default=None
    )
    identifier: str | None = element(tag="Identifier", default=None)
    metadata_url: WMSOnlineResourceResponse | None = element(
        tag="MetadataURL", default=None
    )
    data_url: WMSOnlineResourceResponse | None = element(tag="DataURL", default=None)
    feature_list_url: WMSOnlineResourceResponse | None = element(
        tag="FeatureListURL", default=None
    )
    styles: list[WMSStyleResponse] = element(tag="Style", default=[])
    min_scale_denominator: float | None = element(tag="MinScaleDenominator", default=None)
    max_scale_denominator: float | None = element(tag="MaxScaleDenominator", default=None)
    attributes: list[WMSAttributeResponse] = element(tag="Attribute", default=[])
    layers: list["WMSLayerResponse"] = element(tag="Layer", default=[])


class WMSFormatResponse(BaseXmlModel, tag="Format"):
    format: str


class WMSHTTPResponse(BaseXmlModel, tag="HTTP"):
    get: WMSOnlineResourceResponse = element(tag="Get")


class WMSDCPTypeResponse(BaseXmlModel, tag="DCPType"):
    http: WMSHTTPResponse = element(tag="HTTP")


class WMSOperationResponse(BaseXmlModel):
    formats: list[WMSFormatResponse] = element(tag="Format")
    dcp_type: WMSDCPTypeResponse = element(tag="DCPType")


class WMSGetCapabilitiesOperationResponse(WMSOperationResponse, tag="GetCapabilities"):
    pass


class WMSGetMapOperationResponse(WMSOperationResponse, tag="GetMap"):
    pass


class WMSGetFeatureInfoOperationResponse(WMSOperationResponse, tag="GetFeatureInfo"):
    pass


class WMSRequestResponse(BaseXmlModel, tag="Request"):
    get_capabilities: WMSGetCapabilitiesOperationResponse = element(tag="GetCapabilities")
    get_map: WMSGetMapOperationResponse = element(tag="GetMap")
    get_feature_info: WMSGetFeatureInfoOperationResponse | None = element(
        tag="GetFeatureInfo", default=None
    )


class WMSCapabilityResponse(BaseXmlModel, tag="Capability"):
    request: WMSRequestResponse = element(tag="Request")
    exception: list[str] = element(tag="Exception")
    layer: WMSLayerResponse = element(tag="Layer")


class WMSCapabilitiesResponse(
    BaseXmlModel,
    tag="WMS_Capabilities",
    nsmap={"": "http://www.opengis.net/wms", "xlink": "http://www.w3.org/1999/xlink"},
):
    version: str = attr(name="version")
    update_sequence: str | None = attr(name="updateSequence", default=None)

    service: WMSServiceResponse = element(tag="Service")
    capability: WMSCapabilityResponse = element(tag="Capability")
