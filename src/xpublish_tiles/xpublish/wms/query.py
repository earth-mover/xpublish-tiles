from typing import Any, Literal, Optional, Union, cast

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    RootModel,
    field_validator,
    model_validator,
)


def validate_colorscalerange(v: str | None) -> tuple[float, float] | None:
    if v is None:
        return None

    values = v.split(",")
    if len(values) != 2:
        raise ValueError("colorscalerange must be in the format 'min,max'")

    try:
        min_val = float(values[0])
        max_val = float(values[1])
    except ValueError as e:
        raise ValueError(
            "colorscalerange must be in the format 'min,max' where min and max are valid floats",
        ) from e
    return (min_val, max_val)


def validate_bbox(v: str | None) -> tuple[float, float, float, float] | None:
    if v is None:
        return None

    values = v.split(",") if isinstance(v, str) else v
    if len(values) != 4:
        raise ValueError("bbox must be in the format 'minx,miny,maxx,maxy'")

    try:
        bbox = cast(tuple[float, float, float, float], tuple(float(x) for x in values))
    except ValueError as e:
        raise ValueError(
            "bbox must be in the format 'minx,miny,maxx,maxy' where minx, miny, maxx and maxy are valid floats in the provided CRS",
        ) from e

    return bbox


def validate_style(v: str | None) -> tuple[str, str] | None:
    if v is None:
        return None

    values = v.split("/")
    if len(values) != 2:
        raise ValueError(
            "style must be in the format 'stylename/palettename'. A common default for this is 'raster/default'",
        )

    return (values[0], values[1])


class WMSBaseQuery(BaseModel):
    service: Literal["WMS"] = Field(..., description="Service type. Must be WMS")
    version: Literal["1.1.1", "1.3.0"] = Field(
        ...,
        description="Version of the WMS service",
    )


class WMSGetCapabilitiesQuery(WMSBaseQuery):
    """WMS GetCapabilities query"""

    request: Literal["GetCapabilities"] = Field(..., description="Request type")


class WMSGetMapQuery(WMSBaseQuery):
    """WMS GetMap query"""

    request: Literal["GetMap"] = Field(..., description="Request type")
    layers: str = Field(
        validation_alias=AliasChoices("layername", "layers", "query_layers"),
    )
    styles: tuple[str, str] = Field(
        ("raster", "default"),
        description="Style to use for the query. Defaults to raster/default. Default may be replaced by the name of any colormap defined by matplotlibs defaults",
    )
    crs: Literal["EPSG:4326", "EPSG:3857"] = Field(
        "EPSG:4326",
        description="Coordinate reference system to use for the query. EPSG:4326 and EPSG:3857 are supported for this request",
    )
    time: Optional[str] = Field(
        None,
        description="Optional time to get map for in Y-m-dTH:M:SZ format. Only valid when the layer has a time dimension. When not specified, the default time is used",
    )
    elevation: Optional[str] = Field(
        None,
        description="Optional elevation to get map for. Only valid when the layer has an elevation dimension. When not specified, the default elevation is used",
    )
    bbox: Optional[tuple[float, float, float, float]] = Field(
        None,
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
        description="Color scale range to use for the query in the format 'min,max'",
    )
    autoscale: bool = Field(
        False,
        description="Whether to automatically scale the color scale range based on the data. When specified, colorscalerange is ignored",
    )

    @field_validator("colorscalerange", mode="before")
    @classmethod
    def validate_colorscalerange(cls, v: str | None) -> tuple[float, float] | None:
        return validate_colorscalerange(v)

    @field_validator("bbox", mode="before")
    @classmethod
    def validate_bbox(cls, v: str | None) -> tuple[float, float, float, float] | None:
        return validate_bbox(v)

    @field_validator("styles", mode="before")
    @classmethod
    def validate_style(cls, v: str | None) -> tuple[str, str] | None:
        return validate_style(v)

    @model_validator(mode="after")
    @classmethod
    def validate_dependent_colorscalerange(
        cls,
        v: "WMSGetMapQuery",
    ) -> "WMSGetMapQuery":
        if v.colorscalerange is None and not v.autoscale:
            raise ValueError("colorscalerange is required when autoscale is False")
        if v.bbox is None:
            raise ValueError("bbox must be specified")
        return v


class WMSGetFeatureInfoQuery(WMSBaseQuery):
    """WMS GetFeatureInfo query"""

    request: Literal["GetFeatureInfo", "GetTimeseries", "GetVerticalProfile"] = Field(
        ...,
        description="Request type",
    )
    query_layers: str = Field(
        validation_alias=AliasChoices("layername", "layers", "query_layers"),
    )
    time: Optional[str] = Field(
        None,
        description="Optional time to get feature info for in Y-m-dTH:M:SZ format. Only valid when the layer has a time dimension. To get a range of times, use 'start/end'",
    )
    elevation: Optional[str] = Field(
        None,
        description="Optional elevation to get feature info for. Only valid when the layer has an elevation dimension. To get all elevations, use 'all', to get a range of elevations, use 'start/end'",
    )
    crs: Literal["EPSG:4326"] = Field(
        "EPSG:4326",
        description="Coordinate reference system to use for the query. Currently only EPSG:4326 is supported for this request",
    )
    bbox: tuple[float, float, float, float] = Field(
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
    def validate_bbox(cls, v: str | None) -> tuple[float, float, float, float] | None:
        return validate_bbox(v)


class WMSGetLegendGraphicQuery(WMSBaseQuery):
    """WMS GetLegendGraphic query"""

    request: Literal["GetLegendGraphic"] = Field(..., description="Request type")
    layer: str
    width: int
    height: int
    vertical: bool = False
    colorscalerange: tuple[float, float] = Field(
        ...,
        description="Color scale range to use for the query in the format 'min,max'",
    )
    autoscale: bool = False
    styles: tuple[str, str] = Field(
        ("raster", "default"),
        description="Style to use for the query. Defaults to raster/default. Default may be replaced by the name of any colormap defined by matplotlibs defaults",
    )

    @field_validator("colorscalerange", mode="before")
    @classmethod
    def validate_colorscalerange(cls, v: str | None) -> tuple[float, float] | None:
        return validate_colorscalerange(v)

    @field_validator("styles", mode="before")
    @classmethod
    def validate_style(cls, v: str | None) -> tuple[str, str] | None:
        return validate_style(v)


WMSQueryType = Union[
    WMSGetCapabilitiesQuery,
    WMSGetMapQuery,
    WMSGetFeatureInfoQuery,
    WMSGetLegendGraphicQuery,
]


class WMSQuery(RootModel):
    root: WMSQueryType = Field(discriminator="request")

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
    "autoscale",
    "vertical",
    "item",
    "day",
    "range",
    "x",
    "y",
}
