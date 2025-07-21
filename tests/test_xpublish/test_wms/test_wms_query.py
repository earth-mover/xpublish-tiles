import pytest

from xpublish_tiles.types import ImageFormat
from xpublish_tiles.xpublish.wms.types import (
    WMSGetCapabilitiesQuery,
    WMSGetFeatureInfoQuery,
    WMSGetLegendGraphicQuery,
    WMSGetMapQuery,
    WMSQuery,
)


def test_wms_query_discriminator():
    WMSGetCapabilitiesQuery(service="WMS", version="1.3.0", request="GetCapabilities")

    getcaps_query = WMSQuery(
        service="WMS",
        version="1.3.0",
        request="GetCapabilities",
    )
    assert isinstance(getcaps_query.root, WMSGetCapabilitiesQuery)

    getmap_query = WMSQuery(
        service="WMS",
        version="1.3.0",
        request="GetMap",
        layers="layer1",
        styles="raster/magma",
        crs="EPSG:3857",
        bbox="0,0,1,1",
        width=100,
        height=100,
        colorscalerange="0,100",
    )
    assert isinstance(getmap_query.root, WMSGetMapQuery)
    assert getmap_query.root.colorscalerange == (0, 100)
    assert getmap_query.root.styles == ("raster", "magma")
    assert getmap_query.root.crs.to_epsg() == 3857
    assert getmap_query.root.format == ImageFormat.PNG

    getmap_query_autoscale = WMSQuery(
        service="WMS",
        version="1.3.0",
        request="GetMap",
        layers="layer1",
        styles="raster/default",
        crs="EPSG:3857",
        bbox="0,0,1,1",
        width=100,
        height=100,
    )
    assert isinstance(getmap_query_autoscale.root, WMSGetMapQuery)
    assert getmap_query_autoscale.root.colorscalerange is None
    assert getmap_query_autoscale.root.format == ImageFormat.PNG

    # Fail because colorscalerange is invalid
    with pytest.raises(
        ValueError,
        match="colorscalerange must be in the format 'min,max'",
    ):
        WMSQuery(
            service="WMS",
            version="1.3.0",
            request="GetMap",
            layers="layer1",
            styles="raster/default",
            crs="EPSG:3857",
            bbox="0,0,1,1",
            width=100,
            height=100,
            colorscalerange="0",
        )

    # Fail because bbox is not valid
    with pytest.raises(
        ValueError,
        match="bbox must be in the format 'minx,miny,maxx,maxy'",
    ):
        WMSQuery(
            service="WMS",
            version="1.3.0",
            request="GetMap",
            layers="layer1",
            styles="raster/default",
            crs="EPSG:3857",
            bbox="0,0,1",
            width=100,
            height=100,
            colorscalerange="0,100",
        )

    getfeatureinfo_query = WMSQuery(
        service="WMS",
        version="1.3.0",
        request="GetFeatureInfo",
        query_layers="layer1",
        time="2020-01-01",
        elevation="100",
        crs="EPSG:4326",
        bbox="0,0,1,1",
        width=100,
        height=100,
        x=50,
        y=50,
    )
    assert isinstance(getfeatureinfo_query.root, WMSGetFeatureInfoQuery)
    assert getfeatureinfo_query.root.crs.to_epsg() == 4326

    getlegendgraphic_query = WMSQuery(
        service="WMS",
        version="1.3.0",
        request="GetLegendGraphic",
        layer="layer1",
        width=100,
        height=100,
        vertical=True,
        colorscalerange="0,100",
        styles="raster/default",
    )
    assert isinstance(getlegendgraphic_query.root, WMSGetLegendGraphicQuery)
