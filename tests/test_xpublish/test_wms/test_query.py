import pytest

from xpublish_tiles.xpublish.wms.query import (
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
        tile="1,1,1",
        width=100,
        height=100,
        colorscalerange="0,100",
        autoscale=True,
    )
    assert isinstance(getmap_query.root, WMSGetMapQuery)
    assert getmap_query.root.colorscalerange == (0, 100)
    assert getmap_query.root.styles == ("raster", "magma")

    getmap_query_autoscale = WMSQuery(
        service="WMS",
        version="1.3.0",
        request="GetMap",
        layers="layer1",
        styles="raster/default",
        crs="EPSG:3857",
        tile="1,1,1",
        width=100,
        height=100,
        autoscale=True,
    )
    assert isinstance(getmap_query_autoscale.root, WMSGetMapQuery)
    assert getmap_query_autoscale.root.colorscalerange is None

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
            autoscale=True,
        )

    # Fail because colorscalerange is missing
    with pytest.raises(
        ValueError,
        match="colorscalerange is required when autoscale is False",
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
            autoscale=False,
        )

    with pytest.raises(ValueError, match="bbox or tile must be specified"):
        WMSQuery(
            service="WMS",
            version="1.3.0",
            request="GetMap",
            layers="layer1",
            styles="raster/default",
            crs="EPSG:3857",
            width=100,
            height=100,
            colorscalerange="0,100",
            autoscale=True,
        )

    # Fail because tile and crs are not valid
    with pytest.raises(ValueError, match="tile is only supported for EPSG:3857"):
        WMSQuery(
            service="WMS",
            version="1.3.0",
            request="GetMap",
            layers="layer1",
            styles="raster/default",
            crs="EPSG:4326",
            tile="1,1,1",
            width=100,
            height=100,
            colorscalerange="0,100",
            autoscale=True,
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
            autoscale=True,
        )

    # Fail because style is not valid
    with pytest.raises(
        ValueError,
        match="style must be in the format 'stylename/palettename'",
    ):
        WMSQuery(
            service="WMS",
            version="1.3.0",
            request="GetMap",
            layers="layer1",
            styles="raster",
            crs="EPSG:3857",
            tile="1,1,1",
            width=100,
            height=100,
            colorscalerange="0,100",
            autoscale=True,
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

    getlegendgraphic_query = WMSQuery(
        service="WMS",
        version="1.3.0",
        request="GetLegendGraphic",
        layers="layer1",
        width=100,
        height=100,
        vertical=True,
        colorscalerange="0,100",
        autoscale=True,
        styles="raster/default",
    )
    assert isinstance(getlegendgraphic_query.root, WMSGetLegendGraphicQuery)
