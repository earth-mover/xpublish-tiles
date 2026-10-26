import io
import xml.etree.ElementTree as ET

import morecantile
import numpy as np
import pytest
import xpublish
from fastapi.testclient import TestClient
from PIL import Image

import xarray as xr
from xpublish_tiles.config import config
from xpublish_tiles.testing.datasets import (
    CUBED_SPHERE,
    EU3035,
    GEOZARR_MULTISCALE,
    GLOBAL_HEALPIX_CUBE_L3,
    GLOBAL_HEALPIX_L3,
    RGB,
)
from xpublish_tiles.xpublish.wms import WMSPlugin


@pytest.fixture(scope="session")
def xpublish_app(air_dataset):
    rest = xpublish.Rest({"air": air_dataset}, plugins={"wms": WMSPlugin()})
    return rest.app


@pytest.fixture(scope="session")
def xpublish_client(xpublish_app):
    app = xpublish_app
    return TestClient(app)


def test_get_capabilities_xml(xpublish_client):
    """Test GetCapabilities request returns valid XML by default."""
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={"service": "WMS", "version": "1.3.0", "request": "GetCapabilities"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/xml; charset=utf-8"

    # Parse XML to ensure it's valid
    root = ET.fromstring(response.content)
    assert root.tag.endswith("WMS_Capabilities")
    assert root.get("version") == "1.3.0"

    # Check for required elements
    service = root.find(".//{http://www.opengis.net/wms}Service")
    assert service is not None

    capability = root.find(".//{http://www.opengis.net/wms}Capability")
    assert capability is not None

    # Check for layers
    layers = root.findall(".//{http://www.opengis.net/wms}Layer")
    assert len(layers) > 0


def test_get_capabilities_json(xpublish_client):
    """Test GetCapabilities request returns JSON when requested."""
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetCapabilities",
            "format": "json",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    # Parse JSON to ensure it's valid
    data = response.json()
    assert "version" in data
    assert data["version"] == "1.3.0"
    assert "service" in data
    assert "capability" in data

    # Check service information
    service = data["service"]
    assert service["name"] == "WMS"
    assert "title" in service

    # Check capability information
    capability = data["capability"]
    assert "request" in capability
    assert "layer" in capability


def test_get_capabilities_content_negotiation(xpublish_client):
    """Test content negotiation via Accept header."""
    # Test JSON via Accept header
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={"service": "WMS", "version": "1.3.0", "request": "GetCapabilities"},
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    # Test XML via Accept header (should be default anyway)
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={"service": "WMS", "version": "1.3.0", "request": "GetCapabilities"},
        headers={"Accept": "application/xml"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/xml; charset=utf-8"


def test_get_capabilities_layers(xpublish_client):
    """Test that GetCapabilities includes dataset layers."""
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetCapabilities",
            "format": "json",
        },
    )
    assert response.status_code == 200

    data = response.json()
    root_layer = data["capability"]["layer"]

    # Should have child layers for data variables
    assert "layers" in root_layer
    child_layers = root_layer["layers"]
    assert len(child_layers) > 0

    # Check that air variable is included as a layer
    layer_names = [layer["name"] for layer in child_layers if "name" in layer]
    assert "air" in layer_names

    # Check layer properties
    air_layer = next(layer for layer in child_layers if layer.get("name") == "air")
    assert "title" in air_layer
    assert "crs" in air_layer
    assert "EPSG:4326" in air_layer["crs"]
    assert "EPSG:3857" in air_layer["crs"]


def test_get_capabilities_dimensions(xpublish_client):
    """Test that GetCapabilities includes dataset dimensions."""
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetCapabilities",
            "format": "json",
        },
    )
    assert response.status_code == 200

    data = response.json()
    root_layer = data["capability"]["layer"]
    child_layers = root_layer["layers"]

    # Find the air layer
    air_layer = next(layer for layer in child_layers if layer.get("name") == "air")

    # Check for dimensions
    if air_layer.get("dimensions"):
        dimensions = air_layer["dimensions"]
        dimension_names = [dim["name"] for dim in dimensions]

        # The air dataset should have time dimension
        assert "time" in dimension_names

        # Check time dimension properties
        time_dim = next(dim for dim in dimensions if dim["name"] == "time")
        assert "units" in time_dim
        assert "values" in time_dim


def test_app_router(xpublish_client):
    """Test basic WMS routing functionality."""
    # Test GetCapabilities request (now returns actual capabilities)
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={"service": "WMS", "version": "1.3.0", "request": "GetCapabilities"},
    )
    assert response.status_code == 200
    # Should return XML by default
    assert "xml" in response.headers["content-type"]

    # Test GetMap request
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetMap",
            "layers": "air",
            "styles": "raster/magma",
            "crs": "EPSG:3857",
            "bbox": "-8766409.899970, 5009377.085697, -7514065.628546, 6261721.357122",
            "width": 256,
            "height": 256,
            "time": "2013-01-01T00:00:00",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    # Test invalid request (no parameters)
    response = xpublish_client.get("/datasets/air/wms")
    assert response.status_code == 422


def test_get_legend_graphic(xpublish_client, png_snapshot):
    """GetLegendGraphic returns a colorbar PNG matching the snapshot."""
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetLegendGraphic",
            "layer": "air",
            "styles": "raster/viridis",
            "vertical": "true",
            "width": 200,
            "height": 400,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == png_snapshot


def test_get_legend_graphic_no_label(xpublish_client, png_snapshot):
    """show_label=false suppresses the axis label."""
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetLegendGraphic",
            "layer": "air",
            "styles": "raster/viridis",
            "show_label": "false",
            "width": 120,
            "height": 300,
        },
    )
    assert response.status_code == 200
    assert response.content == png_snapshot


def test_get_legend_graphic_unknown_layer(xpublish_client):
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetLegendGraphic",
            "layer": "missing",
            "width": 100,
            "height": 100,
        },
    )
    assert response.status_code == 422


def test_get_legend_graphic_missing_colorscalerange():
    """Continuous data without valid_min/max and no colorscalerange -> 422."""
    ds = xr.Dataset(
        {
            "no_range": xr.DataArray(
                [[0.0, 1.0], [1.0, 0.0]],
                dims=["lat", "lon"],
                coords={
                    "lat": (["lat"], [0.0, 1.0], {"axis": "Y"}),
                    "lon": (["lon"], [0.0, 1.0], {"axis": "X"}),
                },
            )
        }
    )
    rest = xpublish.Rest({"d": ds}, plugins={"wms": WMSPlugin()})
    client = TestClient(rest.app)

    r = client.get(
        "/datasets/d/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetLegendGraphic",
            "layer": "no_range",
            "styles": "raster/viridis",
            "width": 100,
            "height": 100,
        },
    )
    assert r.status_code == 422
    assert "ServiceExceptionReport" in r.text
    assert "colorscalerange" in r.text


def test_get_capabilities_bounds_and_legends(xpublish_client):
    """Layers carry EX_GeographicBoundingBox, axis-order-correct BoundingBoxes,
    per-layer styles with legend URLs, and are queryable."""
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetCapabilities",
            "format": "json",
        },
    )
    assert response.status_code == 200

    child_layers = response.json()["capability"]["layer"]["layers"]
    air_layer = next(layer for layer in child_layers if layer.get("name") == "air")

    assert air_layer["queryable"] is True

    ex_bbox = air_layer["ex_geographic_bounding_box"]
    assert ex_bbox["west_bound_longitude"] == pytest.approx(-161.25)
    assert ex_bbox["east_bound_longitude"] == pytest.approx(-28.75)
    assert ex_bbox["south_bound_latitude"] == pytest.approx(13.75)
    assert ex_bbox["north_bound_latitude"] == pytest.approx(76.25)

    bboxes = {bbox["crs"]: bbox for bbox in air_layer["bounding_box"]}
    assert {"CRS:84", "EPSG:4326", "EPSG:3857"} <= set(bboxes)
    # EPSG:4326 is north-first in WMS 1.3.0, CRS:84 stays lon,lat
    assert bboxes["EPSG:4326"]["minx"] == bboxes["CRS:84"]["miny"]
    assert bboxes["EPSG:4326"]["miny"] == bboxes["CRS:84"]["minx"]
    assert bboxes["EPSG:4326"]["maxx"] == bboxes["CRS:84"]["maxy"]
    assert bboxes["EPSG:4326"]["maxy"] == bboxes["CRS:84"]["maxx"]

    styles = air_layer["styles"]
    assert len(styles) > 0
    href = styles[0]["legend_url"]["online_resource"]["href"]
    assert "request=GetLegendGraphic" in href
    assert "layer=air" in href


def test_get_map_axis_order(xpublish_client):
    """WMS 1.3.0 EPSG:4326 bboxes are lat,lon ordered; CRS:84 stays lon,lat."""
    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": "air",
        "styles": "raster/viridis",
        "width": 256,
        "height": 256,
        "time": "2013-01-01T00:00:00",
    }
    latlon = xpublish_client.get(
        "/datasets/air/wms",
        params={**params, "crs": "EPSG:4326", "bbox": "15,-160,75,-30"},
    )
    lonlat = xpublish_client.get(
        "/datasets/air/wms",
        params={**params, "crs": "CRS:84", "bbox": "-160,15,-30,75"},
    )
    assert latlon.status_code == 200
    assert lonlat.status_code == 200
    assert latlon.content == lonlat.content


def test_get_map_rejects_wms_111(xpublish_client):
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={
            "service": "WMS",
            "version": "1.1.1",
            "request": "GetMap",
            "layers": "air",
            "crs": "EPSG:4326",
            "bbox": "-160,15,-30,75",
            "width": 256,
            "height": 256,
        },
    )
    assert response.status_code == 400
    assert "ServiceExceptionReport" in response.text
    assert "1.3.0" in response.text


def test_get_map_exception_formats(xpublish_client):
    """Errors honor the EXCEPTIONS parameter: XML report, in-image, or blank."""
    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": "missing",
        "crs": "EPSG:3857",
        "bbox": "-8766409,5009377,-7514065,6261721",
        "width": 64,
        "height": 64,
    }
    response = xpublish_client.get("/datasets/air/wms", params=params)
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("text/xml")
    assert "ServiceExceptionReport" in response.text
    assert "LayerNotDefined" in response.text

    response = xpublish_client.get(
        "/datasets/air/wms", params={**params, "exceptions": "INIMAGE"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    response = xpublish_client.get(
        "/datasets/air/wms", params={**params, "exceptions": "BLANK"}
    )
    assert response.status_code == 200
    blank = Image.open(io.BytesIO(response.content))
    assert blank.size == (64, 64)
    assert blank.getchannel("A").getextrema() == (0, 0)


GET_FEATURE_INFO_PARAMS = {
    "service": "WMS",
    "version": "1.3.0",
    "request": "GetFeatureInfo",
    "query_layers": "air",
    "crs": "EPSG:4326",
    # lat,lon order: EPSG:4326 is north-first in WMS 1.3.0
    "bbox": "15,-160,75,-30",
    "width": 256,
    "height": 256,
    "i": 128,
    "j": 128,
    "time": "2013-01-01T00:00:00",
}


def test_get_feature_info(xpublish_client, air_dataset):
    response = xpublish_client.get("/datasets/air/wms", params=GET_FEATURE_INFO_PARAMS)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    (feature,) = response.json()["features"]

    # centre of pixel (128, 128) in a 256x256 map of lon -160..-30, lat 15..75
    lon, lat = feature["geometry"]["coordinates"]
    assert lon == pytest.approx(-95 + 130 / 512)
    assert lat == pytest.approx(45 - 60 / 512)
    expected = air_dataset.air.sel(time="2013-01-01T00:00:00", lon=265.0, lat=45.0)
    assert feature["properties"] == {
        "air": pytest.approx(expected.item()),
        "lon": 265.0,
        "lat": 45.0,
        "time": "2013-01-01T00:00:00",
    }


def test_get_feature_info_crs84(xpublish_client):
    """CRS:84 is east-first, so the same map sends lon,lat."""
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={**GET_FEATURE_INFO_PARAMS, "crs": "CRS:84", "bbox": "-160,15,-30,75"},
    )
    expected = xpublish_client.get("/datasets/air/wms", params=GET_FEATURE_INFO_PARAMS)
    assert response.status_code == 200
    assert response.json() == expected.json()


def test_get_feature_info_outside_data(xpublish_client):
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={**GET_FEATURE_INFO_PARAMS, "bbox": "-10,0,10,20"},
    )
    assert response.status_code == 200
    assert response.json() == {"type": "FeatureCollection", "features": []}


def test_get_feature_info_pixel_outside_map(xpublish_client):
    response = xpublish_client.get(
        "/datasets/air/wms", params={**GET_FEATURE_INFO_PARAMS, "i": 256}
    )
    assert response.status_code == 422


def test_get_feature_info_query_layers_wins_over_layers(xpublish_client):
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={**GET_FEATURE_INFO_PARAMS, "layers": "air,nope", "query_layers": "air"},
    )
    assert response.status_code == 200
    (feature,) = response.json()["features"]
    assert "air" in feature["properties"]


def test_get_feature_info_unknown_layer(xpublish_client):
    response = xpublish_client.get(
        "/datasets/air/wms", params={**GET_FEATURE_INFO_PARAMS, "query_layers": "nope"}
    )
    assert response.status_code == 422
    assert "LayerNotDefined" in response.text


def test_get_feature_info_covjson(xpublish_client):
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={
            **GET_FEATURE_INFO_PARAMS,
            "info_format": "application/prs.coverage+json",
            "time": "2014-12-31T00:00/..",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/prs.coverage+json"
    covjson = response.json()
    assert covjson["domain"]["domainType"] == "PointSeries"
    assert covjson["ranges"]["air"]["shape"] == [4]


def test_get_feature_info_geojson_rejects_range(xpublish_client):
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={**GET_FEATURE_INFO_PARAMS, "time": "2014-12-31T00:00/.."},
    )
    assert response.status_code == 422
    assert "InvalidDimensionValue" in response.text


@pytest.mark.parametrize(
    "time, expected_steps",
    [(None, 2920), ("2013-01-01T00:00/2013-01-02T00:00", 5)],
)
def test_get_timeseries(xpublish_client, air_dataset, time, expected_steps):
    params = {**GET_FEATURE_INFO_PARAMS, "request": "GetTimeseries"}
    params.pop("time")
    if time is not None:
        params["time"] = time
    response = xpublish_client.get("/datasets/air/wms", params=params)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/prs.coverage+json"
    covjson = response.json()
    assert covjson["domain"]["domainType"] == "PointSeries"
    air = covjson["ranges"]["air"]
    assert air["shape"] == [expected_steps]
    expected = air_dataset.air.sel(lon=265.0, lat=45.0).values[:expected_steps]
    assert air["values"] == pytest.approx(expected.tolist())


def test_get_timeseries_rejects_geojson(xpublish_client):
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={
            **GET_FEATURE_INFO_PARAMS,
            "request": "GetTimeseries",
            "info_format": "application/json",
        },
    )
    assert response.status_code == 422


def test_get_timeseries_too_big(xpublish_client):
    params = {**GET_FEATURE_INFO_PARAMS, "request": "GetTimeseries"}
    params.pop("time")
    with config.set({"max_renderable_size": 1024}):
        response = xpublish_client.get("/datasets/air/wms", params=params)
    assert response.status_code == 413


@pytest.mark.parametrize("elevation", [None, "all", "500/850"])
def test_get_vertical_profile(air_dataset, elevation):
    ds = air_dataset.expand_dims(level=[1000.0, 850.0, 500.0])
    ds.level.attrs.update(axis="Z", positive="down", units="hPa")
    client = TestClient(xpublish.Rest({"air": ds}, plugins={"wms": WMSPlugin()}).app)
    params = {**GET_FEATURE_INFO_PARAMS, "request": "GetVerticalProfile"}
    if elevation is not None:
        params["elevation"] = elevation
    response = client.get("/datasets/air/wms", params=params)
    assert response.status_code == 200
    domain = response.json()["domain"]
    assert domain["domainType"] == "VerticalProfile"
    expected = [1000.0, 850.0, 500.0] if elevation != "500/850" else [850.0, 500.0]
    assert domain["axes"]["z"] == {"values": expected}


def test_get_vertical_profile_without_z(xpublish_client):
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={**GET_FEATURE_INFO_PARAMS, "request": "GetVerticalProfile"},
    )
    assert response.status_code == 400
    assert "Z axis" in response.text


def test_get_map_jpeg(xpublish_client):
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetMap",
            "layers": "air",
            "crs": "CRS:84",
            "bbox": "-160,15,-30,75",
            "width": 128,
            "height": 128,
            "format": "image/jpeg",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    image = Image.open(io.BytesIO(response.content))
    assert image.format == "JPEG"


def test_get_map_opaque_bgcolor(xpublish_client):
    """TRANSPARENT=FALSE flattens the map onto BGCOLOR."""
    response = xpublish_client.get(
        "/datasets/air/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetMap",
            "layers": "air",
            "crs": "CRS:84",
            # extends north of the data so the background shows through
            "bbox": "-140,60,-100,85",
            "width": 128,
            "height": 128,
            "transparent": "false",
            "bgcolor": "0x0000FF",
        },
    )
    assert response.status_code == 200
    image = Image.open(io.BytesIO(response.content))
    assert image.mode == "RGB"
    assert image.getpixel((64, 2)) == (0, 0, 255)


def test_get_map_native_crs_eu3035():
    """Data on a projected native CRS is advertised and renderable natively.

    EPSG:3035 is north-first, so both the advertised BoundingBox and the
    GetMap BBOX use y,x order per WMS 1.3.0.
    """
    ds = EU3035.create()
    rest = xpublish.Rest({"eu3035": ds}, plugins={"wms": WMSPlugin()})
    client = TestClient(rest.app)

    capabilities = client.get(
        "/datasets/eu3035/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetCapabilities",
            "format": "json",
        },
    )
    layer = capabilities.json()["capability"]["layer"]["layers"][0]
    assert "EPSG:3035" in layer["crs"]
    native_bbox = next(b for b in layer["bounding_box"] if b["crs"] == "EPSG:3035")
    # north-first: minx/miny attributes hold northing/easting
    assert native_bbox["minx"] == pytest.approx(1802800.0)
    assert native_bbox["miny"] == pytest.approx(2635780.0)
    assert native_bbox["maxx"] == pytest.approx(5416000.0)
    assert native_bbox["maxy"] == pytest.approx(6248980.0)

    response = client.get(
        "/datasets/eu3035/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetMap",
            "layers": "foo",
            "styles": "raster/viridis",
            "crs": "EPSG:3035",
            "bbox": f"{native_bbox['minx']},{native_bbox['miny']},{native_bbox['maxx']},{native_bbox['maxy']}",
            "width": 256,
            "height": 256,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


@pytest.mark.parametrize(
    "tile,expected_level",
    [
        pytest.param(morecantile.Tile(x=15, y=10, z=5), "2", id="zoom5_coarsest"),
        pytest.param(morecantile.Tile(x=60, y=43, z=7), "0", id="zoom7_finest"),
    ],
)
def test_get_map_multiscale_levels(tile, expected_level):
    """GetMap picks the overview level matching the requested resolution."""
    tree = GEOZARR_MULTISCALE.create()
    rest = xpublish.Rest({"pyramid": tree}, plugins={"wms": WMSPlugin()})
    client = TestClient(rest.app)

    tms = morecantile.tms.get("WebMercatorQuad")
    bounds = tms.xy_bounds(tile)
    response = client.get(
        "/datasets/pyramid/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetMap",
            "layers": "data",
            "crs": "EPSG:3857",
            "bbox": f"{bounds.left},{bounds.bottom},{bounds.right},{bounds.top}",
            "width": 256,
            "height": 256,
        },
    )
    assert response.status_code == 200
    assert response.headers["X-Multiscale-Level"] == expected_level


def test_wms_openapi_schema_generation(xpublish_client):
    """OpenAPI generation should succeed and expose CRS/BBox as string params."""
    response = xpublish_client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    wms_get = schema["paths"]["/datasets/{dataset_id}/wms/"]["get"]
    assert any(param["name"] == "root" for param in wms_get["parameters"])

    assert schema["components"]["schemas"]["CRSParam"]["type"] == "string"
    assert schema["components"]["schemas"]["BBoxParam"]["type"] == "string"


def test_wms_getmap_rgb_variant():
    """WMS GetMap with ``styles=raster/rgb``."""
    rest = xpublish.Rest({"rgb": RGB.create()}, plugins={"wms": WMSPlugin()})
    client = TestClient(rest.app)
    response = client.get(
        "/datasets/rgb/wms",
        params=[
            ("service", "WMS"),
            ("version", "1.3.0"),
            ("request", "GetMap"),
            ("layers", "foo"),
            ("styles", "raster/rgb"),
            ("crs", "EPSG:4326"),
            ("bbox", "-90,-180,90,180"),
            ("width", 256),
            ("height", 256),
        ],
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    arr = np.asarray(Image.open(io.BytesIO(response.content)))
    assert arr[128, 250, 0] > arr[128, 5, 0]

    legend = client.get(
        "/datasets/rgb/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetLegendGraphic",
            "layer": "foo",
            "styles": "raster/rgb",
            "width": 100,
            "height": 300,
        },
    )
    assert legend.status_code == 422
    assert "has no legend" in legend.text


POLYGONS_ONLY_DATASETS = [CUBED_SPHERE, GLOBAL_HEALPIX_L3, GLOBAL_HEALPIX_CUBE_L3]

POLYGONS_ONLY_GETMAP = {
    "service": "WMS",
    "version": "1.3.0",
    "request": "GetMap",
    "layers": "foo",
    "crs": "EPSG:3857",
    "bbox": "-20037508,-20037508,20037508,20037508",
    "width": 256,
    "height": 256,
    "colorscalerange": "0,100",
}


@pytest.fixture(params=POLYGONS_ONLY_DATASETS, ids=lambda d: d.name)
def polygons_only_wms_client(request):
    rest = xpublish.Rest({"polys": request.param.create()}, plugins={"wms": WMSPlugin()})
    return TestClient(rest.app)


def test_polygons_only_grid_wms_default_style(polygons_only_wms_client):
    client = polygons_only_wms_client
    response = client.get("/datasets/polys/wms", params=POLYGONS_ONLY_GETMAP)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    Image.open(io.BytesIO(response.content)).verify()

    response = client.get(
        "/datasets/polys/wms",
        params={
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetLegendGraphic",
            "layer": "foo",
            "width": 100,
            "height": 300,
            "colorscalerange": "0,100",
        },
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"


def test_polygons_only_grid_wms_raster_style_is_422(polygons_only_wms_client):
    response = polygons_only_wms_client.get(
        "/datasets/polys/wms",
        params={**POLYGONS_ONLY_GETMAP, "styles": "raster/default"},
    )
    assert response.status_code == 422, response.text
    assert "ServiceExceptionReport" in response.text
    assert "StyleNotDefined" in response.text
    assert "only supports style=" in response.text


def test_polygons_wms_edgecolor(polygons_only_wms_client):
    plain = polygons_only_wms_client.get(
        "/datasets/polys/wms", params=POLYGONS_ONLY_GETMAP
    )
    edged = polygons_only_wms_client.get(
        "/datasets/polys/wms",
        params={**POLYGONS_ONLY_GETMAP, "edgecolor": "default", "edgewidth": 2},
    )
    assert plain.status_code == 200, plain.text
    assert edged.status_code == 200, edged.text
    assert plain.content != edged.content


AIR_GETMAP = {
    "service": "WMS",
    "version": "1.3.0",
    "request": "GetMap",
    "layers": "air",
    "crs": "EPSG:3857",
    "bbox": "-8766409.899970, 5009377.085697, -7514065.628546, 6261721.357122",
    "width": 256,
    "height": 256,
    "time": "2013-01-01T00:00:00",
}


@pytest.mark.parametrize(
    "extra,match",
    [
        pytest.param(
            {"styles": "raster/default", "edgecolor": "red"}, "edgecolor", id="raster"
        ),
        pytest.param({"edgecolor": "notacolor"}, "Invalid color", id="bad-color"),
        pytest.param({"edgecolor": "red", "edgewidth": 0}, "edgewidth", id="width-0"),
        pytest.param({"edgecolor": "red", "edgewidth": 6}, "edgewidth", id="width-6"),
    ],
)
def test_wms_edgecolor_422(xpublish_client, extra, match):
    response = xpublish_client.get("/datasets/air/wms", params={**AIR_GETMAP, **extra})
    assert response.status_code == 422, response.text
    assert match in response.text
