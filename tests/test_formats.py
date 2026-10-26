import json

import cftime
import numpy as np
import pytest
from pyproj import CRS

from xpublish_tiles.extract import extract_points
from xpublish_tiles.formats import json_values, points_to_covjson, points_to_geojson

WGS84 = CRS.from_epsg(4326)
INSIDE = [(-95.0, 45.0), (-120.0, 30.0)]
OUTSIDE = [(0.0, 0.0)]
DAY = "2013-01-01T00:00/2013-01-02T00:00"


async def extract(ds, points, variables=("air",), **selectors):
    lon, lat = (list(c) for c in zip(*points, strict=True))
    result = await extract_points(
        ds,
        variables=list(variables),
        selectors=selectors,
        crs=WGS84,
        x=np.array(lon),
        y=np.array(lat),
        allow_ranges=True,
    )
    return result, lon, lat


async def covjson(ds, points, variables=("air",), **selectors):
    result, lon, lat = await extract(ds, points, variables, **selectors)
    return points_to_covjson(result, lon=lon, lat=lat)


@pytest.fixture(scope="module")
def air_levels(air_dataset):
    ds = air_dataset.expand_dims(level=[1000.0, 850.0, 500.0])
    ds.level.attrs.update(axis="Z", positive="down", units="hPa")
    return ds


def test_json_values():
    assert json_values(np.array([1.5, np.nan, np.inf])) == [1.5, None, None]
    assert json_values(np.array(["2000-01-01T06", "NaT"], dtype="datetime64[ns]")) == [
        "2000-01-01T06:00:00",
        None,
    ]
    assert json_values(np.array([np.timedelta64(36, "h")])) == ["P1DT12H"]
    assert json_values(np.array([cftime.DatetimeNoLeap(2000, 2, 28)])) == [
        "2000-02-28T00:00:00"
    ]
    assert json_values(np.array(3, dtype=np.int16)) == [3]


@pytest.mark.parametrize(
    "dataset, points, selectors, domain_type, shape",
    [
        ("air_dataset", INSIDE[:1], {"time": "2013-01-01T00:00"}, "Point", {}),
        ("air_dataset", INSIDE[:1], {"time": DAY}, "PointSeries", {"t": 5}),
        (
            "air_dataset",
            INSIDE + OUTSIDE,
            {"time": DAY},
            "MultiPointSeries",
            {"t": 5, "composite": 2},
        ),
        ("air_levels", INSIDE[:1], {"level": "../.."}, "VerticalProfile", {"z": 3}),
        # no standard domain type for these
        (
            "air_levels",
            INSIDE[:1],
            {"level": "../..", "time": DAY},
            None,
            {"t": 5, "z": 3},
        ),
        (
            "air_levels",
            INSIDE,
            {"level": "850", "time": "2013-01-01T00:00"},
            None,
            {"composite": 2},
        ),
    ],
)
async def test_covjson_domain(request, dataset, points, selectors, domain_type, shape):
    result = await covjson(request.getfixturevalue(dataset), points, **selectors)
    json.dumps(result, allow_nan=False)
    assert result["domain"].get("domainType") == domain_type
    air = result["ranges"]["air"]
    assert dict(zip(air["axisNames"], air["shape"], strict=True)) == shape


async def test_covjson_point_series_contents(air_dataset):
    result = await covjson(air_dataset, INSIDE[:1], time=DAY)
    domain = result["domain"]
    assert domain["axes"]["x"] == {"values": [-95.0]}
    assert domain["axes"]["y"] == {"values": [45.0]}
    assert domain["axes"]["t"]["values"][0] == "2013-01-01T00:00:00"
    assert [r["coordinates"] for r in domain["referencing"]] == [["x", "y"], ["t"]]
    expected = air_dataset.air.sel(lon=265.0, lat=45.0, time=slice(*DAY.split("/")))
    assert result["ranges"]["air"]["values"] == pytest.approx(expected.values.tolist())
    assert result["parameters"]["air"]["unit"] == {"symbol": "degK"}


async def test_covjson_vertical_referencing(air_levels):
    result = await covjson(air_levels, INSIDE[:1], level="../..")
    assert result["domain"]["axes"]["z"] == {"values": [1000.0, 850.0, 500.0]}
    (z_axis,) = result["domain"]["referencing"][-1]["system"]["cs"]["csAxes"]
    assert z_axis["direction"] == "down"
    assert z_axis["unit"] == {"symbol": "hPa"}


async def test_covjson_time_axis_ignores_forecast_reference_time(air_dataset):
    ds = air_dataset.assign_coords(
        forecast_reference_time=(
            (),
            np.datetime64("2012-12-31T00:00"),
            {"standard_name": "forecast_reference_time"},
        )
    )
    result = await covjson(ds, INSIDE[:1], time="2013-01-01T00:00/2013-01-01T06:00")
    assert result["domain"]["axes"]["t"]["values"] == [
        "2013-01-01T00:00:00",
        "2013-01-01T06:00:00",
    ]


async def test_covjson_boolean_as_integer(air_dataset):
    ds = air_dataset.assign(freezing=air_dataset.air < 273.15)
    result = await covjson(ds, INSIDE[:1], ["freezing"], time="2013-01-01T00:00")
    freezing = result["ranges"]["freezing"]
    assert freezing["dataType"] == "integer"
    assert freezing["values"] == [1]


async def test_covjson_no_points(air_dataset):
    assert await covjson(air_dataset, OUTSIDE, time="2013-01-01T00:00") == {
        "type": "CoverageCollection",
        "coverages": [],
    }


async def test_geojson(air_dataset):
    result, lon, lat = await extract(
        air_dataset, INSIDE + OUTSIDE, time="2013-01-01T00:00"
    )
    geojson = points_to_geojson(result, lon=lon, lat=lat)
    assert [f["geometry"]["coordinates"] for f in geojson["features"]] == [
        [-95.0, 45.0],
        [-120.0, 30.0],
    ]
    assert geojson["features"][0]["properties"]["time"] == "2013-01-01T00:00:00"
