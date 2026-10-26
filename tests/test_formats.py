import json

import cftime
import numpy as np
import pytest
from pyproj import CRS

from xpublish_tiles.extract import extract_points
from xpublish_tiles.formats import json_values, points_to_covjson, points_to_geojson

WGS84 = CRS.from_epsg(4326)
LON, LAT = [-95.0, -120.0, 0.0], [45.0, 30.0, 0.0]


async def extract(ds, npoints, **selectors):
    return await extract_points(
        ds,
        variables=["air"],
        selectors=selectors,
        crs=WGS84,
        x=np.array(LON[:npoints]),
        y=np.array(LAT[:npoints]),
        allow_ranges=True,
    )


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


async def test_covjson_point_series(air_dataset):
    result = await extract(air_dataset, 1, time="2013-01-01T00:00/2013-01-02T00:00")
    covjson = points_to_covjson(result, lon=LON, lat=LAT)
    json.dumps(covjson, allow_nan=False)

    domain = covjson["domain"]
    assert domain["domainType"] == "PointSeries"
    assert domain["axes"]["x"] == {"values": [-95.0]}
    assert domain["axes"]["y"] == {"values": [45.0]}
    assert domain["axes"]["t"]["values"][0] == "2013-01-01T00:00:00"
    assert len(domain["axes"]["t"]["values"]) == 5
    assert [r["coordinates"] for r in domain["referencing"]] == [["x", "y"], ["t"]]

    air = covjson["ranges"]["air"]
    assert air["axisNames"] == ["t"]
    assert air["shape"] == [5]
    expected = air_dataset.air.sel(time=slice("2013-01-01T00:00", "2013-01-02T00:00"))
    assert air["values"] == pytest.approx(
        expected.sel(lon=265.0, lat=45.0).values.tolist()
    )
    assert covjson["parameters"]["air"]["unit"] == {"symbol": "degK"}


async def test_covjson_multipoint_series_drops_outside_points(air_dataset):
    result = await extract(air_dataset, 3, time="2013-01-01T00:00/2013-01-01T12:00")
    covjson = points_to_covjson(result, lon=LON, lat=LAT)
    domain = covjson["domain"]
    assert domain["domainType"] == "MultiPointSeries"
    # (0, 0) is outside the grid
    assert domain["axes"]["composite"]["values"] == [[-95.0, 45.0], [-120.0, 30.0]]
    air = covjson["ranges"]["air"]
    assert dict(zip(air["axisNames"], air["shape"], strict=True)) == {
        "t": 3,
        "composite": 2,
    }


async def test_covjson_point(air_dataset):
    result = await extract(air_dataset, 1, time="2013-01-01T00:00")
    domain = points_to_covjson(result, lon=LON, lat=LAT)["domain"]
    assert domain["domainType"] == "Point"
    assert domain["axes"]["t"] == {"values": ["2013-01-01T00:00:00"]}


async def test_covjson_vertical_profile(air_levels):
    result = await extract(air_levels, 1, level="../..")
    covjson = points_to_covjson(result, lon=LON, lat=LAT)
    domain = covjson["domain"]
    assert domain["domainType"] == "VerticalProfile"
    assert domain["axes"]["z"] == {"values": [1000.0, 850.0, 500.0]}
    (z_axis,) = domain["referencing"][-1]["system"]["cs"]["csAxes"]
    assert z_axis["direction"] == "down"
    assert z_axis["unit"] == {"symbol": "hPa"}


async def test_covjson_no_standard_domain(air_levels):
    result = await extract(
        air_levels, 1, level="../..", time="2013-01-01T00:00/2013-01-01T12:00"
    )
    covjson = points_to_covjson(result, lon=LON, lat=LAT)
    assert "domainType" not in covjson["domain"]
    assert sorted(covjson["ranges"]["air"]["axisNames"]) == ["t", "z"]


async def test_covjson_time_axis_ignores_forecast_reference_time(air_dataset):
    ds = air_dataset.assign_coords(
        forecast_reference_time=(
            (),
            np.datetime64("2012-12-31T00:00"),
            {"standard_name": "forecast_reference_time"},
        )
    )
    result = await extract(ds, 1, time="2013-01-01T00:00/2013-01-01T06:00")
    covjson = points_to_covjson(result, lon=LON, lat=LAT)
    assert covjson["domain"]["axes"]["t"]["values"] == [
        "2013-01-01T00:00:00",
        "2013-01-01T06:00:00",
    ]
    assert covjson["ranges"]["air"]["shape"] == [2]


async def test_covjson_multipoint_with_level_has_no_domain_type(air_levels):
    result = await extract(air_levels, 3, level="850", time="2013-01-01T00:00")
    domain = points_to_covjson(result, lon=LON, lat=LAT)["domain"]
    assert "domainType" not in domain
    assert domain["axes"]["z"] == {"values": [850.0]}


async def test_covjson_no_points(air_dataset):
    result = await extract_points(
        air_dataset,
        variables=["air"],
        selectors={},
        crs=WGS84,
        x=np.array([0.0]),
        y=np.array([0.0]),
    )
    assert points_to_covjson(result, lon=[0.0], lat=[0.0]) == {
        "type": "CoverageCollection",
        "coverages": [],
    }


async def test_geojson(air_dataset):
    result = await extract(air_dataset, 3, time="2013-01-01T00:00")
    geojson = points_to_geojson(result, lon=LON, lat=LAT)
    assert [f["geometry"]["coordinates"] for f in geojson["features"]] == [
        [-95.0, 45.0],
        [-120.0, 30.0],
    ]
    assert geojson["features"][0]["properties"]["time"] == "2013-01-01T00:00:00"
