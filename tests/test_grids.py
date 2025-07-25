# FIXME: vendor these

from itertools import product

import cf_xarray as cfxr
import cf_xarray.datasets
import morecantile
import numpy as np
import pytest
from pyproj import CRS
from pyproj.aoi import BBox

import xarray as xr
from tests.datasets import Dim
from xpublish_tiles.grids import Curvilinear, GridSystem, Rectilinear, guess_grid_system

HRRR_CRS_WKT = "".join(
    [
        'PROJCRS["unknown",BASEGEOGCRS["unknown",DATUM["unknown",ELLIPSOID["unk',
        'nown",6371229,0,LENGTHUNIT["metre",1,ID["EPSG",9001]]]],PRIMEM["Greenw',
        'ich",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8901]]],CONVER',
        'SION["unknown",METHOD["Lambert Conic Conformal',
        '(2SP)",ID["EPSG",9802]],PARAMETER["Latitude of false origin",38.5,ANGL',
        'EUNIT["degree",0.0174532925199433],ID["EPSG",8821]],PARAMETER["Longitu',
        'de of false origin",262.5,ANGLEUNIT["degree",0.0174532925199433],ID["E',
        'PSG",8822]],PARAMETER["Latitude of 1st standard parallel",38.5,ANGLEUN',
        'IT["degree",0.0174532925199433],ID["EPSG",8823]],PARAMETER["Latitude',
        'of 2nd standard parallel",38.5,ANGLEUNIT["degree",0.0174532925199433],',
        'ID["EPSG",8824]],PARAMETER["Easting at false',
        'origin",0,LENGTHUNIT["metre",1],ID["EPSG",8826]],PARAMETER["Northing',
        'at false origin",0,LENGTHUNIT["metre",1],ID["EPSG",8827]]],CS[Cartesia',
        'n,2],AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1,ID["EPSG",9001]]],A',
        'XIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]',
    ]
)

# FIXME: add tests for datasets with latitude, longitude but no attrs


@pytest.mark.parametrize(
    "ds, array_name, expected",
    (
        (
            cfxr.datasets.forecast,
            "sst",
            Rectilinear(
                crs=CRS.from_user_input(4326),
                bbox=BBox(south=0, north=5, east=4, west=0),
                X="X",
                Y="Y",
                indexes=(),
            ),
        ),
        (
            cfxr.datasets.popds,
            "UVEL",
            Curvilinear(
                crs=CRS.from_user_input(4326),
                bbox=BBox(south=2.5, north=2.5, east=0.5, west=0.5),
                X="ULONG",
                Y="ULAT",
                indexes=(),
            ),
        ),
        (
            cfxr.datasets.rotds,
            "temp",
            Rectilinear(
                crs=CRS.from_cf(
                    {
                        "grid_mapping_name": "rotated_latitude_longitude",
                        "grid_north_pole_latitude": 39.25,
                        "grid_north_pole_longitude": -162.0,
                    }
                ),
                bbox=BBox(south=21.615, north=21.835, east=18.155, west=17.935),
                X="rlon",
                Y="rlat",
                indexes=(),
            ),
        ),
        (
            cfxr.datasets.rotds,
            "temp",
            Rectilinear(
                crs=CRS.from_cf(
                    {
                        "grid_mapping_name": "rotated_latitude_longitude",
                        "grid_north_pole_latitude": 39.25,
                        "grid_north_pole_longitude": -162.0,
                    }
                ),
                bbox=BBox(south=21.615, north=21.835, east=18.155, west=17.935),
                X="rlon",
                Y="rlat",
                indexes=(),
            ),
        ),
        (
            xr.tutorial.open_dataset("hrrr-cube"),
            "dswrf",
            Rectilinear(
                crs=CRS.from_wkt(HRRR_CRS_WKT),
                bbox=BBox(
                    west=-897520.1425219309,
                    south=-27306.15255666431,
                    east=-660520.1425219309,
                    north=89693.84744333569,
                ),
                X="x",
                Y="y",
                indexes=(),
            ),
        ),
    ),
)
def test_grid_detection(ds: xr.Dataset, array_name, expected: GridSystem) -> None:
    actual = guess_grid_system(ds, array_name)
    actual.indexes = ()  # FIXME
    assert expected == actual


# subsetting tests

from tests.datasets import uniform_grid


@pytest.fixture(
    params=tuple(map(",".join, product(["-90->90", "90->-90"], ["-180->180", "0->360"])))
)
def global_datasets(request):
    param = request.param
    dims = []

    nlat, nlon = 720, 1441
    lats = np.linspace(-90, 90, nlat)
    if "90->-90" in param:
        lats = lats[::-1]

    if "-180->180" in param:
        lons = np.linspace(-180, 180, nlon)
    else:
        lons = np.linspace(0, 360, nlon)

    dims = [
        Dim(
            name="latitude",
            size=nlat,
            chunk_size=nlat,
            data=lats,
            attrs={"standard_name": "latitude"},
        ),
        Dim(
            name="longitude",
            size=nlon,
            chunk_size=nlon,
            data=lons,
            attrs={"standard_name": "longitude"},
        ),
    ]
    yield uniform_grid(dims=tuple(dims), dtype=np.float32, attrs={})


# Problematic tiles that test edge cases
PROBLEMATIC_TILES = [
    # Equator crossing tiles (z=2, y=1 and y=2 cross equator)
    (2, 0, 1),  # Tile crossing equator (northern hemisphere)
    (2, 0, 2),  # Tile crossing equator (southern hemisphere)
    # Anti-meridian crossing tiles (x=0 and x=max cross 180°/-180°)
    (3, 0, 4),  # Tile crossing anti-meridian (left side)
    (3, 7, 4),  # Tile crossing anti-meridian (right side)
    # Near-polar tiles
    (4, 8, 0),  # Near north pole
    (4, 8, 15),  # Near south pole
    # Higher zoom problematic cases
    (6, 0, 32),  # Anti-meridian at higher zoom
    (6, 63, 32),  # Anti-meridian at higher zoom (other side)
    (5, 16, 15),  # Equator crossing at zoom 5
    (5, 16, 16),  # Equator crossing at zoom 5 (other side)
]


@pytest.mark.parametrize("z,x,y", PROBLEMATIC_TILES)
def test_subset(global_datasets, z, x, y):
    """Test subsetting with problematic tiles that span equator, anti-meridian, and poles."""
    ds = global_datasets
    grid = guess_grid_system(ds, "foo")

    tms = morecantile.tms.get("WebMercatorQuad")
    tile = morecantile.Tile(x=x, y=y, z=z)
    geo_bounds = tms.bounds(tile)
    bbox_geo = BBox(
        west=geo_bounds[0], south=geo_bounds[1], east=geo_bounds[2], north=geo_bounds[3]
    )

    actual = grid.sel(ds.foo, bbox=bbox_geo)

    # Basic validation that we got a result
    assert isinstance(actual, xr.DataArray)
    assert actual.size > 0

    # Check that coordinates are within expected bounds (exact matching with controlled grid)
    lat_min, lat_max = actual.latitude.min().item(), actual.latitude.max().item()
    assert lat_min >= bbox_geo.south, f"Latitude too low: {lat_min} < {bbox_geo.south}"
    assert lat_max <= bbox_geo.north, f"Latitude too high: {lat_max} > {bbox_geo.north}"

    lon_min, lon_max = actual.longitude.min().item(), actual.longitude.max().item()

    # Assert that returned coordinates match the -180→180 convention of the bounds
    # This ensures consistent output format regardless of input dataset convention
    assert lon_min >= -180.0, f"Longitude should be >= -180: {lon_min}"
    assert lon_max <= 180.0, f"Longitude should be <= 180: {lon_max}"

    # Coordinates should be within the bbox bounds (in -180→180 format)
    # Since Web Mercator tiles never cross anti-meridian, we can use simple bounds checking
    assert lon_min >= bbox_geo.west, f"Longitude too low: {lon_min} < {bbox_geo.west}"
    assert lon_max <= bbox_geo.east, f"Longitude too high: {lon_max} > {bbox_geo.east}"
