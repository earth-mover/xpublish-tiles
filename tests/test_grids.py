# FIXME: vendor these


import numpy as np
import cf_xarray as cfxr
import cf_xarray.datasets
import pytest
from pyproj import CRS
from pyproj.aoi import BBox

import xarray as xr
from tests.tiles import TILES
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


@pytest.mark.parametrize("tile,tms", TILES)
def test_subset(global_datasets, tile, tms):
    """Test subsetting with tiles that span equator, anti-meridian, and poles."""
    ds = global_datasets
    grid = guess_grid_system(ds, "foo")
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

    lon_values = actual.longitude.values
    lon_min, lon_max = lon_values.min().item(), lon_values.max().item()

    # Assert that longitude coordinates are spatially continuous
    # This prevents transparent pixels caused by coordinate discontinuities
    if len(lon_values) > 1:
        sorted_lons = np.sort(lon_values)
        lon_diffs = np.diff(sorted_lons)
        max_gap = lon_diffs.max()
        
        # Allow for reasonable coordinate spacing but catch large discontinuities
        # A gap > 180° indicates a problematic discontinuity
        assert max_gap <= 180.0, f"Longitude coordinates have large gap ({max_gap:.1f}°) indicating spatial discontinuity: {sorted_lons}"

    # Coordinates should be within reasonable bounds for the selected data
    # Allow for both -180→180 and 0→360 conventions, but ensure continuity
    # The actual coordinate values depend on the input data convention and selection logic
    if lon_min >= 0 and lon_max > 180:
        # Data uses 0→360 convention - ensure it's reasonable
        assert lon_min >= 0.0, f"0→360 longitude should be >= 0: {lon_min}"
        assert lon_max <= 360.0, f"0→360 longitude should be <= 360: {lon_max}"
    else:
        # Data uses -180→180 convention - ensure it's reasonable  
        assert lon_min >= -180.0, f"-180→180 longitude should be >= -180: {lon_min}"
        assert lon_max <= 180.0, f"-180→180 longitude should be <= 180: {lon_max}"
