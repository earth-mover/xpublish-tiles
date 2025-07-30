import morecantile
import pytest
from morecantile import Tile

# WebMercatorQuad TMS for creating tiles
WEBMERC_TMS = morecantile.tms.get("WebMercatorQuad")
# WorldMercatorWGS84Quad TMS for WGS84 Mercator tiles
WGS84_TMS = morecantile.tms.get("WorldMercatorWGS84Quad")
# EuropeanETRS89_LAEAQuad TMS for ETRS89 LAEA CRS
ETRS89_TMS = morecantile.tms.get("EuropeanETRS89_LAEAQuad")

# WebMercator tiles (supported TMS)
WEBMERC_TILES = [
    # WebMercatorQuad tiles - European region focus to avoid anti-meridian issues
    pytest.param(Tile(x=2, y=1, z=2), WEBMERC_TMS, id="webmerc_europe_center(2/2/1)"),
    pytest.param(Tile(x=1, y=1, z=2), WEBMERC_TMS, id="webmerc_europe_west(2/1/1)"),
    pytest.param(Tile(x=0, y=0, z=5), WEBMERC_TMS, id="webmerc_europe_south(5/0/0)"),
    # Note: webmerc_europe_east(2/3/1) removed - causes anti-meridian crossing when projected to ETRS89 LAEA
    pytest.param(Tile(x=2, y=0, z=2), WEBMERC_TMS, id="webmerc_europe_north(2/2/0)"),
    pytest.param(Tile(x=2, y=2, z=2), WEBMERC_TMS, id="webmerc_europe_south(2/2/2)"),
    # Higher zoom European region
    pytest.param(Tile(x=8, y=5, z=4), WEBMERC_TMS, id="webmerc_europe_zoom4(4/8/5)"),
    pytest.param(Tile(x=16, y=10, z=5), WEBMERC_TMS, id="webmerc_europe_zoom5(5/16/10)"),
    # Small bbox test
    pytest.param(Tile(x=8, y=8, z=5), WEBMERC_TMS, id="webmerc_small_bbox(5/8/8)"),
    # Anti-meridian (180/-180 degrees) problematic tiles
    # At z=2, x=0 is the western hemisphere, x=3 is the eastern edge
    pytest.param(Tile(x=0, y=1, z=2), WEBMERC_TMS, id="webmerc_antimeridian_west(2/0/1)"),
    pytest.param(Tile(x=3, y=1, z=2), WEBMERC_TMS, id="webmerc_antimeridian_east(2/3/1)"),
    # Higher zoom anti-meridian tiles
    pytest.param(
        Tile(x=0, y=2, z=3), WEBMERC_TMS, id="webmerc_antimeridian_z3_west(3/0/2)"
    ),
    pytest.param(
        Tile(x=7, y=2, z=3), WEBMERC_TMS, id="webmerc_antimeridian_z3_east(3/7/2)"
    ),
    pytest.param(
        Tile(x=31, y=10, z=5), WEBMERC_TMS, id="webmerc_antimeridian_z5_east(5/31/10)"
    ),
    pytest.param(
        Tile(x=0, y=10, z=5), WEBMERC_TMS, id="webmerc_antimeridian_z5_west(5/0/10)"
    ),
    # Prime meridian (0 degrees) problematic tiles
    # At z=2, x=2 covers the prime meridian
    pytest.param(Tile(x=2, y=1, z=2), WEBMERC_TMS, id="webmerc_prime_meridian(2/2/1)"),
    # Higher zoom prime meridian tiles
    pytest.param(Tile(x=4, y=2, z=3), WEBMERC_TMS, id="webmerc_prime_meridian_z3(3/4/2)"),
    pytest.param(
        Tile(x=16, y=10, z=5), WEBMERC_TMS, id="webmerc_prime_meridian_z5(5/16/10)"
    ),
    pytest.param(Tile(x=15, y=10, z=5), WEBMERC_TMS, id="webmerc_prime_west_z5(5/15/10)"),
    pytest.param(Tile(x=17, y=10, z=5), WEBMERC_TMS, id="webmerc_prime_east_z5(5/17/10)"),
    # Equator (0 degrees latitude) tiles
    # In WebMercator, at zoom level z, y coordinate ranges from 0 to 2^z-1
    # The equator is at y = 2^(z-1), so at z=2, equator is between y=1 and y=2
    # At z=3, equator is between y=3 and y=4
    # At z=5, equator is between y=15 and y=16
    pytest.param(Tile(x=1, y=2, z=2), WEBMERC_TMS, id="webmerc_equator_south(2/1/2)"),
    pytest.param(Tile(x=1, y=1, z=2), WEBMERC_TMS, id="webmerc_equator_north(2/1/1)"),
    # Equator tiles at different longitudes
    pytest.param(Tile(x=0, y=2, z=2), WEBMERC_TMS, id="webmerc_equator_west(2/0/2)"),
    pytest.param(Tile(x=3, y=2, z=2), WEBMERC_TMS, id="webmerc_equator_east(2/3/2)"),
    # Higher zoom equator tiles
    pytest.param(Tile(x=4, y=4, z=3), WEBMERC_TMS, id="webmerc_equator_z3_south(3/4/4)"),
    pytest.param(Tile(x=4, y=3, z=3), WEBMERC_TMS, id="webmerc_equator_z3_north(3/4/3)"),
    pytest.param(
        Tile(x=16, y=16, z=5), WEBMERC_TMS, id="webmerc_equator_z5_south(5/16/16)"
    ),
    pytest.param(
        Tile(x=16, y=15, z=5), WEBMERC_TMS, id="webmerc_equator_z5_north(5/16/15)"
    ),
    # Equator at anti-meridian
    pytest.param(
        Tile(x=0, y=16, z=5), WEBMERC_TMS, id="webmerc_equator_antimeridian_west(5/0/16)"
    ),
    pytest.param(
        Tile(x=31, y=16, z=5),
        WEBMERC_TMS,
        id="webmerc_equator_antimeridian_east(5/31/16)",
    ),
    # Note: WGS84 tiles would go here when WorldMercatorWGS84Quad TMS is implemented
]

# All tiles including ETRS89 (some may not be supported)
ETRS89_TILES = [
    # ETRS89 LAEA tiles - European region specific
    # Center of Europe tiles
    pytest.param(Tile(x=1, y=1, z=2), ETRS89_TMS, id="etrs89_center_europe(2/1/1)"),
    pytest.param(Tile(x=0, y=1, z=2), ETRS89_TMS, id="etrs89_west_europe(2/0/1)"),
    pytest.param(Tile(x=2, y=1, z=2), ETRS89_TMS, id="etrs89_east_europe(2/2/1)"),
    # Northern Europe (Scandinavia region)
    pytest.param(Tile(x=1, y=0, z=2), ETRS89_TMS, id="etrs89_north_europe(2/1/0)"),
    # Southern Europe (Mediterranean region)
    pytest.param(Tile(x=1, y=2, z=2), ETRS89_TMS, id="etrs89_south_europe(2/1/2)"),
    # Higher zoom edge cases within Europe
    pytest.param(Tile(x=4, y=4, z=4), ETRS89_TMS, id="etrs89_central_zoom4(4/4/4)"),
    pytest.param(Tile(x=2, y=2, z=3), ETRS89_TMS, id="etrs89_central_zoom3(3/2/2)"),
    pytest.param(Tile(x=6, y=6, z=4), ETRS89_TMS, id="etrs89_southeast_zoom4(4/6/6)"),
    # Small bbox test for ETRS89
    pytest.param(Tile(x=8, y=8, z=5), ETRS89_TMS, id="etrs89_small_bbox(5/8/8)"),
]

# WGS84 tiles for equator anti-meridian and 0 longitude testing
WGS84_TILES = [
    # Equator tiles for testing WGS84 coordinate handling
    # At z=2, equator is between y=1 and y=2 in WGS84 projection
    pytest.param(Tile(x=1, y=1, z=2), WGS84_TMS, id="wgs84_equator_north(2/1/1)"),
    pytest.param(Tile(x=1, y=2, z=2), WGS84_TMS, id="wgs84_equator_south(2/1/2)"),
    # Prime meridian (0 degrees longitude) tiles
    pytest.param(Tile(x=2, y=1, z=2), WGS84_TMS, id="wgs84_prime_meridian(2/2/1)"),
    pytest.param(Tile(x=1, y=1, z=2), WGS84_TMS, id="wgs84_prime_west(2/1/1)"),
    pytest.param(Tile(x=3, y=1, z=2), WGS84_TMS, id="wgs84_prime_east(2/3/1)"),
    # Anti-meridian tiles (180/-180 degrees longitude)
    # At z=2, x=0 covers western edge near anti-meridian
    pytest.param(Tile(x=0, y=1, z=2), WGS84_TMS, id="wgs84_antimeridian_west(2/0/1)"),
    pytest.param(
        Tile(x=0, y=2, z=2), WGS84_TMS, id="wgs84_antimeridian_west_equator(2/0/2)"
    ),
    # Equator at anti-meridian - key test case for coordinate transformation
    pytest.param(
        Tile(x=0, y=1, z=3), WGS84_TMS, id="wgs84_equator_antimeridian_z3(3/0/1)"
    ),
    pytest.param(
        Tile(x=0, y=2, z=3), WGS84_TMS, id="wgs84_equator_antimeridian_south_z3(3/0/2)"
    ),
    # Higher zoom equator anti-meridian tiles
    pytest.param(
        Tile(x=0, y=15, z=5), WGS84_TMS, id="wgs84_equator_antimeridian_north_z5(5/0/15)"
    ),
    pytest.param(
        Tile(x=0, y=16, z=5), WGS84_TMS, id="wgs84_equator_antimeridian_south_z5(5/0/16)"
    ),
]

TILES = WEBMERC_TILES + WGS84_TILES + ETRS89_TILES
