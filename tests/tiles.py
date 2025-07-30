import morecantile
import pytest
from morecantile import Tile

# WebMercatorQuad TMS for creating tiles
_TMS = morecantile.tms.get("WebMercatorQuad")
# EuropeanETRS89_LAEAQuad TMS for ETRS89 LAEA CRS
_ETRS89_TMS = morecantile.tms.get("EuropeanETRS89_LAEAQuad")

TILES = [
    # WebMercatorQuad tiles - European region focus to avoid anti-meridian issues
    pytest.param(Tile(x=2, y=1, z=2), _TMS, id="webmerc_europe_center(2/2/1)"),
    pytest.param(Tile(x=1, y=1, z=2), _TMS, id="webmerc_europe_west(2/1/1)"),
    pytest.param(Tile(x=0, y=0, z=5), _TMS, id="webmerc_europe_south(5/0/0)"),
    # Note: webmerc_europe_east(2/3/1) removed - causes anti-meridian crossing when projected to ETRS89 LAEA
    pytest.param(Tile(x=2, y=0, z=2), _TMS, id="webmerc_europe_north(2/2/0)"),
    pytest.param(Tile(x=2, y=2, z=2), _TMS, id="webmerc_europe_south(2/2/2)"),
    # Higher zoom European region
    pytest.param(Tile(x=8, y=5, z=4), _TMS, id="webmerc_europe_zoom4(4/8/5)"),
    pytest.param(Tile(x=16, y=10, z=5), _TMS, id="webmerc_europe_zoom5(5/16/10)"),
    # Small bbox test
    pytest.param(Tile(x=8, y=8, z=5), _TMS, id="webmerc_small_bbox(5/8/8)"),
    # Anti-meridian (180/-180 degrees) problematic tiles
    # At z=2, x=0 is the western hemisphere, x=3 is the eastern edge
    pytest.param(Tile(x=0, y=1, z=2), _TMS, id="webmerc_antimeridian_west(2/0/1)"),
    pytest.param(Tile(x=3, y=1, z=2), _TMS, id="webmerc_antimeridian_east(2/3/1)"),
    # Higher zoom anti-meridian tiles
    pytest.param(Tile(x=0, y=2, z=3), _TMS, id="webmerc_antimeridian_z3_west(3/0/2)"),
    pytest.param(Tile(x=7, y=2, z=3), _TMS, id="webmerc_antimeridian_z3_east(3/7/2)"),
    pytest.param(Tile(x=31, y=10, z=5), _TMS, id="webmerc_antimeridian_z5_east(5/31/10)"),
    pytest.param(Tile(x=0, y=10, z=5), _TMS, id="webmerc_antimeridian_z5_west(5/0/10)"),
    # Prime meridian (0 degrees) problematic tiles
    # At z=2, x=2 covers the prime meridian
    pytest.param(Tile(x=2, y=1, z=2), _TMS, id="webmerc_prime_meridian(2/2/1)"),
    # Higher zoom prime meridian tiles
    pytest.param(Tile(x=4, y=2, z=3), _TMS, id="webmerc_prime_meridian_z3(3/4/2)"),
    pytest.param(Tile(x=16, y=10, z=5), _TMS, id="webmerc_prime_meridian_z5(5/16/10)"),
    pytest.param(Tile(x=15, y=10, z=5), _TMS, id="webmerc_prime_west_z5(5/15/10)"),
    pytest.param(Tile(x=17, y=10, z=5), _TMS, id="webmerc_prime_east_z5(5/17/10)"),
    # Equator (0 degrees latitude) tiles
    # In WebMercator, at zoom level z, y coordinate ranges from 0 to 2^z-1
    # The equator is at y = 2^(z-1), so at z=2, equator is between y=1 and y=2
    # At z=3, equator is between y=3 and y=4
    # At z=5, equator is between y=15 and y=16
    pytest.param(Tile(x=1, y=2, z=2), _TMS, id="webmerc_equator_south(2/1/2)"),
    pytest.param(Tile(x=1, y=1, z=2), _TMS, id="webmerc_equator_north(2/1/1)"),
    # Equator tiles at different longitudes
    pytest.param(Tile(x=0, y=2, z=2), _TMS, id="webmerc_equator_west(2/0/2)"),
    pytest.param(Tile(x=3, y=2, z=2), _TMS, id="webmerc_equator_east(2/3/2)"),
    # Higher zoom equator tiles
    pytest.param(Tile(x=4, y=4, z=3), _TMS, id="webmerc_equator_z3_south(3/4/4)"),
    pytest.param(Tile(x=4, y=3, z=3), _TMS, id="webmerc_equator_z3_north(3/4/3)"),
    pytest.param(Tile(x=16, y=16, z=5), _TMS, id="webmerc_equator_z5_south(5/16/16)"),
    pytest.param(Tile(x=16, y=15, z=5), _TMS, id="webmerc_equator_z5_north(5/16/15)"),
    # Equator at anti-meridian
    pytest.param(
        Tile(x=0, y=16, z=5), _TMS, id="webmerc_equator_antimeridian_west(5/0/16)"
    ),
    pytest.param(
        Tile(x=31, y=16, z=5), _TMS, id="webmerc_equator_antimeridian_east(5/31/16)"
    ),
    # ETRS89 LAEA tiles - European region specific
    # Center of Europe tiles
    pytest.param(Tile(x=1, y=1, z=2), _ETRS89_TMS, id="etrs89_center_europe(2/1/1)"),
    pytest.param(Tile(x=0, y=1, z=2), _ETRS89_TMS, id="etrs89_west_europe(2/0/1)"),
    pytest.param(Tile(x=2, y=1, z=2), _ETRS89_TMS, id="etrs89_east_europe(2/2/1)"),
    # Northern Europe (Scandinavia region)
    pytest.param(Tile(x=1, y=0, z=2), _ETRS89_TMS, id="etrs89_north_europe(2/1/0)"),
    # Southern Europe (Mediterranean region)
    pytest.param(Tile(x=1, y=2, z=2), _ETRS89_TMS, id="etrs89_south_europe(2/1/2)"),
    # Higher zoom edge cases within Europe
    pytest.param(Tile(x=4, y=4, z=4), _ETRS89_TMS, id="etrs89_central_zoom4(4/4/4)"),
    pytest.param(Tile(x=2, y=2, z=3), _ETRS89_TMS, id="etrs89_central_zoom3(3/2/2)"),
    pytest.param(Tile(x=6, y=6, z=4), _ETRS89_TMS, id="etrs89_southeast_zoom4(4/6/6)"),
    # Small bbox test for ETRS89
    pytest.param(Tile(x=8, y=8, z=5), _ETRS89_TMS, id="etrs89_small_bbox(5/8/8)"),
]
