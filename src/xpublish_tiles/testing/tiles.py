import morecantile
import pytest
from morecantile import Tile

# WebMercatorQuad TMS for creating tiles
WEBMERC_TMS = morecantile.tms.get("WebMercatorQuad")
# WorldMercatorWGS84Quad TMS for WGS84 Mercator tiles
WGS84_TMS = morecantile.tms.get("WorldMercatorWGS84Quad")
# EuropeanETRS89_LAEAQuad TMS for ETRS89 LAEA CRS
ETRS89_TMS = morecantile.tms.get("EuropeanETRS89_LAEAQuad")

# WebMercator tiles - regular cases
WEBMERC_TILES_REGULAR = [
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
    # Additional anti-meridian tiles
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
    # Additional prime meridian tiles
    pytest.param(Tile(x=4, y=2, z=3), WEBMERC_TMS, id="webmerc_prime_meridian_z3(3/4/2)"),
    pytest.param(
        Tile(x=16, y=10, z=5), WEBMERC_TMS, id="webmerc_prime_meridian_z5(5/16/10)"
    ),
    pytest.param(Tile(x=15, y=10, z=5), WEBMERC_TMS, id="webmerc_prime_west_z5(5/15/10)"),
    pytest.param(Tile(x=17, y=10, z=5), WEBMERC_TMS, id="webmerc_prime_east_z5(5/17/10)"),
    # Additional equator tiles
    pytest.param(Tile(x=1, y=1, z=2), WEBMERC_TMS, id="webmerc_equator_north(2/1/1)"),
    pytest.param(Tile(x=0, y=2, z=2), WEBMERC_TMS, id="webmerc_equator_west(2/0/2)"),
    pytest.param(Tile(x=3, y=2, z=2), WEBMERC_TMS, id="webmerc_equator_east(2/3/2)"),
    pytest.param(Tile(x=4, y=4, z=3), WEBMERC_TMS, id="webmerc_equator_z3_south(3/4/4)"),
    pytest.param(Tile(x=4, y=3, z=3), WEBMERC_TMS, id="webmerc_equator_z3_north(3/4/3)"),
    pytest.param(
        Tile(x=16, y=16, z=5), WEBMERC_TMS, id="webmerc_equator_z5_south(5/16/16)"
    ),
    pytest.param(
        Tile(x=16, y=15, z=5), WEBMERC_TMS, id="webmerc_equator_z5_north(5/16/15)"
    ),
    pytest.param(
        Tile(x=31, y=16, z=5),
        WEBMERC_TMS,
        id="webmerc_equator_antimeridian_east(5/31/16)",
    ),
]

# WebMercator tiles - edge cases for integration tests (max 5)
WEBMERC_TILES_EDGE_CASES = [
    # Anti-meridian (180/-180 degrees) problematic tiles
    pytest.param(Tile(x=0, y=1, z=2), WEBMERC_TMS, id="webmerc_antimeridian_west(2/0/1)"),
    pytest.param(Tile(x=3, y=1, z=2), WEBMERC_TMS, id="webmerc_antimeridian_east(2/3/1)"),
    # Prime meridian (0 degrees) problematic tiles
    pytest.param(Tile(x=2, y=1, z=2), WEBMERC_TMS, id="webmerc_prime_meridian(2/2/1)"),
    # Equator (0 degrees latitude) tiles
    pytest.param(Tile(x=1, y=2, z=2), WEBMERC_TMS, id="webmerc_equator_south(2/1/2)"),
    # Equator at anti-meridian - most complex edge case
    pytest.param(
        Tile(x=0, y=16, z=5), WEBMERC_TMS, id="webmerc_equator_antimeridian_west(5/0/16)"
    ),
]

# WebMercator tiles (supported TMS) - combined
WEBMERC_TILES = WEBMERC_TILES_REGULAR + WEBMERC_TILES_EDGE_CASES

# ETRS89 tiles - regular cases
ETRS89_TILES_REGULAR = [
    # ETRS89 LAEA tiles - European region specific
    # Center of Europe tiles
    pytest.param(Tile(x=1, y=1, z=2), ETRS89_TMS, id="etrs89_center_europe(2/1/1)"),
    pytest.param(Tile(x=0, y=1, z=2), ETRS89_TMS, id="etrs89_west_europe(2/0/1)"),
    pytest.param(Tile(x=2, y=1, z=2), ETRS89_TMS, id="etrs89_east_europe(2/2/1)"),
    # Northern Europe (Scandinavia region)
    pytest.param(Tile(x=1, y=0, z=2), ETRS89_TMS, id="etrs89_north_europe(2/1/0)"),
    # Southern Europe (Mediterranean region)
    pytest.param(Tile(x=1, y=2, z=2), ETRS89_TMS, id="etrs89_south_europe(2/1/2)"),
    # Higher zoom cases within Europe
    pytest.param(Tile(x=2, y=2, z=3), ETRS89_TMS, id="etrs89_central_zoom3(3/2/2)"),
    pytest.param(Tile(x=5, y=2, z=3), ETRS89_TMS, id="etrs89_central_zoom3(3/2/5)"),
]

# ETRS89 tiles - edge cases for integration tests (max 5)
ETRS89_TILES_EDGE_CASES = [
    # Higher zoom edge cases within Europe
    pytest.param(Tile(x=4, y=4, z=4), ETRS89_TMS, id="etrs89_central_zoom4(4/4/4)"),
    pytest.param(Tile(x=6, y=6, z=4), ETRS89_TMS, id="etrs89_southeast_zoom4(4/6/6)"),
    # Small bbox test for ETRS89
    pytest.param(Tile(x=8, y=8, z=5), ETRS89_TMS, id="etrs89_small_bbox(5/8/8)"),
    pytest.param(Tile(x=15, y=12, z=5), WEBMERC_TMS, id="webmerc_corner_zoom5(15/12/5)"),
    pytest.param(Tile(x=5, y=1, z=3), WEBMERC_TMS, id="webmerc_corner_zoom3(3/5/1)"),
]

# ETRS89 tiles (some may not be supported) - combined
ETRS89_TILES = ETRS89_TILES_REGULAR + ETRS89_TILES_EDGE_CASES

# WGS84 tiles - regular cases
WGS84_TILES_REGULAR = [
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
]

# WGS84 tiles - edge cases for integration tests (max 5)
WGS84_TILES_EDGE_CASES = [
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

# WGS84 tiles for equator anti-meridian and 0 longitude testing - combined
WGS84_TILES = WGS84_TILES_REGULAR + WGS84_TILES_EDGE_CASES

# HRRR tiles - regular cases
# Actual HRRR domain: Lat(21.14, 47.84), Lon(-134.10, -60.92)
HRRR_TILES_REGULAR = [
    # Low zoom - full domain coverage
    pytest.param(Tile(x=0, y=1, z=2), WEBMERC_TMS, id="hrrr_west_z2(2/0/1)"),
    pytest.param(Tile(x=1, y=1, z=2), WEBMERC_TMS, id="hrrr_east_z2(2/1/1)"),
    # Medium zoom - domain corners and edges
    pytest.param(Tile(x=1, y=2, z=3), WEBMERC_TMS, id="hrrr_sw_corner_z3(3/1/2)"),
    pytest.param(Tile(x=2, y=2, z=3), WEBMERC_TMS, id="hrrr_se_corner_z3(3/2/2)"),
    pytest.param(Tile(x=1, y=2, z=3), WEBMERC_TMS, id="hrrr_nw_corner_z3(3/1/2)"),
    pytest.param(Tile(x=2, y=2, z=3), WEBMERC_TMS, id="hrrr_ne_corner_z3(3/2/2)"),
    pytest.param(Tile(x=1, y=3, z=3), WEBMERC_TMS, id="hrrr_south_z3(3/1/3)"),
    # Higher zoom - precise domain coverage
    pytest.param(Tile(x=4, y=11, z=5), WEBMERC_TMS, id="hrrr_west_edge_z5(5/4/11)"),
    pytest.param(Tile(x=10, y=11, z=5), WEBMERC_TMS, id="hrrr_east_edge_z5(5/10/11)"),
    pytest.param(Tile(x=4, y=13, z=5), WEBMERC_TMS, id="hrrr_south_edge_z5(5/4/13)"),
    pytest.param(Tile(x=7, y=11, z=5), WEBMERC_TMS, id="hrrr_center_z5(5/7/11)"),
    # Very high zoom cases
    pytest.param(Tile(x=16, y=44, z=7), WEBMERC_TMS, id="hrrr_sw_precise_z7(7/16/44)"),
    pytest.param(Tile(x=42, y=44, z=7), WEBMERC_TMS, id="hrrr_se_precise_z7(7/42/44)"),
    pytest.param(Tile(x=29, y=50, z=7), WEBMERC_TMS, id="hrrr_center_z7(7/29/50)"),
]

# HRRR tiles - edge cases for integration tests (max 5)
HRRR_TILES_EDGE_CASES = [
    # Ultra high zoom - precise boundaries (edge cases)
    pytest.param(
        Tile(x=130, y=356, z=10), WEBMERC_TMS, id="hrrr_sw_extreme_z10(10/130/356)"
    ),
    pytest.param(
        Tile(x=338, y=356, z=10), WEBMERC_TMS, id="hrrr_se_extreme_z10(10/338/356)"
    ),
    pytest.param(Tile(x=234, y=403, z=10), WEBMERC_TMS, id="hrrr_center_z10(10/234/403)"),
]

# HRRR tiles for testing Lambert Conformal Conic projection data - combined
HRRR_TILES = HRRR_TILES_REGULAR + HRRR_TILES_EDGE_CASES

# Para tiles - regular cases
# Para is approximately between 2.72°N to 9.93°S and 45.97°W to 58.99°W
PARA_TILES_REGULAR = [
    # Zoom level 4 - broader coverage
    pytest.param(Tile(x=5, y=7, z=4), WEBMERC_TMS, id="para_north_z4(4/5/7)"),
    pytest.param(Tile(x=5, y=8, z=4), WEBMERC_TMS, id="para_south_z4(4/5/8)"),
    # Zoom level 5 - more detailed coverage
    pytest.param(Tile(x=10, y=15, z=5), WEBMERC_TMS, id="para_northwest_z5(5/10/15)"),
    pytest.param(Tile(x=11, y=15, z=5), WEBMERC_TMS, id="para_northeast_z5(5/11/15)"),
    pytest.param(Tile(x=10, y=16, z=5), WEBMERC_TMS, id="para_southwest_z5(5/10/16)"),
    pytest.param(Tile(x=11, y=16, z=5), WEBMERC_TMS, id="para_southeast_z5(5/11/16)"),
    # Zoom level 6 - covering Belém (capital) area at ~1.5°S, 48.5°W
    pytest.param(Tile(x=22, y=31, z=6), WEBMERC_TMS, id="para_belem_z6(6/22/31)"),
    # Zoom level 7 - detailed view
    pytest.param(Tile(x=44, y=63, z=7), WEBMERC_TMS, id="para_north_z7(7/44/63)"),
    pytest.param(Tile(x=45, y=64, z=7), WEBMERC_TMS, id="para_central_z7(7/45/64)"),
    # Zoom level 8 - high detail for southern Para
    pytest.param(Tile(x=88, y=128, z=8), WEBMERC_TMS, id="para_south_z8(8/88/128)"),
]

# Para tiles - edge cases for integration tests (max 5)
PARA_TILES_EDGE_CASES = [
    # test upsampling at very high zoom levels - true edge cases
    pytest.param(
        Tile(x=1480, y=2064, z=12), WEBMERC_TMS, id="para_south_z8(12/1480/2064)"
    ),
    pytest.param(
        Tile(x=2964, y=4129, z=13), WEBMERC_TMS, id="para_south_z8(13/2964/4129)"
    ),
    pytest.param(
        Tile(x=5971, y=8252, z=14), WEBMERC_TMS, id="para_south_z8(14/5971/8252)"
    ),
]

# Para (Brazilian state) tiles for testing South American region - combined
PARA_TILES = PARA_TILES_REGULAR + PARA_TILES_EDGE_CASES

# UTM Zone 33S tiles - regular cases covering southern Africa to Antarctica
UTM33S_TILES_REGULAR = [
    # Zoom 2 - Large coverage
    pytest.param(Tile(x=2, y=2, z=2), WEBMERC_TMS, id="utm33s_africa_z2(2/2/2)"),
    pytest.param(Tile(x=2, y=3, z=2), WEBMERC_TMS, id="utm33s_antarctica_z2(2/2/3)"),
    # Zoom 3 - Medium coverage
    pytest.param(Tile(x=4, y=4, z=3), WEBMERC_TMS, id="utm33s_africa_z3(3/4/4)"),
    pytest.param(Tile(x=4, y=5, z=3), WEBMERC_TMS, id="utm33s_mid_z3(3/4/5)"),
    pytest.param(Tile(x=4, y=6, z=3), WEBMERC_TMS, id="utm33s_deep_z3(3/4/6)"),
    # Zoom 4 - More detailed
    pytest.param(Tile(x=8, y=8, z=4), WEBMERC_TMS, id="utm33s_north_z4(4/8/8)"),
    # pytest.param(Tile(x=8, y=9, z=4), WEBMERC_TMS, id="utm33s_central_z4(4/8/9)"),
    pytest.param(Tile(x=8, y=10, z=4), WEBMERC_TMS, id="utm33s_south_z4(4/8/10)"),
    pytest.param(Tile(x=8, y=11, z=4), WEBMERC_TMS, id="utm33s_antarctica_z4(4/8/11)"),
    pytest.param(
        Tile(x=8, y=14, z=4), WEBMERC_TMS, id="utm33s_deep_antarctica_z4(4/8/14)"
    ),
    # Zoom 5 - Detailed tiles
    pytest.param(Tile(x=17, y=16, z=5), WEBMERC_TMS, id="utm33s_equator_z5(5/17/16)"),
    pytest.param(Tile(x=17, y=17, z=5), WEBMERC_TMS, id="utm33s_north_z5(5/17/17)"),
    pytest.param(Tile(x=17, y=18, z=5), WEBMERC_TMS, id="utm33s_central_z5(5/17/18)"),
    pytest.param(Tile(x=17, y=20, z=5), WEBMERC_TMS, id="utm33s_south_z5(5/17/20)"),
    pytest.param(Tile(x=17, y=23, z=5), WEBMERC_TMS, id="utm33s_antarctica_z5(5/17/23)"),
    # pytest.param(Tile(x=17, y=25, z=5), WEBMERC_TMS, id="utm33s_deep_z5(5/17/25)"),
    pytest.param(
        Tile(x=17, y=22, z=5), WEBMERC_TMS, id="utm33s_mid_antarctica_z5(5/17/22)"
    ),
]

# UTM Zone 33S tiles - edge cases for equator and Antarctica boundaries
UTM33S_TILES_EDGE_CASES = [
    # Equator edge cases (northern boundary at 0°)
    pytest.param(Tile(x=34, y=32, z=6), WEBMERC_TMS, id="utm33s_equator_z6(6/34/32)"),
    pytest.param(Tile(x=68, y=64, z=7), WEBMERC_TMS, id="utm33s_equator_z7(7/68/64)"),
    pytest.param(Tile(x=136, y=128, z=8), WEBMERC_TMS, id="utm33s_equator_z8(8/136/128)"),
    # Antarctica edge cases (southern boundary at -80°S)
    pytest.param(
        Tile(x=17, y=28, z=5), WEBMERC_TMS, id="utm33s_antarctica_edge_z5(5/17/28)"
    ),
    pytest.param(
        Tile(x=34, y=56, z=6), WEBMERC_TMS, id="utm33s_antarctica_edge_z6(6/34/56)"
    ),
    # pytest.param(Tile(x=68, y=112, z=7), WEBMERC_TMS, id="utm33s_antarctica_edge_z7(7/68/112)"),
    # pytest.param(Tile(x=136, y=224, z=8), WEBMERC_TMS, id="utm33s_antarctica_edge_z8(8/136/224)"),
    # Very high zoom equator and Antarctica
    pytest.param(Tile(x=277, y=256, z=9), WEBMERC_TMS, id="utm33s_equator_z9(9/277/256)"),
    pytest.param(
        Tile(x=277, y=448, z=9), WEBMERC_TMS, id="utm33s_antarctica_z9(9/277/448)"
    ),
    pytest.param(
        Tile(x=4372, y=4160, z=13),
        WEBMERC_TMS,
        id="utm33s_center_swatch_z9(13/4160/4372)",
    ),
]

# UTM Zone 33S tiles - combined
UTM33S_TILES = UTM33S_TILES_REGULAR + UTM33S_TILES_EDGE_CASES

# Curvilinear tiles - for testing curvilinear coordinate data
CURVILINEAR_TILES = [
    pytest.param(
        (Tile(x=3, y=5, z=4), WEBMERC_TMS), id="curvilinear_hrrr_east_z4(4/3/5)"
    ),
    pytest.param(
        (Tile(x=7, y=12, z=5), WEBMERC_TMS), id="curvilinear_hrrr_sw_corner_z5(5/7/12)"
    ),
    pytest.param(
        (Tile(x=6, y=11, z=5), WEBMERC_TMS), id="curvilinear_hrrr_se_corner_z5(5/6/11)"
    ),
    pytest.param(
        (Tile(x=27, y=48, z=7), WEBMERC_TMS), id="curvilinear_hrrr_central_z7(7/27/48)"
    ),
    pytest.param(
        (Tile(x=15, y=24, z=6), WEBMERC_TMS), id="curvilinear_central_us_z6(6/15/24)"
    ),
    pytest.param(
        (Tile(x=442, y=744, z=11), WEBMERC_TMS),
        id="curvilinear_central_us_z11(11/442/744)",
    ),
]

# South America benchmark tiles (for Sentinel dataset)
# Coverage area roughly: -82°W to -27°W, 13°N to -55°S
# Focused on the region that's working in the logs (tiles 120-122, 72-73)
# fmt: off
SOUTH_AMERICA_BENCHMARK_TILES = [
    # Zoom 7 - Broader coverage of the working region
    # "7/60/36", "7/61/36",
    # Zoom 8 - The confirmed working tiles
    "8/120/72", "8/121/72", "8/122/72",
    "8/120/73", "8/121/73", "8/122/73",
    # Zoom 9 - Higher detail within the working region
    "9/240/144", "9/241/144", "9/242/144", "9/243/144", "9/244/144", "9/245/144",
    "9/240/145", "9/241/145", "9/242/145", "9/243/145", "9/244/145", "9/245/145",
    "9/240/146", "9/241/146", "9/242/146", "9/243/146", "9/244/146", "9/245/146",
    "9/240/147", "9/241/147", "9/242/147", "9/243/147", "9/244/147", "9/245/147",
    # Zoom 10 - Very high detail for center of region
    "10/482/289", "10/483/289", "10/484/289", "10/485/289",
    "10/482/290", "10/483/290", "10/484/290", "10/485/290",
    "10/482/291", "10/483/291", "10/484/291", "10/485/291",
    # Zoom 11 - Ultra high detail for a small area
    "11/966/580", "11/967/580", "11/966/581", "11/967/581",
]

# UTM Zone 50S benchmark tiles (for high-resolution UTM50S dataset)
# Extracted from server logs - tiles around zoom 13-17
UTM50S_HIRES_BENCHMARK_TILES = [
    # Zoom 12 tiles - 10 additional tiles
    "12/2153/3819", "12/2153/3820", "12/2153/3821", "12/2153/3822", "12/2154/3819",
    "12/2154/3820", "12/2154/3821", "12/2154/3822", "12/2153/3818", "12/2154/3818",
    # Zoom 13 tiles
    "13/4306/7638", "13/4306/7639", "13/4306/7640", "13/4306/7641", "13/4306/7642", "13/4307/7638", "13/4307/7639",
    "13/4307/7640", "13/4307/7641", "13/4307/7642", "13/4308/7638", "13/4308/7639", "13/4308/7640", "13/4308/7641",
    "13/4308/7642", "13/4309/7638", "13/4309/7639", "13/4309/7640", "13/4309/7641", "13/4309/7642", "13/4306/7643",
    "13/4306/7644", "13/4306/7645", "13/4306/7637", "13/4306/7636", "13/4307/7643", "13/4307/7644", "13/4307/7645",
    "13/4307/7637", "13/4307/7636", "13/4308/7643", "13/4308/7644", "13/4308/7645", "13/4308/7637", "13/4308/7636",
    "13/4309/7643", "13/4309/7644", "13/4307/7646", "13/4309/7637", "13/4309/7636", "13/4306/7635", "13/4307/7635",
    "13/4308/7635", "13/4309/7635", "13/4306/7646",
    # Zoom 14 tiles
    "14/8615/15279", "14/8615/15280", "14/8615/15281", "14/8616/15279",
    "14/8616/15280", "14/8616/15281", "14/8617/15279", "14/8617/15280",
    # Zoom 15 tiles
    "15/17231/30559", "15/17231/30560", "15/17231/30561", "15/17232/30559", "15/17232/30560", "15/17232/30561", "15/17233/30559",
    # Zoom 16 tiles
    "16/34463/61119", "16/34463/61120", "16/34463/61121", "16/34463/61122",
    "16/34464/61119", "16/34464/61120", "16/34464/61121", "16/34464/61122",
    "16/34465/61119", "16/34465/61120", "16/34465/61121", "16/34465/61122",
    # Zoom 17 tiles
    "17/68926/122238", "17/68926/122239", "17/68926/122240", "17/68926/122241", "17/68926/122242", "17/68927/122238", "17/68927/122239", "17/68927/122240",
    "17/68927/122241", "17/68927/122242", "17/68928/122238", "17/68928/122239", "17/68928/122240", "17/68928/122241", "17/68928/122242", "17/68929/122238",
    "17/68929/122239", "17/68929/122240", "17/68929/122241", "17/68929/122242", "17/68930/122238", "17/68930/122239", "17/68930/122240", "17/68930/122241",
    "17/68930/122242", "17/68925/122238", "17/68925/122239", "17/68925/122240", "17/68925/122241", "17/68925/122242", "17/68931/122238", "17/68931/122239",
    "17/68931/122240", "17/68931/122241", "17/68931/122242", "17/68932/122238", "17/68932/122239", "17/68932/122240", "17/68932/122241", "17/68932/122242",
]
# fmt: on

TILES = WEBMERC_TILES + WGS84_TILES + ETRS89_TILES
