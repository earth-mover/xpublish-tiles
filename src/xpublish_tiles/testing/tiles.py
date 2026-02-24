from dataclasses import dataclass
from functools import cached_property

import morecantile
from morecantile import Tile, TileMatrixSet


@dataclass(kw_only=True)
class TileTestParam:
    """A container for information about testing a dataset.

    We use this instead of pytest.param to avoid a runtime dependency on pytest.
    """

    tile: Tile
    tms: TileMatrixSet
    name: str  # Test parameter identifier; should be unique

    @cached_property
    def id(self):
        return f"{self.name}({self.tile.z}/{self.tile.x}/{self.tile.y})"


# WebMercatorQuad TMS for creating tiles
WEBMERC_TMS = morecantile.tms.get("WebMercatorQuad")
# WorldMercatorWGS84Quad TMS for WGS84 Mercator tiles
WGS84_TMS = morecantile.tms.get("WorldMercatorWGS84Quad")
# EuropeanETRS89_LAEAQuad TMS for ETRS89 LAEA CRS
ETRS89_TMS = morecantile.tms.get("EuropeanETRS89_LAEAQuad")

# WebMercator tiles - regular cases
WEBMERC_TILES_REGULAR = [
    # WebMercatorQuad tiles - European region focus to avoid anti-meridian issues
    TileTestParam(
        tile=Tile(x=2, y=1, z=2), tms=WEBMERC_TMS, name="webmerc_europe_center"
    ),
    TileTestParam(tile=Tile(x=1, y=1, z=2), tms=WEBMERC_TMS, name="webmerc_europe_west"),
    TileTestParam(tile=Tile(x=0, y=0, z=5), tms=WEBMERC_TMS, name="webmerc_europe_south"),
    # Note: webmerc_europe_east(2/3/1) removed - causes anti-meridian crossing when projected to ETRS89 LAEA
    TileTestParam(tile=Tile(x=2, y=0, z=2), tms=WEBMERC_TMS, name="webmerc_europe_north"),
    TileTestParam(tile=Tile(x=2, y=2, z=2), tms=WEBMERC_TMS, name="webmerc_europe_south"),
    # Higher zoom European region
    TileTestParam(tile=Tile(x=8, y=5, z=4), tms=WEBMERC_TMS, name="webmerc_europe_zoom4"),
    TileTestParam(
        tile=Tile(x=16, y=10, z=5), tms=WEBMERC_TMS, name="webmerc_europe_zoom5"
    ),
    # Small bbox test
    TileTestParam(tile=Tile(x=8, y=8, z=5), tms=WEBMERC_TMS, name="webmerc_small_bbox"),
    # Additional anti-meridian tiles
    TileTestParam(
        tile=Tile(x=0, y=2, z=3), tms=WEBMERC_TMS, name="webmerc_antimeridian_z3_west"
    ),
    TileTestParam(
        tile=Tile(x=7, y=2, z=3), tms=WEBMERC_TMS, name="webmerc_antimeridian_z3_east"
    ),
    TileTestParam(
        tile=Tile(x=31, y=10, z=5), tms=WEBMERC_TMS, name="webmerc_antimeridian_z5_east"
    ),
    TileTestParam(
        tile=Tile(x=0, y=10, z=5), tms=WEBMERC_TMS, name="webmerc_antimeridian_z5_west"
    ),
    # Additional prime meridian tiles
    TileTestParam(
        tile=Tile(x=4, y=2, z=3), tms=WEBMERC_TMS, name="webmerc_prime_meridian_z3"
    ),
    TileTestParam(
        tile=Tile(x=16, y=10, z=5), tms=WEBMERC_TMS, name="webmerc_prime_meridian_z5"
    ),
    TileTestParam(
        tile=Tile(x=15, y=10, z=5), tms=WEBMERC_TMS, name="webmerc_prime_west_z5"
    ),
    TileTestParam(
        tile=Tile(x=17, y=10, z=5), tms=WEBMERC_TMS, name="webmerc_prime_east_z5"
    ),
    # Additional equator tiles
    TileTestParam(
        tile=Tile(x=1, y=1, z=2), tms=WEBMERC_TMS, name="webmerc_equator_north"
    ),
    TileTestParam(tile=Tile(x=0, y=2, z=2), tms=WEBMERC_TMS, name="webmerc_equator_west"),
    TileTestParam(tile=Tile(x=3, y=2, z=2), tms=WEBMERC_TMS, name="webmerc_equator_east"),
    TileTestParam(
        tile=Tile(x=4, y=4, z=3), tms=WEBMERC_TMS, name="webmerc_equator_z3_south"
    ),
    TileTestParam(
        tile=Tile(x=4, y=3, z=3), tms=WEBMERC_TMS, name="webmerc_equator_z3_north"
    ),
    TileTestParam(
        tile=Tile(x=16, y=16, z=5), tms=WEBMERC_TMS, name="webmerc_equator_z5_south"
    ),
    TileTestParam(
        tile=Tile(x=16, y=15, z=5), tms=WEBMERC_TMS, name="webmerc_equator_z5_north"
    ),
    TileTestParam(
        tile=Tile(x=31, y=16, z=5),
        tms=WEBMERC_TMS,
        name="webmerc_equator_antimeridian_east",
    ),
]

# WebMercator tiles - edge cases for integration tests (max 5)
WEBMERC_TILES_EDGE_CASES = [
    # Anti-meridian (180/-180 degrees) problematic tiles
    TileTestParam(
        tile=Tile(x=0, y=1, z=2), tms=WEBMERC_TMS, name="webmerc_antimeridian_west"
    ),
    TileTestParam(
        tile=Tile(x=3, y=1, z=2), tms=WEBMERC_TMS, name="webmerc_antimeridian_east"
    ),
    # Prime meridian (0 degrees) problematic tiles
    TileTestParam(
        tile=Tile(x=2, y=1, z=2), tms=WEBMERC_TMS, name="webmerc_prime_meridian"
    ),
    # Equator (0 degrees latitude) tiles
    TileTestParam(
        tile=Tile(x=1, y=2, z=2), tms=WEBMERC_TMS, name="webmerc_equator_south"
    ),
    # Equator at anti-meridian - most complex edge case
    TileTestParam(
        tile=Tile(x=0, y=16, z=5),
        tms=WEBMERC_TMS,
        name="webmerc_equator_antimeridian_west",
    ),
]

# WebMercator tiles (supported TMS) - combined
WEBMERC_TILES = WEBMERC_TILES_REGULAR + WEBMERC_TILES_EDGE_CASES

# ETRS89 tiles - regular cases
ETRS89_TILES_REGULAR = [
    # ETRS89 LAEA tiles - European region specific
    # Center of Europe tiles
    TileTestParam(tile=Tile(x=1, y=1, z=2), tms=ETRS89_TMS, name="etrs89_center_europe"),
    TileTestParam(tile=Tile(x=0, y=1, z=2), tms=ETRS89_TMS, name="etrs89_west_europe"),
    TileTestParam(tile=Tile(x=2, y=1, z=2), tms=ETRS89_TMS, name="etrs89_east_europe"),
    # Northern Europe (Scandinavia region)
    TileTestParam(tile=Tile(x=1, y=0, z=2), tms=ETRS89_TMS, name="etrs89_north_europe"),
    # Southern Europe (Mediterranean region)
    TileTestParam(tile=Tile(x=1, y=2, z=2), tms=ETRS89_TMS, name="etrs89_south_europe"),
    # Higher zoom cases within Europe
    TileTestParam(tile=Tile(x=2, y=2, z=3), tms=ETRS89_TMS, name="etrs89_central_zoom3"),
    TileTestParam(tile=Tile(x=5, y=2, z=3), tms=ETRS89_TMS, name="etrs89_central_zoom3"),
]

# ETRS89 tiles - edge cases for integration tests (max 5)
ETRS89_TILES_EDGE_CASES = [
    # Higher zoom edge cases within Europe
    TileTestParam(tile=Tile(x=4, y=4, z=4), tms=ETRS89_TMS, name="etrs89_central_zoom4"),
    TileTestParam(
        tile=Tile(x=6, y=6, z=4), tms=ETRS89_TMS, name="etrs89_southeast_zoom4"
    ),
    # Small bbox test for ETRS89
    TileTestParam(tile=Tile(x=8, y=8, z=5), tms=ETRS89_TMS, name="etrs89_small_bbox"),
    TileTestParam(
        tile=Tile(x=15, y=12, z=5), tms=WEBMERC_TMS, name="webmerc_corner_zoom5"
    ),
    TileTestParam(tile=Tile(x=5, y=1, z=3), tms=WEBMERC_TMS, name="webmerc_corner_zoom3"),
]

# ETRS89 tiles (some may not be supported) - combined
ETRS89_TILES = ETRS89_TILES_REGULAR + ETRS89_TILES_EDGE_CASES

# WGS84 tiles - regular cases
WGS84_TILES_REGULAR = [
    # Equator tiles for testing WGS84 coordinate handling
    # At z=2, equator is between y=1 and y=2 in WGS84 projection
    TileTestParam(tile=Tile(x=1, y=1, z=2), tms=WGS84_TMS, name="wgs84_equator_north"),
    TileTestParam(tile=Tile(x=1, y=2, z=2), tms=WGS84_TMS, name="wgs84_equator_south"),
    # Prime meridian (0 degrees longitude) tiles
    TileTestParam(tile=Tile(x=2, y=1, z=2), tms=WGS84_TMS, name="wgs84_prime_meridian"),
    TileTestParam(tile=Tile(x=1, y=1, z=2), tms=WGS84_TMS, name="wgs84_prime_west"),
    TileTestParam(tile=Tile(x=3, y=1, z=2), tms=WGS84_TMS, name="wgs84_prime_east"),
    # Anti-meridian tiles (180/-180 degrees longitude)
    # At z=2, x=0 covers western edge near anti-meridian
    TileTestParam(
        tile=Tile(x=0, y=1, z=2), tms=WGS84_TMS, name="wgs84_antimeridian_west"
    ),
]

# WGS84 tiles - edge cases for integration tests (max 5)
WGS84_TILES_EDGE_CASES = [
    TileTestParam(
        tile=Tile(x=0, y=2, z=2), tms=WGS84_TMS, name="wgs84_antimeridian_west_equator"
    ),
    # Equator at anti-meridian - key test case for coordinate transformation
    TileTestParam(
        tile=Tile(x=0, y=1, z=3), tms=WGS84_TMS, name="wgs84_equator_antimeridian_z3"
    ),
    TileTestParam(
        tile=Tile(x=0, y=2, z=3),
        tms=WGS84_TMS,
        name="wgs84_equator_antimeridian_south_z3",
    ),
    # Higher zoom equator anti-meridian tiles
    TileTestParam(
        tile=Tile(x=0, y=15, z=5),
        tms=WGS84_TMS,
        name="wgs84_equator_antimeridian_north_z5",
    ),
    TileTestParam(
        tile=Tile(x=0, y=16, z=5),
        tms=WGS84_TMS,
        name="wgs84_equator_antimeridian_south_z5",
    ),
]

# WGS84 tiles for equator anti-meridian and 0 longitude testing - combined
WGS84_TILES = WGS84_TILES_REGULAR + WGS84_TILES_EDGE_CASES

# HRRR tiles - regular cases
# Actual HRRR domain: Lat(21.14, 47.84), Lon(-134.10, -60.92)
HRRR_TILES_REGULAR = [
    # Low zoom - full domain coverage
    TileTestParam(tile=Tile(x=0, y=1, z=2), tms=WEBMERC_TMS, name="hrrr_west_z2"),
    TileTestParam(tile=Tile(x=1, y=1, z=2), tms=WEBMERC_TMS, name="hrrr_east_z2"),
    # Medium zoom - domain corners and edges
    TileTestParam(tile=Tile(x=1, y=2, z=3), tms=WEBMERC_TMS, name="hrrr_sw_corner_z3"),
    TileTestParam(tile=Tile(x=2, y=2, z=3), tms=WEBMERC_TMS, name="hrrr_se_corner_z3"),
    TileTestParam(tile=Tile(x=1, y=2, z=3), tms=WEBMERC_TMS, name="hrrr_nw_corner_z3"),
    TileTestParam(tile=Tile(x=2, y=2, z=3), tms=WEBMERC_TMS, name="hrrr_ne_corner_z3"),
    TileTestParam(tile=Tile(x=1, y=3, z=3), tms=WEBMERC_TMS, name="hrrr_south_z3"),
    # Higher zoom - precise domain coverage
    TileTestParam(tile=Tile(x=4, y=11, z=5), tms=WEBMERC_TMS, name="hrrr_west_edge_z5"),
    TileTestParam(tile=Tile(x=10, y=11, z=5), tms=WEBMERC_TMS, name="hrrr_east_edge_z5"),
    TileTestParam(tile=Tile(x=4, y=13, z=5), tms=WEBMERC_TMS, name="hrrr_south_edge_z5"),
    TileTestParam(tile=Tile(x=7, y=11, z=5), tms=WEBMERC_TMS, name="hrrr_center_z5"),
    # Very high zoom cases
    TileTestParam(tile=Tile(x=16, y=44, z=7), tms=WEBMERC_TMS, name="hrrr_sw_precise_z7"),
    TileTestParam(tile=Tile(x=42, y=44, z=7), tms=WEBMERC_TMS, name="hrrr_se_precise_z7"),
    TileTestParam(tile=Tile(x=29, y=50, z=7), tms=WEBMERC_TMS, name="hrrr_center_z7"),
]

# HRRR tiles - edge cases for integration tests (max 5)
HRRR_TILES_EDGE_CASES = [
    # Ultra high zoom - precise boundaries (edge cases)
    TileTestParam(
        tile=Tile(x=130, y=356, z=10), tms=WEBMERC_TMS, name="hrrr_sw_extreme_z10"
    ),
    TileTestParam(
        tile=Tile(x=338, y=356, z=10), tms=WEBMERC_TMS, name="hrrr_se_extreme_z10"
    ),
    TileTestParam(tile=Tile(x=234, y=403, z=10), tms=WEBMERC_TMS, name="hrrr_center_z10"),
]

# HRRR tiles for testing Lambert Conformal Conic projection data - combined
HRRR_TILES = HRRR_TILES_REGULAR + HRRR_TILES_EDGE_CASES

# Para tiles - regular cases
# Para is approximately between 2.72°N to 9.93°S and 45.97°W to 58.99°W
PARA_TILES_REGULAR = [
    # Zoom level 4 - broader coverage
    TileTestParam(tile=Tile(x=5, y=7, z=4), tms=WEBMERC_TMS, name="para_north_z4"),
    TileTestParam(tile=Tile(x=5, y=8, z=4), tms=WEBMERC_TMS, name="para_south_z4"),
    # Zoom level 5 - more detailed coverage
    TileTestParam(tile=Tile(x=10, y=15, z=5), tms=WEBMERC_TMS, name="para_northwest_z5"),
    TileTestParam(tile=Tile(x=11, y=15, z=5), tms=WEBMERC_TMS, name="para_northeast_z5"),
    TileTestParam(tile=Tile(x=10, y=16, z=5), tms=WEBMERC_TMS, name="para_southwest_z5"),
    TileTestParam(tile=Tile(x=11, y=16, z=5), tms=WEBMERC_TMS, name="para_southeast_z5"),
    # Zoom level 6 - covering Belém (capital) area at ~1.5°S, 48.5°W
    TileTestParam(tile=Tile(x=22, y=31, z=6), tms=WEBMERC_TMS, name="para_belem_z6"),
    # Zoom level 7 - detailed view
    TileTestParam(tile=Tile(x=44, y=63, z=7), tms=WEBMERC_TMS, name="para_north_z7"),
    TileTestParam(tile=Tile(x=45, y=64, z=7), tms=WEBMERC_TMS, name="para_central_z7"),
    # Zoom level 8 - high detail for southern Para
    TileTestParam(tile=Tile(x=88, y=128, z=8), tms=WEBMERC_TMS, name="para_south_z8"),
]

# Para tiles - edge cases for integration tests (max 5)
PARA_TILES_EDGE_CASES = [
    # test upsampling at very high zoom levels - true edge cases
    TileTestParam(tile=Tile(x=1480, y=2064, z=12), tms=WEBMERC_TMS, name="para_south_z8"),
    TileTestParam(tile=Tile(x=2964, y=4129, z=13), tms=WEBMERC_TMS, name="para_south_z8"),
    TileTestParam(tile=Tile(x=5971, y=8252, z=14), tms=WEBMERC_TMS, name="para_south_z8"),
]

# Para (Brazilian state) tiles for testing South American region - combined
PARA_TILES = PARA_TILES_REGULAR + PARA_TILES_EDGE_CASES

# UTM Zone 33S tiles - regular cases covering southern Africa to Antarctica
UTM33S_TILES_REGULAR = [
    # Zoom 2 - Large coverage
    TileTestParam(tile=Tile(x=2, y=2, z=2), tms=WEBMERC_TMS, name="utm33s_africa_z2"),
    TileTestParam(tile=Tile(x=2, y=3, z=2), tms=WEBMERC_TMS, name="utm33s_antarctica_z2"),
    # Zoom 3 - Medium coverage
    TileTestParam(tile=Tile(x=4, y=4, z=3), tms=WEBMERC_TMS, name="utm33s_africa_z3"),
    TileTestParam(tile=Tile(x=4, y=5, z=3), tms=WEBMERC_TMS, name="utm33s_mid_z3"),
    TileTestParam(tile=Tile(x=4, y=6, z=3), tms=WEBMERC_TMS, name="utm33s_deep_z3"),
    # Zoom 4 - More detailed
    TileTestParam(tile=Tile(x=8, y=8, z=4), tms=WEBMERC_TMS, name="utm33s_north_z4"),
    # TileTestParam(tile=Tile(x=8, y=9, z=4), tms=WEBMERC_TMS, name="utm33s_central_z4"),
    TileTestParam(tile=Tile(x=8, y=10, z=4), tms=WEBMERC_TMS, name="utm33s_south_z4"),
    TileTestParam(
        tile=Tile(x=8, y=11, z=4), tms=WEBMERC_TMS, name="utm33s_antarctica_z4"
    ),
    TileTestParam(
        tile=Tile(x=8, y=14, z=4), tms=WEBMERC_TMS, name="utm33s_deep_antarctica_z4"
    ),
    # Zoom 5 - Detailed tiles
    TileTestParam(tile=Tile(x=17, y=16, z=5), tms=WEBMERC_TMS, name="utm33s_equator_z5"),
    TileTestParam(tile=Tile(x=17, y=17, z=5), tms=WEBMERC_TMS, name="utm33s_north_z5"),
    TileTestParam(tile=Tile(x=17, y=18, z=5), tms=WEBMERC_TMS, name="utm33s_central_z5"),
    TileTestParam(tile=Tile(x=17, y=20, z=5), tms=WEBMERC_TMS, name="utm33s_south_z5"),
    TileTestParam(
        tile=Tile(x=17, y=23, z=5), tms=WEBMERC_TMS, name="utm33s_antarctica_z5"
    ),
    # TileTestParam(tile=Tile(x=17, y=25, z=5), tms=WEBMERC_TMS, name="utm33s_deep_z5"),
    TileTestParam(
        tile=Tile(x=17, y=22, z=5), tms=WEBMERC_TMS, name="utm33s_mid_antarctica_z5"
    ),
]

# UTM Zone 33S tiles - edge cases for equator and Antarctica boundaries
UTM33S_TILES_EDGE_CASES = [
    # Equator edge cases (northern boundary at 0°)
    TileTestParam(tile=Tile(x=34, y=32, z=6), tms=WEBMERC_TMS, name="utm33s_equator_z6"),
    TileTestParam(tile=Tile(x=68, y=64, z=7), tms=WEBMERC_TMS, name="utm33s_equator_z7"),
    TileTestParam(
        tile=Tile(x=136, y=128, z=8), tms=WEBMERC_TMS, name="utm33s_equator_z8"
    ),
    # Antarctica edge cases (southern boundary at -80°S)
    TileTestParam(
        tile=Tile(x=17, y=28, z=5), tms=WEBMERC_TMS, name="utm33s_antarctica_edge_z5"
    ),
    TileTestParam(
        tile=Tile(x=34, y=56, z=6), tms=WEBMERC_TMS, name="utm33s_antarctica_edge_z6"
    ),
    # TileTestParam(tile=Tile(x=68, y=112, z=7), tms=WEBMERC_TMS, name="utm33s_antarctica_edge_z7"),
    # TileTestParam(tile=Tile(x=136, y=224, z=8), tms=WEBMERC_TMS, name="utm33s_antarctica_edge_z8"),
    # Very high zoom equator and Antarctica
    TileTestParam(
        tile=Tile(x=277, y=256, z=9), tms=WEBMERC_TMS, name="utm33s_equator_z9"
    ),
    TileTestParam(
        tile=Tile(x=277, y=448, z=9), tms=WEBMERC_TMS, name="utm33s_antarctica_z9"
    ),
    TileTestParam(
        tile=Tile(x=4372, y=4160, z=13), tms=WEBMERC_TMS, name="utm33s_center_swatch_z9"
    ),
]

# UTM Zone 33S tiles - combined
UTM33S_TILES = UTM33S_TILES_REGULAR + UTM33S_TILES_EDGE_CASES

# Curvilinear tiles - for testing curvilinear coordinate data
CURVILINEAR_TILES = [
    TileTestParam(
        tile=Tile(x=3, y=5, z=4), tms=WEBMERC_TMS, name="curvilinear_hrrr_east_z4"
    ),
    TileTestParam(
        tile=Tile(x=7, y=12, z=5), tms=WEBMERC_TMS, name="curvilinear_hrrr_sw_corner_z5"
    ),
    TileTestParam(
        tile=Tile(x=6, y=11, z=5), tms=WEBMERC_TMS, name="curvilinear_hrrr_se_corner_z5"
    ),
    TileTestParam(
        tile=Tile(x=27, y=48, z=7), tms=WEBMERC_TMS, name="curvilinear_hrrr_central_z7"
    ),
    TileTestParam(
        tile=Tile(x=15, y=24, z=6), tms=WEBMERC_TMS, name="curvilinear_central_us_z6"
    ),
    TileTestParam(
        tile=Tile(x=442, y=744, z=11), tms=WEBMERC_TMS, name="curvilinear_central_us_z11"
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
