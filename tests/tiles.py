import morecantile
import pytest

# WebMercatorQuad TMS for creating tiles
_TMS = morecantile.tms.get("WebMercatorQuad")
# EuropeanETRS89_LAEAQuad TMS for ETRS89 LAEA CRS
_ETRS89_TMS = morecantile.tms.get("EuropeanETRS89_LAEAQuad")

TILES = [
    # WebMercatorQuad tiles - European region focus to avoid anti-meridian issues
    pytest.param(
        morecantile.Tile(x=2, y=1, z=2), _TMS, id="webmerc_europe_center(2/2/1)"
    ),
    pytest.param(morecantile.Tile(x=1, y=1, z=2), _TMS, id="webmerc_europe_west(2/1/1)"),
    # Note: webmerc_europe_east(2/3/1) removed - causes anti-meridian crossing when projected to ETRS89 LAEA
    pytest.param(morecantile.Tile(x=2, y=0, z=2), _TMS, id="webmerc_europe_north(2/2/0)"),
    pytest.param(morecantile.Tile(x=2, y=2, z=2), _TMS, id="webmerc_europe_south(2/2/2)"),
    # Higher zoom European region
    pytest.param(morecantile.Tile(x=8, y=5, z=4), _TMS, id="webmerc_europe_zoom4(4/8/5)"),
    pytest.param(
        morecantile.Tile(x=16, y=10, z=5), _TMS, id="webmerc_europe_zoom5(5/16/10)"
    ),
    # Small bbox test
    pytest.param(morecantile.Tile(x=8, y=8, z=5), _TMS, id="webmerc_small_bbox(5/8/8)"),
    # ETRS89 LAEA tiles - European region specific
    # Center of Europe tiles
    pytest.param(
        morecantile.Tile(x=1, y=1, z=2), _ETRS89_TMS, id="etrs89_center_europe(2/1/1)"
    ),
    pytest.param(
        morecantile.Tile(x=0, y=1, z=2), _ETRS89_TMS, id="etrs89_west_europe(2/0/1)"
    ),
    pytest.param(
        morecantile.Tile(x=2, y=1, z=2), _ETRS89_TMS, id="etrs89_east_europe(2/2/1)"
    ),
    # Northern Europe (Scandinavia region)
    pytest.param(
        morecantile.Tile(x=1, y=0, z=2), _ETRS89_TMS, id="etrs89_north_europe(2/1/0)"
    ),
    # Southern Europe (Mediterranean region)
    pytest.param(
        morecantile.Tile(x=1, y=2, z=2), _ETRS89_TMS, id="etrs89_south_europe(2/1/2)"
    ),
    # Higher zoom edge cases within Europe
    pytest.param(
        morecantile.Tile(x=4, y=4, z=4), _ETRS89_TMS, id="etrs89_central_zoom4(4/4/4)"
    ),
    pytest.param(
        morecantile.Tile(x=2, y=2, z=3), _ETRS89_TMS, id="etrs89_central_zoom3(3/2/2)"
    ),
    pytest.param(
        morecantile.Tile(x=6, y=6, z=4), _ETRS89_TMS, id="etrs89_southeast_zoom4(4/6/6)"
    ),
    # Small bbox test for ETRS89
    pytest.param(
        morecantile.Tile(x=8, y=8, z=5), _ETRS89_TMS, id="etrs89_small_bbox(5/8/8)"
    ),
]
