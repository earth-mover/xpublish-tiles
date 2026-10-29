"""Mapbox Vector Tile (MVT 2.1) encoding of polygon render contexts.

Each variable becomes one layer named after it. Each feature carries the data
``value`` and, when a colour scale is known, the ``color`` the polygons style
would paint it, so a client may style by either. ``Features`` grids also set
the feature ``id`` to its position along the feature dim.

Needs the ``[vector]`` extra; import this module lazily from core code.
"""

import io
import struct
from itertools import pairwise
from numbers import Number

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
from spatialpandas.geometry import MultiPolygonArray

from xpublish_tiles.grids import Features, GeometryIndexer
from xpublish_tiles.lib import apply_range_colors, create_colormap_from_dict
from xpublish_tiles.logger import get_context_logger, log_duration
from xpublish_tiles.render import discrete_color_key
from xpublish_tiles.render.polygons import PolygonsRenderer
from xpublish_tiles.types import (
    DEFAULT_MARKERSIZE,
    ContinuousData,
    DataType,
    DiscreteData,
    GeometryArray,
    ImageFormat,
    NullRenderContext,
    OutputBBox,
    PopulatedRenderContext,
    RenderContext,
    TileFormat,
)

try:
    import shapely
    from shapely import GeometryType
except ImportError as e:
    raise ImportError(
        "Vector tiles require shapely; install `xpublish-tiles[vector]`."
    ) from e

EXTENT = 4096
# Tile units kept past each edge, so clipped polygon edges stay off-screen.
BUFFER = 64
# Tile units; well under a pixel at any display size.
SIMPLIFY_TOLERANCE = 1.0

_MOVE_TO = 1 | (1 << 3)
_LINE_TO = 2
_CLOSE_PATH = 7 | (1 << 3)
_POLYGON = 3

_POLYGONAL = (GeometryType.POLYGON, GeometryType.MULTIPOLYGON)


def _varint(n: int) -> bytes:
    out = bytearray()
    while n > 0x7F:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)


def _field(number: int, payload: bytes) -> bytes:
    """A length-delimited protobuf field."""
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _varints(values: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Varint-encode ``values``; return the bytes and each value's end offset."""
    values = np.asarray(values, dtype=np.uint64)
    nbytes = np.ones(values.shape, dtype=np.int64)
    for i in range(1, 10):
        nbytes += values >= np.uint64(1 << (7 * i))
    ends = np.cumsum(nbytes)
    starts = ends - nbytes
    out = np.empty(int(ends[-1]) if ends.size else 0, dtype=np.uint8)
    for i in range(int(nbytes.max(initial=0))):
        has = nbytes > i
        septet = (values[has] >> np.uint64(7 * i)) & np.uint64(0x7F)
        more = (nbytes[has] > i + 1).astype(np.uint64) << np.uint64(7)
        out[starts[has] + i] = septet | more
    return out.tobytes(), ends


def _zigzag(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.int64)
    return ((v << 1) ^ (v >> 63)).astype(np.uint64)


def _to_tile_geometries(polygons: GeometryArray, bbox: OutputBBox) -> np.ndarray:
    """Output-CRS polygons as shapely geometries in tile units (y down)."""
    # spatialpandas offsets run outermost first; coordinate offsets count floats.
    offsets = [np.asarray(o, dtype=np.int64) for o in polygons.buffer_offsets]
    offsets[-1] = offsets[-1] // 2
    xy = np.asarray(polygons.buffer_values, dtype=np.float64).reshape(-1, 2)
    coords = np.empty_like(xy)
    coords[:, 0] = (xy[:, 0] - bbox.west) * (EXTENT / (bbox.east - bbox.west))
    coords[:, 1] = (bbox.north - xy[:, 1]) * (EXTENT / (bbox.north - bbox.south))
    kind = (
        GeometryType.MULTIPOLYGON
        if isinstance(polygons, MultiPolygonArray)
        else GeometryType.POLYGON
    )
    return shapely.from_ragged_array(kind, coords, tuple(offsets[::-1]))


def _polygonal(geom: "shapely.Geometry") -> "shapely.Geometry":
    """Polygonal parts of a collection that clipping left lines or points in."""
    parts = shapely.get_parts(geom)
    parts = parts[np.isin(shapely.get_type_id(parts), _POLYGONAL)]
    return shapely.multipolygons(shapely.get_parts(parts))


def prepare_geometries(geoms: np.ndarray) -> np.ndarray:
    """Simplify, clip to the buffered tile and snap to the integer grid.

    Exteriors wind positive in tile coordinates (clockwise on screen), as MVT
    requires. Entries that vanish come back empty.
    """
    detailed = shapely.get_num_coordinates(geoms) > 5
    if detailed.any():
        geoms[detailed] = shapely.simplify(geoms[detailed], SIMPLIFY_TOLERANCE)

    lo, hi = -BUFFER, EXTENT + BUFFER
    bounds = shapely.bounds(geoms)
    crossing = (bounds[:, :2] < lo).any(axis=1) | (bounds[:, 2:] > hi).any(axis=1)
    if crossing.any():
        geoms[crossing] = shapely.intersection(
            geoms[crossing], shapely.box(lo, lo, hi, hi), grid_size=1.0
        )
    if (~crossing).any():
        geoms[~crossing] = shapely.set_precision(geoms[~crossing], 1.0)

    collections = shapely.get_type_id(geoms) == GeometryType.GEOMETRYCOLLECTION
    for i in np.flatnonzero(collections & ~shapely.is_empty(geoms)):
        geoms[i] = _polygonal(geoms[i])
    return shapely.orient_polygons(geoms, exterior_cw=False)


def encode_geometries(geoms: np.ndarray) -> list[bytes]:
    """Packed MVT polygon command streams, one per non-empty (multi)polygon."""
    geom_type, coords, offsets = shapely.to_ragged_array(geoms)
    if geom_type == GeometryType.POLYGON:
        ring_offsets, geom_offsets = offsets
        part_offsets = np.arange(len(geom_offsets))
    else:
        ring_offsets, part_offsets, geom_offsets = offsets
    ring_offsets = np.asarray(ring_offsets, dtype=np.int64)
    feature_rings = np.asarray(part_offsets, dtype=np.int64)[
        np.asarray(geom_offsets, dtype=np.int64)
    ]

    # Rings close implicitly (ClosePath), so drop each closing vertex.
    keep = np.ones(len(coords), dtype=bool)
    keep[ring_offsets[1:] - 1] = False
    points = np.rint(coords[keep]).astype(np.int64)
    npoints = np.diff(ring_offsets) - 1

    # The cursor carries across rings and parts, and restarts at (0, 0) per feature.
    deltas = np.diff(points, axis=0, prepend=np.zeros((1, 2), dtype=np.int64))
    point_starts = np.concatenate([[0], np.cumsum(npoints)])
    feature_first_point = point_starts[feature_rings[:-1]]
    deltas[feature_first_point] = points[feature_first_point]

    lengths = 2 * npoints + 3
    ring_starts = np.concatenate([[0], np.cumsum(lengths)])
    stream = np.empty(int(ring_starts[-1]), dtype=np.uint64)
    stream[ring_starts[:-1]] = _MOVE_TO
    stream[ring_starts[:-1] + 3] = _LINE_TO | ((npoints - 1) << 3)
    stream[ring_starts[1:] - 1] = _CLOSE_PATH

    ring_of_point = np.repeat(np.arange(len(npoints)), npoints)
    k = np.arange(len(points)) - point_starts[ring_of_point]
    xpos = ring_starts[ring_of_point] + 2 * k + 2 - (k == 0)
    stream[xpos] = _zigzag(deltas[:, 0])
    stream[xpos + 1] = _zigzag(deltas[:, 1])

    packed, ends = _varints(stream)
    byte_offsets = np.concatenate([[0], ends])[ring_starts[feature_rings]]
    return [packed[a:b] for a, b in pairwise(byte_offsets)]


def feature_colors(
    values: np.ndarray,
    datatype: DataType,
    *,
    variant: str,
    colorscalerange: tuple[Number, Number] | None = None,
    colormap: dict[str, str] | None = None,
    abovemaxcolor: str | None = None,
    belowmincolor: str | None = None,
) -> np.ndarray | None:
    """``(N, 4)`` RGBA bytes the polygons style would paint ``values``.

    ``None`` when continuous data has no colour scale: neither a
    ``colorscalerange`` nor ``valid_min``/``valid_max``.
    """
    if isinstance(datatype, ContinuousData):
        if colorscalerange is None:
            if datatype.valid_min is None or datatype.valid_max is None:
                return None
            colorscalerange = (datatype.valid_min, datatype.valid_max)
        if colormap is not None:
            cmap = create_colormap_from_dict(colormap)
        else:
            cmap = mpl.colormaps.get_cmap(variant)
        cmap = apply_range_colors(cmap, abovemaxcolor, belowmincolor)
        vmin, vmax = colorscalerange
        norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))  # ty: ignore[invalid-argument-type]
        return np.rint(cmap(norm(values)) * 255).astype(np.uint8)
    if isinstance(datatype, DiscreteData):
        key = discrete_color_key(datatype, variant=variant, colormap=colormap)
        uniques, inverse = np.unique(values, return_inverse=True)
        lut = np.array(
            [
                mcolors.to_rgba(key[v]) if v in key else (0.0, 0.0, 0.0, 0.0)
                for v in uniques.tolist()
            ]
        )
        return np.rint(lut * 255).astype(np.uint8)[inverse]
    raise NotImplementedError(f"Unsupported datatype: {type(datatype)}")


def _hex_colors(rgba: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Distinct ``#rrggbb[aa]`` strings, and each row's index into them."""
    packed = rgba.astype(np.uint32) @ np.array([1 << 24, 1 << 16, 1 << 8, 1], np.uint32)
    uniques, inverse = np.unique(packed, return_inverse=True)
    names = [
        f"#{c >> 8:06x}" if c & 0xFF == 0xFF else f"#{c:08x}" for c in uniques.tolist()
    ]
    return names, inverse


def _value_messages(values: np.ndarray) -> tuple[list[bytes], np.ndarray]:
    """Distinct MVT ``Value`` messages for ``values``, and each one's index."""
    uniques, inverse = np.unique(values, return_inverse=True)
    if values.dtype.kind == "f" and values.dtype.itemsize <= 4:
        messages = [b"\x15" + struct.pack("<f", v) for v in uniques.tolist()]
    elif values.dtype.kind == "f":
        messages = [b"\x19" + struct.pack("<d", v) for v in uniques.tolist()]
    elif values.dtype.kind == "b":
        messages = [b"\x38" + _varint(int(v)) for v in uniques.tolist()]
    elif values.dtype.kind == "u":
        messages = [b"\x28" + _varint(v) for v in uniques.tolist()]
    elif values.dtype.kind == "i":
        messages = [b"\x30" + _varint((v << 1) ^ (v >> 63)) for v in uniques.tolist()]
    else:
        raise TypeError(f"Cannot encode {values.dtype} values in a vector tile.")
    return messages, inverse


def encode_layer(
    name: str,
    geometries: list[bytes],
    values: np.ndarray,
    *,
    colors: np.ndarray | None = None,
    ids: np.ndarray | None = None,
) -> bytes:
    """One MVT layer of polygon features with ``value`` and optional ``color``."""
    keys = ["value"]
    value_messages, value_index = _value_messages(values)
    tag_columns = [np.zeros_like(value_index), value_index]
    if colors is not None:
        keys.append("color")
        names, color_index = _hex_colors(colors)
        color_offset = len(value_messages)
        value_messages += [_field(1, n.encode()) for n in names]
        tag_columns += [np.ones_like(color_index), color_index + color_offset]
    packed_tags, tag_ends = _varints(np.stack(tag_columns, axis=1).ravel())
    tag_offsets = np.concatenate(
        [[0], tag_ends[len(tag_columns) - 1 :: len(tag_columns)]]
    )

    layer = bytearray(b"\x78\x02")  # version = 2
    layer += _field(1, name.encode())
    for i, geometry in enumerate(geometries):
        feature = bytearray()
        if ids is not None:
            feature += b"\x08" + _varint(int(ids[i]))
        feature += _field(2, packed_tags[tag_offsets[i] : tag_offsets[i + 1]])
        feature += b"\x18" + _varint(_POLYGON)
        feature += _field(4, geometry)
        layer += _field(2, bytes(feature))
    for key in keys:
        layer += _field(3, key.encode())
    for message in value_messages:
        layer += _field(4, message)
    layer += b"\x28" + _varint(EXTENT)
    return bytes(layer)


def _feature_ids(context: PopulatedRenderContext) -> np.ndarray | None:
    if not isinstance(context.grid, Features):
        return None
    (patch,) = context.patches
    assert isinstance(patch.indexer, GeometryIndexer)
    return patch.indexer.indices


class MVTRenderer(PolygonsRenderer):
    """The polygons style's features, encoded as a Mapbox Vector Tile.

    Chosen by ``f=mvt`` rather than by style, so it is not in the renderer
    registry; its ``polygons`` style id selects the polygon geometry path.
    """

    def render(
        self,
        *,
        contexts: dict[str, RenderContext],
        buffer: io.BytesIO,
        width: int,
        height: int,
        variant: str,
        colorscalerange: tuple[Number, Number] | None = None,
        format: ImageFormat | TileFormat = TileFormat.MVT,
        context_logger=None,
        colormap: dict[str, str] | None = None,
        abovemaxcolor: str | None = None,
        belowmincolor: str | None = None,
        edgecolor: str | None = None,
        edgewidth: int = 1,
        markersize: int = DEFAULT_MARKERSIZE,
    ) -> None:
        """Write one MVT layer per variable with polygons in the tile to ``buffer``."""
        logger = context_logger if context_logger is not None else get_context_logger()
        if variant == "default":
            variant = self.default_variant()

        for name, context in contexts.items():
            if isinstance(context, NullRenderContext):
                continue
            assert isinstance(context, PopulatedRenderContext)
            assert context.polygons is not None and context.da is not None
            if len(context.polygons) == 0:
                continue

            with log_duration(
                f"mvt geometry {len(context.polygons)}", logger, key="mvt_geometry"
            ):
                geoms = prepare_geometries(
                    _to_tile_geometries(context.polygons, context.bbox)
                )
            values = np.asarray(context.da.values)
            keep = ~shapely.is_empty(geoms)
            if values.dtype.kind == "f":
                keep &= np.isfinite(values)
            if not keep.any():
                continue

            values = values[keep]
            ids = _feature_ids(context)
            colors = feature_colors(
                values,
                context.datatype,
                variant=variant,
                colorscalerange=colorscalerange,
                colormap=colormap,
                abovemaxcolor=abovemaxcolor,
                belowmincolor=belowmincolor,
            )
            with log_duration(f"encode mvt {keep.sum()}", logger, key="encode"):
                layer = encode_layer(
                    str(name),
                    encode_geometries(geoms[keep]),
                    values,
                    colors=colors,
                    ids=None if ids is None else ids[keep],
                )
                buffer.write(_field(3, layer))

    def render_error(self, **kwargs):
        raise NotImplementedError("Vector tiles report errors as HTTP errors.")
