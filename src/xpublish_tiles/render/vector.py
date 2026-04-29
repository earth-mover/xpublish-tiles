"""Vector tile renderer (MVT + GeoJSON).

Reuses the polygons-style geometry pipeline (cell rings + per-cell values
already in the output CRS), then either:

* MVT: quantize rings to the tile-local integer grid, encode each polygon as
  an MVT command stream, frame as protobuf, gzip.
* GeoJSON: reproject rings to WGS84 (RFC 7946) and serialize as a
  FeatureCollection.

Width/height are not meaningful for vector tiles and are ignored; coarsening
targets ``config["mvt_extent"]`` (default 4096) via the ``vector`` branch in
``max_render_shape``.
"""

from __future__ import annotations

import gzip
import io
import json
import struct
from numbers import Number

import numba
import numpy as np
import pyproj

from xpublish_tiles.config import config
from xpublish_tiles.lib import transformer_from_crs
from xpublish_tiles.logger import get_context_logger, log_duration
from xpublish_tiles.render import Renderer, register_renderer, render_error_image
from xpublish_tiles.types import (
    ImageFormat,
    NullRenderContext,
    PopulatedRenderContext,
    RenderContext,
)
from xpublish_tiles.utils import NUMBA_THREADING_LOCK

# MVT command ids (vector-tile-spec 2.1)
_CMD_MOVETO = 1
_CMD_LINETO = 2
_CMD_CLOSEPATH = 7

# MVT geometry types
_GEOM_POLYGON = 3

# Protobuf wire types
_WIRE_VARINT = 0
_WIRE_LEN = 2

# Field numbers
_TILE_LAYERS = 3
_LAYER_NAME = 1
_LAYER_FEATURES = 2
_LAYER_KEYS = 3
_LAYER_VALUES = 4
_LAYER_EXTENT = 5
_LAYER_VERSION = 15
_FEATURE_ID = 1
_FEATURE_TAGS = 2
_FEATURE_TYPE = 3
_FEATURE_GEOMETRY = 4
_VALUE_DOUBLE = 3


def _tag(field_num: int, wire_type: int) -> int:
    return (field_num << 3) | wire_type


def _write_varint(buf: bytearray, n: int) -> None:
    """Write a non-negative integer as a protobuf varint."""
    while n > 0x7F:
        buf.append((n & 0x7F) | 0x80)
        n >>= 7
    buf.append(n & 0x7F)


def _write_tag(buf: bytearray, field_num: int, wire_type: int) -> None:
    _write_varint(buf, _tag(field_num, wire_type))


def _write_string(buf: bytearray, field_num: int, s: str) -> None:
    data = s.encode("utf-8")
    _write_tag(buf, field_num, _WIRE_LEN)
    _write_varint(buf, len(data))
    buf.extend(data)


def _write_uint32(buf: bytearray, field_num: int, n: int) -> None:
    _write_tag(buf, field_num, _WIRE_VARINT)
    _write_varint(buf, n)


def _write_double(buf: bytearray, field_num: int, x: float) -> None:
    _write_tag(buf, field_num, 1)  # 64-bit fixed
    buf.extend(struct.pack("<d", x))


def _write_len_delim(buf: bytearray, field_num: int, body: bytes | bytearray) -> None:
    _write_tag(buf, field_num, _WIRE_LEN)
    _write_varint(buf, len(body))
    buf.extend(body)


def _quantize_rings(rings: np.ndarray, *, bbox, extent: int) -> np.ndarray:
    """Float rings in output CRS → int32 rings in MVT tile-local space.

    MVT origin is top-left, Y increases downward.
    """
    sx = extent / (bbox.east - bbox.west)
    sy = extent / (bbox.north - bbox.south)
    # rings: (N, M, 2). Build float scratch then round to int32.
    qx = (rings[..., 0] - bbox.west) * sx
    qy = (bbox.north - rings[..., 1]) * sy
    out = np.empty(rings.shape, dtype=np.int32)
    np.rint(qx, out=qx)
    np.rint(qy, out=qy)
    out[..., 0] = qx
    out[..., 1] = qy
    return out


@numba.njit(cache=True, boundscheck=False)
def _encode_polygons(
    q: np.ndarray,
    valid: np.ndarray,
    cmd_buf: np.ndarray,
    cmd_len: np.ndarray,
):
    """Per-polygon: emit MoveTo(v0) + LineTo(v1..) + ClosePath as uint32 cmd stream.

    ``q`` is ``(N, M, 2)`` int32 quantized rings. The closing vertex
    ``q[i, M-1]`` equals ``q[i, 0]`` and is consumed implicitly by ClosePath.
    Consecutive duplicates after quantization are dropped. Polygons with
    fewer than 3 distinct vertices are marked invalid.

    Outputs:
      ``cmd_buf[i, :cmd_len[i]]`` — uint32 cmd+param stream for polygon ``i``.
      ``valid[i]`` — 1 if polygon should be emitted, else 0.
    """
    n, m, _ = q.shape
    for i in range(n):
        # First-pass: collect distinct vertex indices in [0, m-2]
        # (skip the closing vertex at m-1 since it duplicates v0).
        # We write directly into cmd_buf to avoid an extra buffer.
        # Layout: [moveto_cmd, dx, dy, lineto_cmd, dx1, dy1, ..., closepath_cmd]
        prev_x = q[i, 0, 0]
        prev_y = q[i, 0, 1]
        # MoveTo command at slot 0; deltas at 1, 2.
        cmd_buf[i, 0] = (_CMD_MOVETO & 0x7) | (1 << 3)
        cmd_buf[i, 1] = (prev_x << 1) ^ (prev_x >> 31)
        cmd_buf[i, 2] = (prev_y << 1) ^ (prev_y >> 31)
        # LineTo command placeholder at slot 3; deltas start at slot 4.
        write_idx = 4
        n_lineto = 0
        for k in range(1, m - 1):
            cur_x = q[i, k, 0]
            cur_y = q[i, k, 1]
            if cur_x == prev_x and cur_y == prev_y:
                continue
            dx = cur_x - prev_x
            dy = cur_y - prev_y
            cmd_buf[i, write_idx] = (dx << 1) ^ (dx >> 31)
            cmd_buf[i, write_idx + 1] = (dy << 1) ^ (dy >> 31)
            write_idx += 2
            n_lineto += 1
            prev_x = cur_x
            prev_y = cur_y
        if n_lineto < 2:
            valid[i] = 0
            cmd_len[i] = 0
            continue
        cmd_buf[i, 3] = (_CMD_LINETO & 0x7) | (n_lineto << 3)
        cmd_buf[i, write_idx] = (_CMD_CLOSEPATH & 0x7) | (1 << 3)
        cmd_len[i] = write_idx + 1
        valid[i] = 1


def _encode_mvt_layer(
    *,
    name: str,
    extent: int,
    rings_q: np.ndarray,
    values: np.ndarray,
) -> bytes:
    """Build a single MVT Layer protobuf body.

    Each feature gets a single tag (key="value", value=feature's data value
    encoded as a Value.double_value). Features whose data value is NaN are
    omitted entirely (matches transparent rendering semantics).
    """
    n, m, _ = rings_q.shape
    cmd_stride = 2 * m + 1  # upper bound: MoveTo(1+2) + LineTo(1+2*(m-2)) + Close(1)
    cmd_buf = np.zeros((n, cmd_stride), dtype=np.uint32)
    cmd_len = np.zeros(n, dtype=np.int32)
    valid = np.zeros(n, dtype=np.uint8)

    with NUMBA_THREADING_LOCK:
        _encode_polygons(rings_q, valid, cmd_buf, cmd_len)

    finite = np.isfinite(values)
    valid &= finite.view(np.uint8)

    body = bytearray()
    _write_uint32(body, _LAYER_VERSION, 2)
    _write_string(body, _LAYER_NAME, name)
    _write_uint32(body, _LAYER_EXTENT, extent)

    # Build the values table: one Value entry per emitted feature with that
    # feature's data value. (Float dedup is rarely a win for continuous data;
    # discrete data could be optimized later.)
    value_buf = bytearray()
    feature_value_idx = np.full(n, -1, dtype=np.int32)
    next_idx = 0
    for i in range(n):
        if not valid[i]:
            continue
        feature_value_idx[i] = next_idx
        value_body = bytearray()
        _write_double(value_body, _VALUE_DOUBLE, float(values[i]))
        value_buf.extend(_emit_len_delim(_LAYER_VALUES, value_body))
        next_idx += 1

    _write_string(body, _LAYER_KEYS, "value")
    body.extend(value_buf)

    # Features
    for i in range(n):
        if not valid[i]:
            continue
        feat = bytearray()
        _write_uint32(feat, _FEATURE_ID, i + 1)
        # tags = packed [key_idx=0, value_idx]
        tags_payload = bytearray()
        _write_varint(tags_payload, 0)
        _write_varint(tags_payload, int(feature_value_idx[i]))
        _write_len_delim(feat, _FEATURE_TAGS, tags_payload)
        _write_uint32(feat, _FEATURE_TYPE, _GEOM_POLYGON)
        # Geometry: packed uint32 cmd stream
        geom_payload = bytearray()
        L = int(cmd_len[i])
        for k in range(L):
            _write_varint(geom_payload, int(cmd_buf[i, k]))
        _write_len_delim(feat, _FEATURE_GEOMETRY, geom_payload)
        _write_len_delim(body, _LAYER_FEATURES, feat)

    return bytes(body)


def _emit_len_delim(field_num: int, body: bytes | bytearray) -> bytes:
    """Encode ``tag + len + body`` as a standalone bytes object."""
    out = bytearray()
    _write_tag(out, field_num, _WIRE_LEN)
    _write_varint(out, len(body))
    out.extend(body)
    return bytes(out)


def _encode_mvt_tile(layers: list[bytes]) -> bytes:
    """Wrap one or more pre-encoded layer bodies as a Tile message."""
    out = bytearray()
    for layer_body in layers:
        out.extend(_emit_len_delim(_TILE_LAYERS, layer_body))
    return bytes(out)


def _rings_to_geojson_features(
    rings_4326: np.ndarray, values: np.ndarray, *, var_name: str
) -> list[dict]:
    """Build GeoJSON Feature dicts from (N, M, 2) WGS84 rings."""
    n, m, _ = rings_4326.shape
    features: list[dict] = []
    for i in range(n):
        v = values[i]
        if not np.isfinite(v):
            continue
        coords = rings_4326[i].tolist()
        # Drop trailing repeats so the ring closes cleanly. We always keep
        # the first vertex repeated as the last to satisfy GeoJSON's closed-
        # ring requirement.
        # Find the last vertex that differs from the first; trim padding.
        last = m - 1
        while last > 1 and coords[last] == coords[last - 1]:
            last -= 1
        ring = coords[: last + 1]
        # Ensure closure: the final coord must equal the first.
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        if len(ring) < 4:
            continue
        features.append(
            {
                "type": "Feature",
                "id": i,
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {var_name: float(v)},
            }
        )
    return features


def _reproject_rings_to_wgs84(rings: np.ndarray, source_crs: pyproj.CRS) -> np.ndarray:
    """Reproject (N, M, 2) rings from source_crs to WGS84 (lon, lat)."""
    if source_crs.equals(pyproj.CRS.from_epsg(4326)):
        return rings
    transformer = transformer_from_crs(source_crs, pyproj.CRS.from_epsg(4326))
    flat_x = rings[..., 0].ravel()
    flat_y = rings[..., 1].ravel()
    lon, lat = transformer.transform(flat_x, flat_y)
    out = np.empty_like(rings)
    out[..., 0] = np.asarray(lon).reshape(rings.shape[:-1])
    out[..., 1] = np.asarray(lat).reshape(rings.shape[:-1])
    return out


@register_renderer
class VectorTileRenderer(Renderer):
    """Renderer that emits MVT or GeoJSON from the polygons geometry pipeline."""

    def render(
        self,
        *,
        contexts: dict[str, RenderContext],
        buffer: io.BytesIO,
        width: int,
        height: int,
        variant: str,
        colorscalerange: tuple[Number, Number] | None = None,
        format: ImageFormat = ImageFormat.MVT,
        context_logger=None,
        colormap: dict[str, str] | None = None,
        abovemaxcolor: str | None = None,
        belowmincolor: str | None = None,
    ):
        logger = context_logger if context_logger is not None else get_context_logger()

        if format not in self.supported_formats():
            raise ValueError(
                f"VectorTileRenderer cannot emit {format!r}; supported: "
                f"{sorted(f.value for f in self.supported_formats())}"
            )

        layers: list[bytes] = []
        geojson_features: list[dict] = []

        for var_name, context in contexts.items():
            if isinstance(context, NullRenderContext):
                continue
            assert isinstance(context, PopulatedRenderContext)
            if context.cell_rings is None or len(context.cell_rings) == 0:
                continue
            assert context.da is not None
            rings = np.asarray(context.cell_rings)
            values = np.asarray(context.da.values).ravel()

            if format is ImageFormat.MVT:
                extent = config.get("mvt_extent")
                with log_duration(
                    f"vector quantize+encode {rings.shape[0]} polys", "▦", logger
                ):
                    rings_q = _quantize_rings(rings, bbox=context.bbox, extent=extent)
                    layer_body = _encode_mvt_layer(
                        name=str(var_name),
                        extent=extent,
                        rings_q=rings_q,
                        values=values,
                    )
                layers.append(layer_body)
            elif format is ImageFormat.GEOJSON:
                if context.crs is None:
                    raise ValueError(
                        "GeoJSON output requires the render context CRS to be set."
                    )
                with log_duration(f"vector geojson {rings.shape[0]} polys", "▦", logger):
                    rings_4326 = _reproject_rings_to_wgs84(rings, context.crs)
                    geojson_features.extend(
                        _rings_to_geojson_features(
                            rings_4326, values, var_name=str(var_name)
                        )
                    )

        if format is ImageFormat.MVT:
            tile = _encode_mvt_tile(layers)
            buffer.write(gzip.compress(tile, compresslevel=1))
        else:
            payload = json.dumps(
                {"type": "FeatureCollection", "features": geojson_features},
                allow_nan=False,
                separators=(",", ":"),
            )
            buffer.write(payload.encode("utf-8"))

    def render_error(
        self,
        *,
        buffer: io.BytesIO,
        width: int,
        height: int,
        message: str,
        format: ImageFormat = ImageFormat.MVT,
        cmap: str = "",
        colorscalerange: tuple[Number, Number] | None = None,
        **kwargs,
    ) -> None:
        if format is ImageFormat.GEOJSON:
            payload = json.dumps(
                {"type": "FeatureCollection", "features": [], "error": message},
                separators=(",", ":"),
            )
            buffer.write(payload.encode("utf-8"))
        elif format is ImageFormat.MVT:
            # An empty MVT tile is a valid (zero-byte after gzip) response.
            buffer.write(gzip.compress(b"", compresslevel=1))
        else:
            error_buffer = render_error_image(
                message, width=width or 256, height=height or 256, format=format
            )
            buffer.write(error_buffer.getvalue())
            error_buffer.close()

    @staticmethod
    def style_id() -> str:
        return "vector"

    @staticmethod
    def geometry_kind() -> str:
        return "polygons"

    @staticmethod
    def supported_variants() -> list[str]:
        return ["default"]

    @staticmethod
    def default_variant() -> str:
        return "default"

    @staticmethod
    def supported_formats() -> set[ImageFormat]:
        return {ImageFormat.MVT, ImageFormat.GEOJSON}

    @classmethod
    def response_headers(cls, format: ImageFormat) -> dict[str, str]:
        if format is ImageFormat.MVT:
            return {"Content-Encoding": "gzip"}
        return {}

    @classmethod
    def describe_style(cls, variant: str) -> dict[str, str]:
        return {
            "id": f"{cls.style_id()}/{variant}",
            "title": "Vector",
            "description": (
                "Vector rendering — emits Mapbox Vector Tile (MVT) protobuf "
                "or GeoJSON feature collections of grid-cell polygons."
            ),
        }
