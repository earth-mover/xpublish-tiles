"""Mapbox Vector Tile (MVT) encoder.

Self-contained encoder that turns ``(N, M, 2)`` polygon-ring arrays + per-
feature scalar values into raw MVT protobuf bytes. The float→int quantization
is vectorized numpy; the per-polygon command stream (MoveTo / LineTo /
ClosePath) is JIT-compiled with numba; the protobuf framing is written
directly with hand-rolled varint/length-delimited helpers.

Why hand-rolled instead of a library? Evaluated on this PR (#236):

* ``mapbox-vector-tile`` — pulls in shapely; pure-Python encoder is 10–100×
  slower than numba at our worst-case feature count.
* ``vt2pbf`` — pure-Python; ~70% slower than numba end-to-end at 262k polys
  (1.3 s → 2.3 s per tile).
* ``python-vtzero`` (and ``rio-tiler-mvt`` which wraps it) — Cython binding
  has no typed numeric ``add_property`` overload (would force float→str,
  losing precision and bloating tiles) and no batch numpy geometry path
  (~1.3M ``set_point`` Python→Cython calls per worst-case tile).

If a library emerges that supports typed numeric properties + a batch
ndarray geometry path, swap it in here behind ``encode_mvt_tile``.
"""

from __future__ import annotations

import struct

import numba
import numpy as np

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

# Field numbers from vector_tile.proto
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


def _emit_len_delim(field_num: int, body: bytes | bytearray) -> bytes:
    """Encode ``tag + len + body`` as a standalone bytes object."""
    out = bytearray()
    _write_tag(out, field_num, _WIRE_LEN)
    _write_varint(out, len(body))
    out.extend(body)
    return bytes(out)


def quantize_rings(rings: np.ndarray, *, bbox, extent: int) -> np.ndarray:
    """Float rings in output CRS → int32 rings in MVT tile-local space.

    MVT origin is top-left, Y increases downward.
    """
    sx = extent / (bbox.east - bbox.west)
    sy = extent / (bbox.north - bbox.south)
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
        prev_x = q[i, 0, 0]
        prev_y = q[i, 0, 1]
        cmd_buf[i, 0] = (_CMD_MOVETO & 0x7) | (1 << 3)
        cmd_buf[i, 1] = (prev_x << 1) ^ (prev_x >> 31)
        cmd_buf[i, 2] = (prev_y << 1) ^ (prev_y >> 31)
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


def encode_mvt_layer(
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
    cmd_stride = 2 * m + 1  # MoveTo(1+2) + LineTo(1+2*(m-2)) + Close(1)
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

    for i in range(n):
        if not valid[i]:
            continue
        feat = bytearray()
        _write_uint32(feat, _FEATURE_ID, i + 1)
        tags_payload = bytearray()
        _write_varint(tags_payload, 0)
        _write_varint(tags_payload, int(feature_value_idx[i]))
        _write_len_delim(feat, _FEATURE_TAGS, tags_payload)
        _write_uint32(feat, _FEATURE_TYPE, _GEOM_POLYGON)
        geom_payload = bytearray()
        L = int(cmd_len[i])
        for k in range(L):
            _write_varint(geom_payload, int(cmd_buf[i, k]))
        _write_len_delim(feat, _FEATURE_GEOMETRY, geom_payload)
        _write_len_delim(body, _LAYER_FEATURES, feat)

    return bytes(body)


def encode_mvt_tile(layers: list[bytes]) -> bytes:
    """Wrap one or more pre-encoded layer bodies as a Tile message."""
    out = bytearray()
    for layer_body in layers:
        out.extend(_emit_len_delim(_TILE_LAYERS, layer_body))
    return bytes(out)
