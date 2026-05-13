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
_GEOM_POINT = 1
_GEOM_LINESTRING = 2
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


@numba.njit(cache=True, boundscheck=False)
def _encode_multiring_polygons(
    rings_flat: np.ndarray,
    ring_starts: np.ndarray,
    poly_ring_starts: np.ndarray,
    cmd_buf: np.ndarray,
    poly_cmd_starts: np.ndarray,
    cmd_len: np.ndarray,
    valid: np.ndarray,
):
    """Encode multi-ring polygons (outer + holes) as MVT command streams.

    ``rings_flat`` is ``(total_pts, 2)`` int32 quantized vertices with NO
    closing-vertex duplicates. Ring ``r`` spans ``rings_flat[ring_starts[r]:
    ring_starts[r+1]]``. Polygon ``p`` owns rings ``[poly_ring_starts[p],
    poly_ring_starts[p+1])``; the first ring of each polygon is the outer
    boundary, the rest are holes.

    Per-ring winding is corrected against the MVT spec (outer = positive
    shoelace area in tile coords; inner = negative): rings are emitted in
    reverse order if their signed area has the wrong sign. Degenerate rings
    (<3 distinct vertices after dedup) are dropped; if the outer ring is
    degenerate the whole polygon is invalidated.

    The MVT cursor persists across rings within a polygon. Per spec,
    ClosePath returns the cursor to the start of the most recent MoveTo, so
    each subsequent ring's MoveTo delta is computed against that anchor.
    """
    n_polys = poly_ring_starts.shape[0] - 1
    for p in range(n_polys):
        ring_lo = poly_ring_starts[p]
        ring_hi = poly_ring_starts[p + 1]
        cmd_off = poly_cmd_starts[p]
        write_idx = 0
        cursor_x = np.int32(0)
        cursor_y = np.int32(0)
        outer_ok = False
        outer_drop = False

        for r in range(ring_lo, ring_hi):
            pt_lo = ring_starts[r]
            pt_hi = ring_starts[r + 1]
            m = pt_hi - pt_lo
            outer = r == ring_lo
            if m < 3:
                if outer:
                    outer_drop = True
                    break
                continue

            # Shoelace signed area in tile coords (Y flipped).
            area2 = np.int64(0)
            for k in range(m):
                a = pt_lo + k
                b = pt_lo + ((k + 1) % m)
                area2 += np.int64(rings_flat[a, 0]) * np.int64(rings_flat[b, 1])
                area2 -= np.int64(rings_flat[b, 0]) * np.int64(rings_flat[a, 1])

            need_reverse = (outer and area2 < 0) or ((not outer) and area2 > 0)
            if need_reverse:
                start_k = m - 1
                step = -1
            else:
                start_k = 0
                step = 1

            x0 = rings_flat[pt_lo + start_k, 0]
            y0 = rings_flat[pt_lo + start_k, 1]
            dx = x0 - cursor_x
            dy = y0 - cursor_y
            ring_anchor = write_idx
            cmd_buf[cmd_off + write_idx] = (_CMD_MOVETO & 0x7) | (1 << 3)
            write_idx += 1
            cmd_buf[cmd_off + write_idx] = (dx << 1) ^ (dx >> 31)
            write_idx += 1
            cmd_buf[cmd_off + write_idx] = (dy << 1) ^ (dy >> 31)
            write_idx += 1

            lineto_header = write_idx
            cmd_buf[cmd_off + write_idx] = 0
            write_idx += 1

            n_lineto = 0
            prev_x = x0
            prev_y = y0
            for j in range(1, m):
                k = start_k + j * step
                cur_x = rings_flat[pt_lo + k, 0]
                cur_y = rings_flat[pt_lo + k, 1]
                if cur_x == prev_x and cur_y == prev_y:
                    continue
                dx = cur_x - prev_x
                dy = cur_y - prev_y
                cmd_buf[cmd_off + write_idx] = (dx << 1) ^ (dx >> 31)
                write_idx += 1
                cmd_buf[cmd_off + write_idx] = (dy << 1) ^ (dy >> 31)
                write_idx += 1
                n_lineto += 1
                prev_x = cur_x
                prev_y = cur_y

            if n_lineto < 2:
                if outer:
                    outer_drop = True
                    break
                # Drop hole: rewind to before this ring's MoveTo.
                write_idx = ring_anchor
                continue

            cmd_buf[cmd_off + lineto_header] = (_CMD_LINETO & 0x7) | (n_lineto << 3)
            cmd_buf[cmd_off + write_idx] = (_CMD_CLOSEPATH & 0x7) | (1 << 3)
            write_idx += 1
            cursor_x = x0
            cursor_y = y0
            if outer:
                outer_ok = True

        if outer_drop or not outer_ok:
            cmd_len[p] = 0
            valid[p] = 0
        else:
            cmd_len[p] = write_idx
            valid[p] = 1


def encode_mvt_polygon_layer(
    *,
    name: str,
    extent: int,
    rings_flat_q: np.ndarray,
    ring_starts: np.ndarray,
    poly_ring_starts: np.ndarray,
    properties: dict[str, np.ndarray],
) -> bytes:
    """Build an MVT layer of multi-ring Polygon features (outer + holes).

    Inputs are pre-quantized via :func:`quantize_rings`. Each polygon may
    carry one or more rings — the first is the outer boundary, the rest are
    holes. Properties is a per-variable ``(n_polys,)`` float array; a
    polygon is dropped if **any** of its property values is non-finite or
    if its outer ring degenerates after winding/dedup.
    """
    n_polys = poly_ring_starts.shape[0] - 1
    if n_polys == 0:
        body = bytearray()
        _write_uint32(body, _LAYER_VERSION, 2)
        _write_string(body, _LAYER_NAME, name)
        _write_uint32(body, _LAYER_EXTENT, extent)
        return bytes(body)

    poly_n_pts = (
        ring_starts[poly_ring_starts[1:]] - ring_starts[poly_ring_starts[:-1]]
    ).astype(np.int64)
    poly_n_rings = np.diff(poly_ring_starts).astype(np.int64)
    poly_cmd_upper = 2 * poly_n_pts + 3 * poly_n_rings
    poly_cmd_starts = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(poly_cmd_upper)]
    ).astype(np.int32)
    cmd_buf = np.zeros(int(poly_cmd_starts[-1]), dtype=np.uint32)
    cmd_len = np.zeros(n_polys, dtype=np.int32)
    valid = np.zeros(n_polys, dtype=np.uint8)

    with NUMBA_THREADING_LOCK:
        _encode_multiring_polygons(
            rings_flat_q,
            ring_starts.astype(np.int32),
            poly_ring_starts.astype(np.int32),
            cmd_buf,
            poly_cmd_starts,
            cmd_len,
            valid,
        )

    var_names = list(properties.keys())
    finite_mask = np.ones(n_polys, dtype=bool)
    for vals in properties.values():
        finite_mask &= np.isfinite(vals)
    valid &= finite_mask.view(np.uint8)

    body = bytearray()
    _write_uint32(body, _LAYER_VERSION, 2)
    _write_string(body, _LAYER_NAME, name)
    _write_uint32(body, _LAYER_EXTENT, extent)

    value_buf = bytearray()
    feature_tag_payloads: list[bytes | None] = [None] * n_polys
    next_value_idx = 0
    for i in range(n_polys):
        if not valid[i]:
            continue
        tags = bytearray()
        for k_idx, var_name in enumerate(var_names):
            value_body = bytearray()
            _write_double(value_body, _VALUE_DOUBLE, float(properties[var_name][i]))
            value_buf.extend(_emit_len_delim(_LAYER_VALUES, value_body))
            _write_varint(tags, k_idx)
            _write_varint(tags, next_value_idx)
            next_value_idx += 1
        feature_tag_payloads[i] = bytes(tags)

    for var_name in var_names:
        _write_string(body, _LAYER_KEYS, var_name)
    body.extend(value_buf)

    for i in range(n_polys):
        tags_payload = feature_tag_payloads[i]
        if tags_payload is None:
            continue
        feat = bytearray()
        _write_uint32(feat, _FEATURE_ID, i + 1)
        _write_len_delim(feat, _FEATURE_TAGS, tags_payload)
        _write_uint32(feat, _FEATURE_TYPE, _GEOM_POLYGON)
        geom_payload = bytearray()
        L = int(cmd_len[i])
        cmd_off = int(poly_cmd_starts[i])
        for k in range(L):
            _write_varint(geom_payload, int(cmd_buf[cmd_off + k]))
        _write_len_delim(feat, _FEATURE_GEOMETRY, geom_payload)
        _write_len_delim(body, _LAYER_FEATURES, feat)

    return bytes(body)


@numba.njit(cache=True, boundscheck=False)
def _encode_linestrings(
    points_flat: np.ndarray,
    line_starts: np.ndarray,
    cmd_buf: np.ndarray,
    line_cmd_starts: np.ndarray,
    cmd_len: np.ndarray,
    valid: np.ndarray,
):
    """Encode a flat run of LineStrings as MVT command streams.

    Per line: MoveTo(absolute first vertex) + LineTo(remaining deltas), no
    ClosePath. The cursor resets to (0, 0) per line because each line is
    its own feature here. Lines with fewer than 2 distinct vertices after
    dedup are marked invalid.
    """
    n_lines = line_starts.shape[0] - 1
    for i in range(n_lines):
        lo = line_starts[i]
        hi = line_starts[i + 1]
        m = hi - lo
        cmd_off = line_cmd_starts[i]
        if m < 2:
            cmd_len[i] = 0
            valid[i] = 0
            continue

        x0 = points_flat[lo, 0]
        y0 = points_flat[lo, 1]
        cmd_buf[cmd_off + 0] = (_CMD_MOVETO & 0x7) | (1 << 3)
        cmd_buf[cmd_off + 1] = (x0 << 1) ^ (x0 >> 31)
        cmd_buf[cmd_off + 2] = (y0 << 1) ^ (y0 >> 31)
        write_idx = 4
        n_lineto = 0
        prev_x = x0
        prev_y = y0
        for k in range(1, m):
            cur_x = points_flat[lo + k, 0]
            cur_y = points_flat[lo + k, 1]
            if cur_x == prev_x and cur_y == prev_y:
                continue
            dx = cur_x - prev_x
            dy = cur_y - prev_y
            cmd_buf[cmd_off + write_idx] = (dx << 1) ^ (dx >> 31)
            write_idx += 1
            cmd_buf[cmd_off + write_idx] = (dy << 1) ^ (dy >> 31)
            write_idx += 1
            n_lineto += 1
            prev_x = cur_x
            prev_y = cur_y

        if n_lineto < 1:
            cmd_len[i] = 0
            valid[i] = 0
            continue
        cmd_buf[cmd_off + 3] = (_CMD_LINETO & 0x7) | (n_lineto << 3)
        cmd_len[i] = write_idx
        valid[i] = 1


def encode_mvt_linestring_layer(
    *,
    name: str,
    extent: int,
    points_flat_q: np.ndarray,
    line_starts: np.ndarray,
    properties: dict[str, np.ndarray],
) -> bytes:
    """Build an MVT Layer of LineString features.

    ``points_flat_q`` is ``(total_pts, 2)`` int32 already-quantized
    coordinates; line ``i`` spans ``points_flat_q[line_starts[i]:line_starts[i+1]]``.
    Each line carries one entry from each ``properties`` array (typed
    Value.double_value). Lines with non-finite property values or fewer
    than 2 distinct vertices after dedup are dropped.
    """
    n_lines = line_starts.shape[0] - 1
    if n_lines == 0:
        body = bytearray()
        _write_uint32(body, _LAYER_VERSION, 2)
        _write_string(body, _LAYER_NAME, name)
        _write_uint32(body, _LAYER_EXTENT, extent)
        return bytes(body)

    line_n_pts = np.diff(line_starts).astype(np.int64)
    # Per line upper bound: MoveTo(3) + LineTo header(1) + LineTo params(2*(M-1))
    # = 2*M + 2. Safe upper bound, plenty of margin for dedup'd commands.
    line_cmd_upper = 2 * line_n_pts + 2
    line_cmd_starts = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(line_cmd_upper)]
    ).astype(np.int32)
    cmd_buf = np.zeros(int(line_cmd_starts[-1]), dtype=np.uint32)
    cmd_len = np.zeros(n_lines, dtype=np.int32)
    valid = np.zeros(n_lines, dtype=np.uint8)

    with NUMBA_THREADING_LOCK:
        _encode_linestrings(
            points_flat_q,
            line_starts.astype(np.int32),
            cmd_buf,
            line_cmd_starts,
            cmd_len,
            valid,
        )

    var_names = list(properties.keys())
    finite_mask = np.ones(n_lines, dtype=bool)
    for vals in properties.values():
        finite_mask &= np.isfinite(vals)
    valid &= finite_mask.view(np.uint8)

    body = bytearray()
    _write_uint32(body, _LAYER_VERSION, 2)
    _write_string(body, _LAYER_NAME, name)
    _write_uint32(body, _LAYER_EXTENT, extent)

    value_buf = bytearray()
    feature_tag_payloads: list[bytes | None] = [None] * n_lines
    next_value_idx = 0
    for i in range(n_lines):
        if not valid[i]:
            continue
        tags = bytearray()
        for k_idx, var_name in enumerate(var_names):
            value_body = bytearray()
            _write_double(value_body, _VALUE_DOUBLE, float(properties[var_name][i]))
            value_buf.extend(_emit_len_delim(_LAYER_VALUES, value_body))
            _write_varint(tags, k_idx)
            _write_varint(tags, next_value_idx)
            next_value_idx += 1
        feature_tag_payloads[i] = bytes(tags)

    for var_name in var_names:
        _write_string(body, _LAYER_KEYS, var_name)
    body.extend(value_buf)

    for i in range(n_lines):
        tags_payload = feature_tag_payloads[i]
        if tags_payload is None:
            continue
        feat = bytearray()
        _write_uint32(feat, _FEATURE_ID, i + 1)
        _write_len_delim(feat, _FEATURE_TAGS, tags_payload)
        _write_uint32(feat, _FEATURE_TYPE, _GEOM_LINESTRING)
        geom_payload = bytearray()
        L = int(cmd_len[i])
        cmd_off = int(line_cmd_starts[i])
        for k in range(L):
            _write_varint(geom_payload, int(cmd_buf[cmd_off + k]))
        _write_len_delim(feat, _FEATURE_GEOMETRY, geom_payload)
        _write_len_delim(body, _LAYER_FEATURES, feat)

    return bytes(body)


@numba.njit(cache=True, boundscheck=False)
def _encode_points_inplace(q: np.ndarray, cmd_buf: np.ndarray):
    """Encode (N, 2) int32 points as one-vertex MoveTo commands.

    Each row of ``cmd_buf`` becomes ``[moveto_cmd, zigzag_x, zigzag_y]``.
    """
    n = q.shape[0]
    for i in range(n):
        x = q[i, 0]
        y = q[i, 1]
        cmd_buf[i, 0] = (_CMD_MOVETO & 0x7) | (1 << 3)
        cmd_buf[i, 1] = (x << 1) ^ (x >> 31)
        cmd_buf[i, 2] = (y << 1) ^ (y >> 31)


def encode_mvt_point_layer(
    *,
    name: str,
    extent: int,
    points_q: np.ndarray,
    properties: dict[str, np.ndarray],
) -> bytes:
    """Build a single MVT Layer of Point features carrying typed properties.

    ``points_q`` is ``(N, 2)`` int32 quantized point coordinates in tile-local
    space. ``properties`` maps each variable name to an ``(N,)`` float array.
    A feature is dropped entirely if **any** of its property values is non-
    finite (NaN/Inf) — useful for vector-field cases like ``u``/``v`` wind
    components where a partially-missing feature isn't actionable.
    """
    n = points_q.shape[0]
    var_names = list(properties.keys())

    finite_mask = np.ones(n, dtype=bool)
    for vals in properties.values():
        finite_mask &= np.isfinite(vals)

    cmd_buf = np.zeros((n, 3), dtype=np.uint32)
    with NUMBA_THREADING_LOCK:
        _encode_points_inplace(points_q, cmd_buf)

    # Build values table + per-feature tag-payloads in one pass.
    value_buf = bytearray()
    feature_tag_payloads: list[bytes | None] = [None] * n
    next_value_idx = 0
    for i in range(n):
        if not finite_mask[i]:
            continue
        tags = bytearray()
        for k_idx, var_name in enumerate(var_names):
            value_body = bytearray()
            _write_double(value_body, _VALUE_DOUBLE, float(properties[var_name][i]))
            value_buf.extend(_emit_len_delim(_LAYER_VALUES, value_body))
            _write_varint(tags, k_idx)
            _write_varint(tags, next_value_idx)
            next_value_idx += 1
        feature_tag_payloads[i] = bytes(tags)

    body = bytearray()
    _write_uint32(body, _LAYER_VERSION, 2)
    _write_string(body, _LAYER_NAME, name)
    _write_uint32(body, _LAYER_EXTENT, extent)
    for var_name in var_names:
        _write_string(body, _LAYER_KEYS, var_name)
    body.extend(value_buf)

    for i in range(n):
        tags_payload = feature_tag_payloads[i]
        if tags_payload is None:
            continue
        feat = bytearray()
        _write_uint32(feat, _FEATURE_ID, i + 1)
        _write_len_delim(feat, _FEATURE_TAGS, tags_payload)
        _write_uint32(feat, _FEATURE_TYPE, _GEOM_POINT)
        geom_payload = bytearray()
        for k in range(3):
            _write_varint(geom_payload, int(cmd_buf[i, k]))
        _write_len_delim(feat, _FEATURE_GEOMETRY, geom_payload)
        _write_len_delim(body, _LAYER_FEATURES, feat)

    return bytes(body)
