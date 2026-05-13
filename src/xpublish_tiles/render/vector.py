"""Vector tile renderer (MVT + GeoJSON).

Reuses the polygons-style geometry pipeline (cell rings + per-cell values
already in the output CRS) for ``cells`` and ``points``, and the raster
pipeline (regular gridded scalar field) for ``contours``:

* ``vector/cells`` — one polygon feature per grid cell, value as property.
* ``vector/points`` — one point feature per grid cell at the ring centroid,
  with all requested ``variables`` attached as typed properties (intended for
  vector-field cases like wind ``u``/``v``).
* ``vector/contours`` — filled contour bands as multi-ring polygons (outer +
  holes) emitted by :mod:`contourpy`, one feature per band per variable
  carrying ``value_lo``, ``value_hi``, ``value_mid`` properties. Clients can
  render as a fill layer, as a line layer (drawing polygon outlines = the
  isolines), or both on the same source.

Output goes through:

* MVT — :mod:`xpublish_tiles.render.mvt` for quantization + protobuf
  encoding, then gzipped.
* GeoJSON — RFC 7946 ``FeatureCollection``. Requires the request CRS to be
  geographic (CRS84 / EPSG:4326); rejected otherwise.

Width/height are not meaningful for vector tiles and are ignored; coarsening
targets ``config["mvt_extent"]`` (default 4096) via the ``vector`` branch in
``max_render_shape``.
"""

from __future__ import annotations

import gzip
import io
import json
from numbers import Number

import numpy as np
from contourpy import FillType, LineType, contour_generator

from xpublish_tiles.config import config
from xpublish_tiles.logger import get_context_logger, log_duration
from xpublish_tiles.render import Renderer, register_renderer, render_error_image
from xpublish_tiles.render.mvt import (
    encode_mvt_layer,
    encode_mvt_linestring_layer,
    encode_mvt_point_layer,
    encode_mvt_polygon_layer,
    encode_mvt_tile,
    quantize_rings,
)
from xpublish_tiles.types import (
    ImageFormat,
    NullRenderContext,
    PopulatedRenderContext,
    RenderContext,
)


def _rings_to_geojson_polygon_features(
    rings: np.ndarray, values: np.ndarray, *, var_name: str
) -> list[dict]:
    """Build GeoJSON Polygon Feature dicts from (N, M, 2) lon/lat rings.

    Rings must already be in WGS84 lon/lat — the caller is responsible for
    validating the output CRS (RFC 7946 GeoJSON requires CRS84).
    """
    n, m, _ = rings.shape
    features: list[dict] = []
    for i in range(n):
        v = values[i]
        if not np.isfinite(v):
            continue
        coords = rings[i].tolist()
        # Drop trailing repeats so the ring closes cleanly. We always keep
        # the first vertex repeated as the last to satisfy GeoJSON's closed-
        # ring requirement.
        last = m - 1
        while last > 1 and coords[last] == coords[last - 1]:
            last -= 1
        ring = coords[: last + 1]
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


def _centroids_from_rings(rings: np.ndarray) -> np.ndarray:
    """Approximate cell centroids from (N, M, 2) cell rings.

    Uses the mean of all-but-the-closing vertex. For non-pole cells padded
    from M=5 to M=6 (see the polar pole-split in the polygons pipeline) the
    closing-vertex pad is duplicated, giving a slight bias toward the first
    corner — visually negligible for points display.
    """
    return rings[:, :-1, :].mean(axis=1)


def _points_to_geojson_features(
    centroids: np.ndarray, properties: dict[str, np.ndarray]
) -> list[dict]:
    """Build GeoJSON Point Features. Drops a feature if any property value is non-finite."""
    n = centroids.shape[0]
    var_names = list(properties.keys())

    finite_mask = np.ones(n, dtype=bool)
    for vals in properties.values():
        finite_mask &= np.isfinite(vals)

    features: list[dict] = []
    for i in range(n):
        if not finite_mask[i]:
            continue
        features.append(
            {
                "type": "Feature",
                "id": i,
                "geometry": {
                    "type": "Point",
                    "coordinates": centroids[i].tolist(),
                },
                "properties": {name: float(properties[name][i]) for name in var_names},
            }
        )
    return features


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
        levels: tuple[float, ...] | None = None,
        smoothing: float | None = None,
    ):
        logger = context_logger if context_logger is not None else get_context_logger()

        if format not in self.supported_formats():
            raise ValueError(
                f"VectorTileRenderer cannot emit {format!r}; supported: "
                f"{sorted(f.value for f in self.supported_formats())}"
            )

        resolved = self.default_variant() if variant == "default" else variant
        if resolved == "cells":
            self._render_cells(
                contexts=contexts, buffer=buffer, format=format, logger=logger
            )
        elif resolved == "points":
            self._render_points(
                contexts=contexts, buffer=buffer, format=format, logger=logger
            )
        elif resolved == "contours":
            if levels is None or len(levels) < 2:
                raise ValueError(
                    "vector/contours requires the `levels` query parameter — "
                    "a comma-separated, strictly-increasing list of at least "
                    "two values defining the band boundaries (e.g. "
                    "'levels=0,5,10,15')."
                )
            self._render_contours(
                contexts=contexts,
                buffer=buffer,
                format=format,
                levels=levels,
                smoothing=smoothing,
                logger=logger,
            )
        else:
            raise ValueError(f"Unknown vector variant: {variant!r}")

    def _render_cells(self, *, contexts, buffer, format, logger):
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
                    f"vector cells quantize+encode {rings.shape[0]} polys", "▦", logger
                ):
                    rings_q = quantize_rings(rings, bbox=context.bbox, extent=extent)
                    layers.append(
                        encode_mvt_layer(
                            name=str(var_name),
                            extent=extent,
                            rings_q=rings_q,
                            values=values,
                        )
                    )
            elif format is ImageFormat.GEOJSON:
                _require_geographic_crs(context.crs)
                with log_duration(
                    f"vector cells geojson {rings.shape[0]} polys", "▦", logger
                ):
                    geojson_features.extend(
                        _rings_to_geojson_polygon_features(
                            rings, values, var_name=str(var_name)
                        )
                    )

        _write_output(buffer, format, layers=layers, geojson_features=geojson_features)

    def _render_points(self, *, contexts, buffer, format, logger):
        # Multi-variable: one MVT layer (or one GeoJSON FeatureCollection)
        # with every requested variable attached as a property on each point.
        # All variables must share the same grid (same `cell_rings`); we use
        # the first populated context as the geometry reference.
        ref_ctx = next(
            (
                c
                for c in contexts.values()
                if isinstance(c, PopulatedRenderContext)
                and c.cell_rings is not None
                and len(c.cell_rings) > 0
            ),
            None,
        )
        if ref_ctx is None:
            _write_output(buffer, format, layers=[], geojson_features=[])
            return
        assert ref_ctx.cell_rings is not None  # narrow for the type checker

        rings = np.asarray(ref_ctx.cell_rings)
        centroids = _centroids_from_rings(rings)
        properties: dict[str, np.ndarray] = {
            str(var_name): np.asarray(ctx.da.values).ravel()
            for var_name, ctx in contexts.items()
            if isinstance(ctx, PopulatedRenderContext) and ctx.da is not None
        }
        if not properties:
            _write_output(buffer, format, layers=[], geojson_features=[])
            return

        if format is ImageFormat.MVT:
            extent = config.get("mvt_extent")
            with log_duration(
                f"vector points encode {centroids.shape[0]} pts × {len(properties)} vars",
                "▦",
                logger,
            ):
                points_q = quantize_rings(centroids, bbox=ref_ctx.bbox, extent=extent)
                layer = encode_mvt_point_layer(
                    name="points",
                    extent=extent,
                    points_q=points_q,
                    properties=properties,
                )
            _write_output(buffer, format, layers=[layer], geojson_features=[])
        elif format is ImageFormat.GEOJSON:
            _require_geographic_crs(ref_ctx.crs)
            with log_duration(
                f"vector points geojson {centroids.shape[0]} pts × {len(properties)} vars",
                "▦",
                logger,
            ):
                features = _points_to_geojson_features(centroids, properties)
            _write_output(buffer, format, layers=[], geojson_features=features)

    def _render_contours(self, *, contexts, buffer, format, levels, smoothing, logger):
        # One MVT layer per variable; one combined GeoJSON FeatureCollection
        # tagged by ``variable`` (lets multi-var requests share a transport).
        layers: list[bytes] = []
        geojson_features: list[dict] = []

        for var_name, context in contexts.items():
            if isinstance(context, NullRenderContext):
                continue
            assert isinstance(context, PopulatedRenderContext)
            if context.da is None:
                continue

            grid = context.grid
            try:
                da = context.da.transpose(grid.Ydim, grid.Xdim)
            except (AttributeError, ValueError):
                # Non-2D grids fall through with no contours emitted.
                continue
            z = np.asarray(da.values, dtype=np.float64)
            if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 2:
                continue

            x_arr, y_arr = _coord_arrays_for_contour(da, grid, z.shape)
            if x_arr is None or y_arr is None:
                continue

            if smoothing is not None and smoothing > 0:
                with log_duration(
                    f"vector contours pre-blur sigma={smoothing} {z.shape}",
                    "▦",
                    logger,
                ):
                    z = _gaussian_smooth_with_nans(z, sigma=float(smoothing))

            with log_duration(
                f"vector contours generate {z.shape} {len(levels) - 1} bands",
                "▦",
                logger,
            ):
                gen = contour_generator(
                    x=x_arr,
                    y=y_arr,
                    z=z,
                    fill_type=FillType.OuterOffset,
                    line_type=LineType.ChunkCombinedOffset,
                )
                rings_list, ring_lengths, rings_per_poly, props_lists = (
                    _collect_stacked_contour_polygons(gen, levels)
                )
                line_traces = gen.multi_lines(list(levels))
                line_pts_list, line_lengths, line_values = _collect_contour_lines(
                    line_traces, levels
                )

            extent = config.get("mvt_extent")

            if rings_list:
                rings_flat = np.concatenate(rings_list, axis=0).astype(np.float64)
                ring_starts = np.concatenate(([0], np.cumsum(ring_lengths))).astype(
                    np.int32
                )
                poly_ring_starts = np.concatenate(
                    ([0], np.cumsum(rings_per_poly))
                ).astype(np.int32)
                properties = {
                    k: np.asarray(v, dtype=np.float64) for k, v in props_lists.items()
                }
                if format is ImageFormat.MVT:
                    with log_duration(
                        f"vector contours encode {len(rings_per_poly)} stacks",
                        "▦",
                        logger,
                    ):
                        rings_q = quantize_rings(
                            rings_flat, bbox=context.bbox, extent=extent
                        )
                        layers.append(
                            encode_mvt_polygon_layer(
                                name=str(var_name),
                                extent=extent,
                                rings_flat_q=rings_q,
                                ring_starts=ring_starts,
                                poly_ring_starts=poly_ring_starts,
                                properties=properties,
                            )
                        )
                else:  # GEOJSON
                    _require_geographic_crs(context.crs)
                    with log_duration(
                        f"vector contours geojson {len(rings_per_poly)} stacks",
                        "▦",
                        logger,
                    ):
                        geojson_features.extend(
                            _contour_polys_to_geojson_features(
                                rings_flat=rings_flat,
                                ring_starts=ring_starts,
                                poly_ring_starts=poly_ring_starts,
                                properties=properties,
                                var_name=str(var_name),
                            )
                        )

            if line_pts_list:
                lines_flat = np.concatenate(line_pts_list, axis=0).astype(np.float64)
                line_starts = np.concatenate(([0], np.cumsum(line_lengths))).astype(
                    np.int32
                )
                line_props = {"value": np.asarray(line_values, dtype=np.float64)}
                if format is ImageFormat.MVT:
                    with log_duration(
                        f"vector contours encode {len(line_lengths)} lines",
                        "▦",
                        logger,
                    ):
                        lines_q = quantize_rings(
                            lines_flat, bbox=context.bbox, extent=extent
                        )
                        layers.append(
                            encode_mvt_linestring_layer(
                                name=f"{var_name}_lines",
                                extent=extent,
                                points_flat_q=lines_q,
                                line_starts=line_starts,
                                properties=line_props,
                            )
                        )
                else:  # GEOJSON
                    _require_geographic_crs(context.crs)
                    with log_duration(
                        f"vector contours geojson {len(line_lengths)} lines",
                        "▦",
                        logger,
                    ):
                        geojson_features.extend(
                            _contour_lines_to_geojson_features(
                                lines_flat=lines_flat,
                                line_starts=line_starts,
                                values=line_props["value"],
                                var_name=str(var_name),
                            )
                        )

        _write_output(buffer, format, layers=layers, geojson_features=geojson_features)

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
    def geometry_kind(variant: str) -> str:
        # cells + points share the polygon pipeline (point = cell-ring centroid).
        # contours runs marching squares on the regular gridded scalar field
        # produced by the raster pipeline.
        if variant in ("cells", "points", "default"):
            return "polygons"
        if variant == "contours":
            return "raster"
        raise ValueError(f"Unknown vector variant: {variant!r}")

    @staticmethod
    def supported_variants() -> list[str]:
        return ["cells", "points", "contours"]

    @staticmethod
    def default_variant() -> str:
        return "cells"

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
        descriptions = {
            "cells": (
                "One MVT/GeoJSON polygon feature per grid cell, with the cell "
                "value attached as a typed property. Clients style on the value."
            ),
            "points": (
                "One MVT/GeoJSON point feature per grid cell at the cell centroid. "
                "All requested variables are attached as typed properties on each "
                "point — useful for vector fields (e.g. variables=u,v for wind "
                "arrows). Features missing any property value are dropped."
            ),
            "contours": (
                "Filled contour bands as multi-ring polygons (outer + holes), one "
                "feature per band per variable. Each carries 'value_lo', "
                "'value_hi', and 'value_mid' properties; clients render as a "
                "fill layer (color by band), as a line layer (polygon outlines = "
                "isolines), or both on the same source. Requires the `levels` "
                "query parameter."
            ),
        }
        resolved = cls.default_variant() if variant == "default" else variant
        if resolved not in descriptions:
            raise ValueError(f"Unknown vector variant: {variant!r}")
        return {
            "id": f"{cls.style_id()}/{resolved}",
            "title": f"Vector — {resolved}",
            "description": descriptions[resolved],
        }


def _gaussian_smooth_with_nans(z: np.ndarray, *, sigma: float) -> np.ndarray:
    """Gaussian-blur a 2D scalar field, treating NaNs as missing.

    A naive ``gaussian_filter`` propagates NaNs across the kernel footprint,
    which would erase valid data near any masked cell. We instead blur the
    NaN-zeroed field and the validity mask separately, then divide — the
    standard "normalized convolution" trick. NaN cells stay NaN in the
    output; valid cells get a re-weighted average of nearby valid cells.
    """
    from scipy.ndimage import gaussian_filter

    finite = np.isfinite(z)
    if finite.all():
        return gaussian_filter(z, sigma=sigma, mode="nearest")
    if not finite.any():
        return z
    z_filled = np.where(finite, z, 0.0)
    weight = finite.astype(np.float64)
    num = gaussian_filter(z_filled, sigma=sigma, mode="nearest")
    den = gaussian_filter(weight, sigma=sigma, mode="nearest")
    out = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    out[~finite] = np.nan
    return out


def _coord_arrays_for_contour(
    da, grid, z_shape: tuple[int, int]
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (x, y) arrays sized to match a (Ydim, Xdim)-transposed z.

    contourpy accepts either both 1D (``x.shape == (nx,), y.shape == (ny,)``)
    or both 2D (``x.shape == y.shape == z.shape``). Coords already aligned
    via ``da.transpose(grid.Ydim, grid.Xdim)`` so we just normalize shape.
    """
    try:
        x = np.asarray(da[grid.X].values, dtype=np.float64)
        y = np.asarray(da[grid.Y].values, dtype=np.float64)
    except (KeyError, AttributeError):
        return None, None
    ny, nx = z_shape
    if x.ndim == 1 and y.ndim == 1:
        if x.shape == (nx,) and y.shape == (ny,):
            return x, y
        return None, None
    if x.ndim == 2 and y.ndim == 2:
        if x.shape == z_shape and y.shape == z_shape:
            return x, y
        if x.T.shape == z_shape and y.T.shape == z_shape:
            return np.ascontiguousarray(x.T), np.ascontiguousarray(y.T)
    return None, None


def _collect_stacked_contour_polygons(
    gen, levels: tuple[float, ...]
) -> tuple[list[np.ndarray], list[int], list[int], dict[str, list[float]]]:
    """Build stacked "z >= L_i" polygons (one per level boundary).

    Why stacked instead of banded: ``multi_filled`` traces each band's
    outer and hole rings independently, so adjacent bands' shared boundary
    at level L gets two near-identical-but-not-pixel-identical traces.
    After MVT quantization those become 1–2 pixel seams between fills.
    Stacking instead emits one polygon per level (the simply-connected
    "above L" region, which may have holes for local minima at L). Each
    polygon's outer boundary at L is **one** trace, full stop. MapLibre
    renders polygons in source order, so we emit lowest-L first; the
    visible color in any pixel ends up being the topmost stack's
    ``[L_i, L_{i+1}]`` band — exactly the band containing z.

    For the topmost level we don't emit a stack: anything above it
    visually inherits the highest band's color (matches the palette-
    clamping the client already does for out-of-range values).
    """
    rings_list: list[np.ndarray] = []
    ring_lengths: list[int] = []
    rings_per_poly: list[int] = []
    props_lists: dict[str, list[float]] = {
        "value_lo": [],
        "value_hi": [],
        "value_mid": [],
    }

    for i in range(len(levels) - 1):
        lo = float(levels[i])
        hi = float(levels[i + 1])
        mid = 0.5 * (lo + hi)
        points_list, offsets_list = gen.filled(lo, np.inf)
        for pts, offs in zip(points_list, offsets_list, strict=True):
            pts = np.asarray(pts, dtype=np.float64)
            offs = np.asarray(offs, dtype=np.int64)
            n_rings = offs.shape[0] - 1
            kept_rings: list[np.ndarray] = []
            for j in range(n_rings):
                s, e = int(offs[j]), int(offs[j + 1])
                ring = pts[s:e]
                if (
                    ring.shape[0] >= 2
                    and ring[0, 0] == ring[-1, 0]
                    and ring[0, 1] == ring[-1, 1]
                ):
                    ring = ring[:-1]
                if ring.shape[0] < 3:
                    if j == 0:
                        kept_rings = []
                        break
                    continue
                kept_rings.append(np.ascontiguousarray(ring))
            if not kept_rings:
                continue
            rings_list.extend(kept_rings)
            ring_lengths.extend(r.shape[0] for r in kept_rings)
            rings_per_poly.append(len(kept_rings))
            props_lists["value_lo"].append(lo)
            props_lists["value_hi"].append(hi)
            props_lists["value_mid"].append(mid)

    return rings_list, ring_lengths, rings_per_poly, props_lists


def _contour_polys_to_geojson_features(
    *,
    rings_flat: np.ndarray,
    ring_starts: np.ndarray,
    poly_ring_starts: np.ndarray,
    properties: dict[str, np.ndarray],
    var_name: str,
) -> list[dict]:
    """Build GeoJSON Polygon Features (with holes) from the flat ring arrays."""
    n_polys = poly_ring_starts.shape[0] - 1
    features: list[dict] = []
    for p in range(n_polys):
        rs, re = int(poly_ring_starts[p]), int(poly_ring_starts[p + 1])
        rings_for_feat: list[list[list[float]]] = []
        for r in range(rs, re):
            ps, pe = int(ring_starts[r]), int(ring_starts[r + 1])
            ring = rings_flat[ps:pe].tolist()
            ring.append([ring[0][0], ring[0][1]])
            rings_for_feat.append(ring)
        if not rings_for_feat:
            continue
        features.append(
            {
                "type": "Feature",
                "id": p,
                "geometry": {"type": "Polygon", "coordinates": rings_for_feat},
                "properties": {
                    "kind": "fill",
                    "variable": var_name,
                    "value_lo": float(properties["value_lo"][p]),
                    "value_hi": float(properties["value_hi"][p]),
                    "value_mid": float(properties["value_mid"][p]),
                },
            }
        )
    return features


def _collect_contour_lines(
    line_traces, levels: tuple[float, ...]
) -> tuple[list[np.ndarray], list[int], list[float]]:
    """Flatten contourpy ``multi_lines`` output (one entry per level) into
    flat-array form for the LineString encoder.

    contourpy with ``LineType.ChunkCombinedOffset`` returns, per level, a
    tuple ``(pts_chunks, offset_chunks)``. Each ``pts_chunks[ci]`` is
    either ``None`` (empty chunk) or an ``(N_chunk_pts, 2)`` array; the
    matching ``offset_chunks[ci]`` slices that array into individual line
    segments. We concat all segments across all levels into one flat list
    and remember each segment's level value.
    """
    pts_list: list[np.ndarray] = []
    line_lengths: list[int] = []
    values: list[float] = []
    for lvl, (pts_chunks, offset_chunks) in zip(levels, line_traces, strict=True):
        for chunk_pts, chunk_offs in zip(pts_chunks, offset_chunks, strict=True):
            if chunk_pts is None or chunk_offs is None:
                continue
            chunk_pts = np.asarray(chunk_pts, dtype=np.float64)
            chunk_offs = np.asarray(chunk_offs, dtype=np.int64)
            for i in range(chunk_offs.shape[0] - 1):
                s, e = int(chunk_offs[i]), int(chunk_offs[i + 1])
                if e - s < 2:
                    continue
                pts_list.append(np.ascontiguousarray(chunk_pts[s:e]))
                line_lengths.append(e - s)
                values.append(float(lvl))
    return pts_list, line_lengths, values


def _contour_lines_to_geojson_features(
    *,
    lines_flat: np.ndarray,
    line_starts: np.ndarray,
    values: np.ndarray,
    var_name: str,
) -> list[dict]:
    """Build GeoJSON LineString Features for labeled isolines."""
    n_lines = line_starts.shape[0] - 1
    features: list[dict] = []
    for i in range(n_lines):
        s, e = int(line_starts[i]), int(line_starts[i + 1])
        coords = lines_flat[s:e].tolist()
        if len(coords) < 2:
            continue
        features.append(
            {
                "type": "Feature",
                "id": i,
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "kind": "line",
                    "variable": var_name,
                    "value": float(values[i]),
                },
            }
        )
    return features


def _require_geographic_crs(crs) -> None:
    if crs is None or not crs.is_geographic:
        raise ValueError(
            "GeoJSON output requires a geographic CRS (CRS84 / EPSG:4326). "
            "Pick a tile matrix set whose CRS is lon/lat — e.g. "
            "WorldCRS84Quad — or request f=mvt instead."
        )


def _write_output(
    buffer: io.BytesIO,
    format: ImageFormat,
    *,
    layers: list[bytes],
    geojson_features: list[dict],
) -> None:
    if format is ImageFormat.MVT:
        tile = encode_mvt_tile(layers)
        buffer.write(gzip.compress(tile, compresslevel=1))
    else:
        payload = json.dumps(
            {"type": "FeatureCollection", "features": geojson_features},
            allow_nan=False,
            separators=(",", ":"),
        )
        buffer.write(payload.encode("utf-8"))
