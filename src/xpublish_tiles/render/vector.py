"""Vector tile renderer (MVT + GeoJSON).

Reuses the polygons-style geometry pipeline (cell rings + per-cell values
already in the output CRS) to emit two variants today:

* ``vector/cells`` — one polygon feature per grid cell, value as property.
* ``vector/points`` — one point feature per grid cell at the ring centroid,
  with all requested ``variables`` attached as typed properties (intended for
  vector-field cases like wind ``u``/``v``).

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

from xpublish_tiles.config import config
from xpublish_tiles.logger import get_context_logger, log_duration
from xpublish_tiles.render import Renderer, register_renderer, render_error_image
from xpublish_tiles.render.mvt import (
    encode_mvt_layer,
    encode_mvt_point_layer,
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
        # A future `contours` variant will return "raster".
        if variant in ("cells", "points", "default"):
            return "polygons"
        raise ValueError(f"Unknown vector variant: {variant!r}")

    @staticmethod
    def supported_variants() -> list[str]:
        return ["cells", "points"]

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
        }
        resolved = cls.default_variant() if variant == "default" else variant
        if resolved not in descriptions:
            raise ValueError(f"Unknown vector variant: {variant!r}")
        return {
            "id": f"{cls.style_id()}/{resolved}",
            "title": f"Vector — {resolved}",
            "description": descriptions[resolved],
        }


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
