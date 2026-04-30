"""Vector tile renderer (MVT + GeoJSON).

Reuses the polygons-style geometry pipeline (cell rings + per-cell values
already in the output CRS), then either:

* MVT: delegates to :mod:`xpublish_tiles.render.mvt` for quantization +
  protobuf encoding, then gzips the result.
* GeoJSON: serializes the rings as an RFC 7946 FeatureCollection. Requires
  the request CRS to be geographic (CRS84 / EPSG:4326); rejected otherwise.

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
from xpublish_tiles.render.mvt import encode_mvt_layer, encode_mvt_tile, quantize_rings
from xpublish_tiles.types import (
    ImageFormat,
    NullRenderContext,
    PopulatedRenderContext,
    RenderContext,
)


def _rings_to_geojson_features(
    rings: np.ndarray, values: np.ndarray, *, var_name: str
) -> list[dict]:
    """Build GeoJSON Feature dicts from (N, M, 2) lon/lat rings.

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
                    rings_q = quantize_rings(rings, bbox=context.bbox, extent=extent)
                    layer_body = encode_mvt_layer(
                        name=str(var_name),
                        extent=extent,
                        rings_q=rings_q,
                        values=values,
                    )
                layers.append(layer_body)
            elif format is ImageFormat.GEOJSON:
                if context.crs is None or not context.crs.is_geographic:
                    raise ValueError(
                        "GeoJSON output requires a geographic CRS (CRS84 / EPSG:4326). "
                        "Pick a tile matrix set whose CRS is lon/lat — e.g. "
                        "WorldCRS84Quad — or request f=mvt instead."
                    )
                with log_duration(f"vector geojson {rings.shape[0]} polys", "▦", logger):
                    geojson_features.extend(
                        _rings_to_geojson_features(rings, values, var_name=str(var_name))
                    )

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
        # cells: one polygon per grid cell (current behavior).
        # contours / points (planned) will use the raster geometry pipeline.
        if variant in ("cells", "default"):
            return "polygons"
        raise ValueError(f"Unknown vector variant: {variant!r}")

    @staticmethod
    def supported_variants() -> list[str]:
        return ["cells"]

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
        }
        resolved = cls.default_variant() if variant == "default" else variant
        if resolved not in descriptions:
            raise ValueError(f"Unknown vector variant: {variant!r}")
        return {
            "id": f"{cls.style_id()}/{resolved}",
            "title": f"Vector — {resolved}",
            "description": descriptions[resolved],
        }
