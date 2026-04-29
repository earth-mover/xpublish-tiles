"""Configuration management for xpublish-tiles using donfig."""

from __future__ import annotations

import donfig

config = donfig.Config(
    "xpublish_tiles",
    defaults=[
        {
            "num_threads": 8,
            "transform_chunk_size": 1024,
            "detect_approx_rectilinear": True,
            "rectilinear_check_min_size": 512,
            # Ideally, we'd want to pad with 1.
            # However, due to floating point roundoff when datashader *infers* the cell edges,
            # we might end up with the last grid cell of a global dataset ending very slightly before
            # the bounds of the Canvas. This then results in transparent pixels
            "default_pad": 2,
            # in bytes; ~10,000 * 10,000 float64 pixels; takes the pipeline ~ 1s
            "max_renderable_size": 1024**3,
            "max_pixel_factor": 3,  # coarsen down to this many input grid cells per output pixel
            "max_num_geometries": 1_500_000,  # coarsend down to this many geometries.
            "mvt_extent": 4096,  # MVT tile-local integer quantization grid (Mapbox/MapLibre default)
            # Max number of polygon features per axis for vector tiles. The
            # MVT extent is the *quantization* grid (subpixel precision), not a
            # feature-count target — emitting one polygon per quantization cell
            # produces unreadable output at low zoom. 512 ≈ 2× a standard 256px
            # display tile, which is enough detail without flooding the client.
            "vector_max_features_per_side": 512,
            "async_load": True,
            "async_load_timeout_per_tile": 20,  # seconds; None to disable
            "num_concurrent_data_loads": 4,  # max concurrent tile data loads; None to disable
            "grid_cache_max_size": 16,  # maximum number of grid systems to cache
        }
    ],
    paths=[],
)
