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
            "max_renderable_size": 400_000_000,  # 10,000 * 10,000 pixels - this takes the pipeline ~ 1s
            "max_pixel_factor": 4,  # coarsen down to this many input grid cells per output pixel
            "async_load": True,
        }
    ],
    paths=[],
)
