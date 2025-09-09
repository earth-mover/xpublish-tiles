"""Configuration management for xpublish-tiles using donfig."""

from __future__ import annotations

import donfig

config = donfig.Config(
    "xpublish_tiles",
    defaults=[
        {
            "num_threads": 8,
            "rectilinear_check_subsample_step": 2,
            "transform_chunk_size": 1024,
            "detect_approx_rectilinear": True,
            # Ideally, we'd want to pad with 1.
            # However, due to floating point roundoff when datashader *infers* the cell edges,
            # we might end up with the last grid cell of a global dataset ending very slightly before
            # the bounds of the Canvas. This then results in transparent pixels
            "default_pad": 2,
        }
    ],
    paths=[],
    env_var="XPUBLISH_TILES_CONFIG_PATH",
    env={
        "xpublish_tiles": {
            "num_threads": "XPUBLISH_TILES_NUM_THREADS",
            "rectilinear_check_subsample_step": "XPUBLISH_TILES_RECTILINEAR_CHECK_SUBSAMPLE_STEP",
            "transform_chunk_size": "XPUBLISH_TILES_TRANSFORM_CHUNK_SIZE",
            "detect_approx_rectilinear": "XPUBLISH_TILES_DETECT_APPROX_RECTILINEAR",
            "default_pad": "XPUBLISH_TILES_DEFAULT_PAD",
        }
    },
)
