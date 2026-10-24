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
            # vertices summed over the features of one Features tile (provisional)
            "max_renderable_vertices": 5_000_000,
            "max_pixel_factor": 3,  # coarsen down to this many input grid cells per output pixel
            "async_load": True,
            "async_load_timeout_per_tile": 20,  # seconds; None to disable
            "num_concurrent_data_loads": 4,  # max concurrent tile data loads; None to disable
            "grid_cache_max_size": 16,  # maximum number of grid systems to cache
            # one of "nearest", "coarser", "finer"; see OverviewSelectionStrategy
            "overview_selection_strategy": "nearest",
            # cells; max seam drift |360/|dx| - n| accepted for a periodic RasterAffine x axis
            "rasterix_period_atol": 0.05,
            # degrees; minzoom ignores geographic-grid tiles poleward of this (Web
            # Mercator limit, so WebMercatorQuad is unaffected)
            "minzoom_polar_cutoff": 85.0511,
            # Callable[[str], None] that gets each request's summary block; None writes
            # to stdout. It runs in the request's context, so it can read contextvars.
            "log_summary_writer": None,
            # /tiles/ extents: an axis with more constant-step runs gets no repeating
            # intervals (OGC API EDR ``R<count>/<start>/<step>`` values)
            "max_repeating_intervals": 16,
            # /tiles/ extents: a jittered time axis gets its median step as a nominal
            # resolution if this fraction of steps is within ``nominal_step_rtol`` of
            # a whole multiple of it
            "nominal_step_min_fraction": 0.99,
            "nominal_step_rtol": 0.01,
        }
    ],
    paths=[],
)
