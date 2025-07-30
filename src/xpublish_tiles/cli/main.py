"""Simple CLI for playing with xpublish-tiles, with a generated sample dataset"""

import argparse

import cf_xarray  # noqa: F401
import numpy as np
import xpublish

import xarray as xr
from xpublish_tiles.datasets import Dim, uniform_grid
from xpublish_tiles.xpublish.tiles.plugin import TilesPlugin
from xpublish_tiles.xpublish.wms.plugin import WMSPlugin


def create_global_dataset() -> xr.Dataset:
    dims = []

    nlat, nlon = 720, 1441
    lats = np.linspace(-90, 90, nlat)
    lons = np.linspace(-180, 180, nlon)

    dims = [
        Dim(
            name="latitude",
            size=nlat,
            chunk_size=nlat,
            data=lats,
            attrs={"standard_name": "latitude"},
        ),
        Dim(
            name="longitude",
            size=nlon,
            chunk_size=nlon,
            data=lons,
            attrs={"standard_name": "longitude"},
        ),
    ]
    return uniform_grid(dims=tuple(dims), dtype=np.float32, attrs={})


def get_dataset_for_name(name: str) -> xr.Dataset:
    if name == "global":
        return create_global_dataset()
    elif name == "air":
        ds = xr.tutorial.open_dataset("air_temperature")
        ds = ds.isel(time=0)
        return ds
    raise ValueError(f"Unknown dataset name: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Simple CLI for playing with xpublish-tiles"
    )
    parser.add_argument(
        "--dataset",
        choices=["global", "air"],
        default="global",
        help="Dataset to serve (default: global)",
    )
    args = parser.parse_args()

    ds = get_dataset_for_name(args.dataset)

    rest = xpublish.SingleDatasetRest(
        ds, plugins={"tiles": TilesPlugin(), "wms": WMSPlugin()}
    )
    rest.serve(host="0.0.0.0", port=8080)
