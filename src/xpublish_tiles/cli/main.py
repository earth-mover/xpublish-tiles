"""Simple CLI for playing with xpublish-tiles, with a generated sample dataset"""

import argparse

import cf_xarray  # noqa: F401
import xpublish

import xarray as xr
from xpublish_tiles.datasets import create_global_dataset
from xpublish_tiles.xpublish.tiles.plugin import TilesPlugin
from xpublish_tiles.xpublish.wms.plugin import WMSPlugin


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
