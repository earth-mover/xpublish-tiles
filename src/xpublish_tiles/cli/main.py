"""Simple CLI for playing with xpublish-tiles, with a generated sample dataset"""

import argparse
from typing import cast

import cf_xarray  # noqa: F401
import numpy as np
import xpublish
from fastapi.middleware.cors import CORSMiddleware

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


def get_dataset_for_name(name: str, branch: str = "main", group: str = "") -> xr.Dataset:
    if name == "global":
        return create_global_dataset()
    elif name == "air":
        return xr.tutorial.open_dataset("air_temperature")

    try:
        from arraylake import Client

        import icechunk

        client = Client()
        repo = cast(icechunk.Repository, client.get_repo(name))
        session = repo.readonly_session(branch=branch)
        return xr.open_zarr(
            session.store,
            group=group if len(group) else None,
            zarr_format=3,
            consolidated=False,
        )
    except ImportError as ie:
        raise ImportError(
            f"Arraylake is not installed, no dataset available named {name}"
        ) from ie
    except Exception as e:
        raise ValueError(
            f"Error occurred while getting dataset from Arraylake: {e}"
        ) from e


def main():
    parser = argparse.ArgumentParser(
        description="Simple CLI for playing with xpublish-tiles"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to serve on (default: 8080)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="global",
        help="Dataset to serve (default: global). If an arraylake dataset is specified, the arraylake-org and arraylake-repo must be provided, along with an optional branch and group",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Branch to use for Arraylake (default: main). ",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="",
        help="Group to use for Arraylake (default: '').",
    )
    args = parser.parse_args()

    ds = get_dataset_for_name(args.dataset, args.branch, args.group)

    rest = xpublish.SingleDatasetRest(
        ds,
        plugins={"tiles": TilesPlugin(), "wms": WMSPlugin()},
    )
    rest.app.add_middleware(CORSMiddleware, allow_origins=["*"])
    rest.serve(host="0.0.0.0", port=8080)
