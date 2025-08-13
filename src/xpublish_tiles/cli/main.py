"""Simple CLI for playing with xpublish-tiles, with a generated sample dataset"""

import argparse
from typing import cast

import cf_xarray  # noqa: F401
import xpublish
from fastapi.middleware.cors import CORSMiddleware

import xarray as xr
from xpublish_tiles.datasets import EU3035_HIRES, HRRR, PARA_HIRES, create_global_dataset
from xpublish_tiles.xpublish.tiles.plugin import TilesPlugin
from xpublish_tiles.xpublish.wms.plugin import WMSPlugin


def get_dataset_for_name(
    name: str, branch: str = "main", group: str = "", icechunk_cache: bool = False
) -> xr.Dataset:
    if name == "global":
        ds = create_global_dataset().assign_attrs(_xpublish_id=name)
    elif name == "air":
        ds = xr.tutorial.open_dataset("air_temperature").assign_attrs(_xpublish_id=name)
    elif name == "hrrr":
        ds = HRRR.create().assign_attrs(_xpublish_id=name)
    elif name == "para":
        ds = PARA_HIRES.create().assign_attrs(_xpublish_id=name)
    elif name == "eu3035":
        ds = EU3035_HIRES.create().assign_attrs(_xpublish_id=name)
    else:
        # Arraylake path
        try:
            from arraylake import Client

            import icechunk

            config: icechunk.RepositoryConfig | None = None
            if icechunk_cache:
                config = icechunk.RepositoryConfig(
                    caching=icechunk.CachingConfig(
                        num_bytes_chunks=1073741824,
                        num_chunk_refs=1073741824,
                        num_bytes_attributes=100_000_000,
                    )
                )

            client = Client()
            repo = cast(icechunk.Repository, client.get_repo(name, config=config))
            session = repo.readonly_session(branch=branch)
            ds = xr.open_zarr(
                session.store,
                group=group if len(group) else None,
                zarr_format=3,
                consolidated=False,
            )
            # Add _xpublish_id for caching - use name, branch, and group for arraylake
            xpublish_id = f"{name}:{branch}"
            if group:
                xpublish_id += f":{group}"
            ds.attrs["_xpublish_id"] = xpublish_id
        except ImportError as ie:
            raise ImportError(
                f"Arraylake is not installed, no dataset available named {name}"
            ) from ie
        except Exception as e:
            raise ValueError(
                f"Error occurred while getting dataset from Arraylake: {e}"
            ) from e

    return ds


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
        help="Dataset to serve (default: global). Options: global, air, hrrr, eu3035, or an arraylake dataset name. If an arraylake dataset is specified, the arraylake-org and arraylake-repo must be provided, along with an optional branch and group",
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
    parser.add_argument(
        "--cache",
        action="store_true",
        default=False,
        help="Enable the icechunk cache for Arraylake datasets (default: False)",
    )
    args = parser.parse_args()

    ds = get_dataset_for_name(args.dataset, args.branch, args.group, args.cache)

    xr.set_options(keep_attrs=True)
    rest = xpublish.SingleDatasetRest(
        ds,
        plugins={"tiles": TilesPlugin(), "wms": WMSPlugin()},
    )
    rest.app.add_middleware(CORSMiddleware, allow_origins=["*"])
    rest.serve(host="0.0.0.0", port=args.port)
