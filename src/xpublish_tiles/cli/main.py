"""Simple CLI for playing with xpublish-tiles, with a generated sample dataset"""

import argparse
import logging
import threading
import warnings
from typing import cast

import cf_xarray  # noqa: F401
import xpublish
from fastapi.middleware.cors import CORSMiddleware

import xarray as xr
from xpublish_tiles.cli.bench import run_benchmark
from xpublish_tiles.testing.datasets import (
    CURVILINEAR,
    EU3035,
    EU3035_HIRES,
    GLOBAL_6KM,
    GLOBAL_BENCHMARK_TILES,
    HRRR,
    IFS,
    PARA,
    PARA_HIRES,
    SENTINEL2_NOCOORDS,
    UTM33S,
    UTM33S_HIRES,
    create_global_dataset,
)
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
        ds = PARA.create().assign_attrs(_xpublish_id=name)
    elif name == "eu3035":
        ds = EU3035.create().assign_attrs(_xpublish_id=name)
    elif name == "eu3035_hires":
        ds = EU3035_HIRES.create().assign_attrs(_xpublish_id=name)
    elif name == "ifs":
        ds = IFS.create().assign_attrs(_xpublish_id=name)
    elif name == "curvilinear":
        ds = CURVILINEAR.create().assign_attrs(_xpublish_id=name)
    elif name == "sentinel":
        ds = SENTINEL2_NOCOORDS.create().assign_attrs(_xpublish_id=name)
    elif name == "global-6km":
        ds = GLOBAL_6KM.create().assign_attrs(_xpublish_id=name)
    elif name.startswith("xarray://"):
        # xarray tutorial dataset - format: xarray://dataset_name
        tutorial_name = name.removeprefix("xarray://")
        # these are mostly netCDF files and async loading does not work
        ds = xr.tutorial.load_dataset(tutorial_name).assign_attrs(_xpublish_id=name)
    elif name.startswith("local://"):
        # Local icechunk dataset path
        import icechunk

        # Parse the local path - format: local://dataset_name or local:///path/to/repo::dataset_name
        local_path = name.removeprefix("local://")

        # Check if a custom path is specified (separated by ::)
        if "::" in local_path:
            repo_path, dataset_name = local_path.rsplit("::", 1)
        else:
            # Use default path
            repo_path = "/tmp/tiles-icechunk/"
            dataset_name = local_path

        try:
            storage = icechunk.local_filesystem_storage(repo_path)
            repo = icechunk.Repository.open(storage)

            config: icechunk.RepositoryConfig | None = None
            if icechunk_cache:
                config = icechunk.RepositoryConfig(
                    caching=icechunk.CachingConfig(
                        num_bytes_chunks=1073741824,
                        num_chunk_refs=1073741824,
                        num_bytes_attributes=100_000_000,
                    )
                )

            session = repo.readonly_session(branch=branch)
            ds = xr.open_zarr(
                session.store,
                group=dataset_name,
                zarr_format=3,
                consolidated=False,
                chunks=None,
            )
            # Add _xpublish_id for caching
            xpublish_id = f"local:{dataset_name}:{branch}"
            ds.attrs["_xpublish_id"] = xpublish_id
        except Exception as e:
            raise ValueError(
                f"Error loading local dataset '{dataset_name}' from {repo_path}: {e}"
            ) from e
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
                chunks=None,
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


def get_dataset_object_for_name(name: str):
    """Get the Dataset object for benchmark tiles."""
    # Handle local:// paths by extracting the dataset name
    if name.startswith("local://"):
        # Extract dataset name from local path
        local_path = name.removeprefix("local://")
        if "::" in local_path:
            # Custom path format: local:///path/to/repo::dataset_name
            _, dataset_name = local_path.rsplit("::", 1)
        else:
            # Default format: local://dataset_name
            dataset_name = local_path

        # Map dataset name to Dataset object
        if dataset_name == "hrrr":
            return HRRR
        elif dataset_name == "para":
            return PARA
        elif dataset_name == "para_hires":
            return PARA_HIRES
        elif dataset_name == "eu3035":
            return EU3035
        elif dataset_name == "eu3035_hires":
            return EU3035_HIRES
        elif dataset_name == "ifs":
            return IFS
        elif dataset_name == "sentinel":
            return SENTINEL2_NOCOORDS
        elif dataset_name == "global-6km":
            return GLOBAL_6KM
        elif dataset_name == "utm33s":
            return UTM33S
        elif dataset_name == "utm33s_hires":
            return UTM33S_HIRES
        else:
            return None

    # Handle non-local dataset names
    if name == "hrrr":
        return HRRR
    elif name == "para":
        return PARA
    elif name == "para_hires":
        return PARA_HIRES
    elif name == "eu3035":
        return EU3035
    elif name == "eu3035_hires":
        return EU3035_HIRES
    elif name == "ifs":
        return IFS
    elif name == "sentinel":
        return SENTINEL2_NOCOORDS
    elif name == "global-6km":
        return GLOBAL_6KM
    else:
        return None


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
        help="Dataset to serve (default: global). Options: global, air, hrrr, para, eu3035, ifs, curvilinear, sentinel, global-6km, xarray://<tutorial_name> (loads xarray tutorial dataset), local://<group_name> (loads group from /tmp/tiles-icechunk/), local:///custom/path::<group_name> (loads group from custom icechunk repo), or an arraylake dataset name",
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
        default=True,
        help="Enable the icechunk cache for Arraylake datasets (default: True)",
    )
    parser.add_argument(
        "--spy",
        action="store_true",
        help="Run benchmark requests with the specified dataset",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=12,
        help="Number of concurrent requests for benchmarking (default: 12)",
    )
    parser.add_argument(
        "--where",
        type=str,
        choices=["local", "local-booth", "prod"],
        default="local",
        help="Where to run benchmark requests: 'local' for localhost (starts server), 'local-booth' for localhost (no server), or 'prod' for production (default: local)",
    )
    parser.add_argument(
        "--log-level",
        type=str.lower,
        choices=["debug", "info", "warning", "error"],
        default="warning",
        help="Set the logging level for xpublish_tiles (default: warning)",
    )
    args = parser.parse_args()

    # Configure logging based on CLI argument
    log_level = getattr(logging, args.log_level.upper())

    # Configure xpublish_tiles logger with handler
    logger = logging.getLogger("xpublish_tiles")
    logger.setLevel(log_level)

    # Add console handler if not already present
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Always disable numba, datashader, and PIL debug logs
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("datashader").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Determine dataset to use and benchmarking mode
    dataset_name = args.dataset
    benchmarking = args.spy

    ds = get_dataset_for_name(dataset_name, args.branch, args.group, args.cache)

    xr.set_options(keep_attrs=True)
    if args.where == "local":
        rest = xpublish.SingleDatasetRest(
            ds,
            plugins={"tiles": TilesPlugin(), "wms": WMSPlugin()},
        )
        rest.app.add_middleware(CORSMiddleware, allow_origins=["*"])
    elif args.where == "local-booth":
        # For local-booth, we don't start the REST server
        # Just prepare for benchmarking against existing localhost server
        pass

    # If benchmarking, start the benchmark thread after a delay
    bench_thread = None
    if benchmarking:
        # Get dataset object for potential benchmark tiles
        dataset_obj = get_dataset_object_for_name(dataset_name)
        if dataset_obj and dataset_obj.benchmark_tiles:
            benchmark_tiles = dataset_obj.benchmark_tiles
        else:
            warnings.warn(
                "Unknown dataset; using global tiles", RuntimeWarning, stacklevel=2
            )
            benchmark_tiles = GLOBAL_BENCHMARK_TILES

        # Get the first variable from the dataset
        if not ds.data_vars:
            raise ValueError(f"No data variables found in dataset '{dataset_name}'")
        first_var = next(iter(ds.data_vars))

        # Check if we need colorscalerange
        needs_colorscale = (
            "valid_min" not in ds[first_var].attrs
            or "valid_max" not in ds[first_var].attrs
        )

        bench_thread = threading.Thread(
            target=run_benchmark,
            args=(
                args.port,
                "requests",
                dataset_name,
                benchmark_tiles,
                args.concurrency,
                args.where,
                first_var,
                needs_colorscale,
            ),
            daemon=True,
        )
        bench_thread.start()

    if args.where == "local":
        rest.serve(host="0.0.0.0", port=args.port)
    elif args.where in ["local-booth", "prod"] and bench_thread:
        # When running benchmarks against production or local-booth, wait for the thread to complete
        bench_thread.join()


if __name__ == "__main__":
    main()
