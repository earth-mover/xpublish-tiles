"""Simple CLI for playing with xpublish-tiles, with a generated sample dataset"""

import argparse
import json
import logging
import os
import subprocess
import threading
import time
import warnings
from typing import cast

import cf_xarray  # noqa: F401
import uvicorn
import xpublish
from fastapi.middleware.cors import CORSMiddleware

import xarray as xr
from xpublish_tiles.cli.bench import run_benchmark
from xpublish_tiles.cli.titiler_bench import run_titiler_benchmark
from xpublish_tiles.logger import setup_logging
from xpublish_tiles.testing.datasets import (
    DATASET_LOOKUP,
    GLOBAL_BENCHMARK_TILES,
    create_global_dataset,
)
from xpublish_tiles.xpublish.tiles.plugin import TilesPlugin
from xpublish_tiles.xpublish.wms.plugin import WMSPlugin

try:
    import icechunk as ic

    ICECHUNK_CONFIG = ic.RepositoryConfig(
        caching=ic.CachingConfig(
            num_bytes_chunks=1073741824,
            num_chunk_refs=1073741824,
            num_bytes_attributes=100_000_000,
            num_snapshot_nodes=100_000_000,
        ),
        manifest=ic.ManifestConfig(
            preload=ic.ManifestPreloadConfig(
                preload_if=ic.ManifestPreloadCondition.false()
            )
        ),
    )
except ImportError:
    ICECHUNK_CONFIG = None


def create_onecrs_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Create a single-CRS version of a dataset by dropping alternate CRS coordinates."""
    # Drop alternate CRS coordinates and their corresponding CRS variables
    coords_to_drop = [
        coord
        for coord in ds.coords
        if coord in ["x_3857", "y_3857", "x_4326", "y_4326", "crs_3857", "crs_4326"]
    ]
    if coords_to_drop:
        ds = ds.drop_vars(coords_to_drop)
    # Set grid_mapping to just spatial_ref
    ds["foo"].attrs["grid_mapping"] = "spatial_ref"
    return ds


def get_dataset_for_name(
    name: str, branch: str = "main", group: str = "", icechunk_cache: bool = False
) -> xr.Dataset:
    if name == "global":
        ds = create_global_dataset().assign_attrs(_xpublish_id=name)
    elif name == "air":
        ds = xr.tutorial.open_dataset("air_temperature").assign_attrs(_xpublish_id=name)
    elif name in DATASET_LOOKUP:
        ds = DATASET_LOOKUP[name].create().assign_attrs(_xpublish_id=name)
    elif name.startswith("xarray://"):
        # xarray tutorial dataset - format: xarray://dataset_name
        tutorial_name = name.removeprefix("xarray://")
        # these are mostly netCDF files and async loading does not work
        ds = xr.tutorial.load_dataset(tutorial_name).assign_attrs(_xpublish_id=name)
    elif name.startswith("local://"):
        try:
            import icechunk
        except ImportError as ie:
            raise ImportError("icechunk is not installed") from ie

        local_path = name.removeprefix("local://")

        repo_path, dataset_name = (
            local_path.rsplit("::", 1)
            if "::" in local_path
            else ("/tmp/tiles-icechunk/", local_path)
        )

        try:
            if "s3://" in repo_path:
                storage = icechunk.s3_storage(
                    bucket=repo_path.removeprefix("s3://").split("/")[0],
                    prefix="/".join(repo_path.removeprefix("s3://").split("/")[1:]),
                )
            else:
                storage = icechunk.local_filesystem_storage(repo_path)

            config: icechunk.RepositoryConfig | None = None
            if icechunk_cache:
                config = ICECHUNK_CONFIG
            repo = icechunk.Repository.open(storage, config=config)

            session = repo.readonly_session(branch=branch)
            ds = xr.open_zarr(
                session.store,
                group="utm50s_hires"
                if dataset_name == "utm50s_hires_onecrs"
                else dataset_name,
                zarr_format=3,
                consolidated=False,
                chunks=None,
            )
            # Add _xpublish_id for caching
            xpublish_id = f"local:{dataset_name}:{branch}"
            ds.attrs["_xpublish_id"] = xpublish_id
            # Handle synthetic datasets
            if dataset_name == "utm50s_hires_onecrs":
                ds = create_onecrs_dataset(ds)
        except Exception as e:
            raise ValueError(
                f"Error loading local dataset '{dataset_name}' from {repo_path}: {e}"
            ) from e
    else:
        try:
            from arraylake import Client

            import icechunk

            config: icechunk.RepositoryConfig | None = None
            if icechunk_cache:
                config = ICECHUNK_CONFIG

            client = Client()
            repo = cast(icechunk.Repository, client.get_repo(name, config=config))
            session = repo.readonly_session(branch=branch)
            ds = xr.open_zarr(
                session.store,
                group=group or None,
                zarr_format=3,
                consolidated=False,
                chunks=None,
            )
            # Add _xpublish_id for caching - use name, branch, and group for arraylake
            xpublish_id = f"{name}:{branch}"
            xpublish_id += f":{group}" if group else ""
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


def _setup_benchmark_server(ds, port):
    """Setup and start a benchmark server in a separate thread."""
    rest = xpublish.SingleDatasetRest(
        ds, plugins={"tiles": TilesPlugin(), "wms": WMSPlugin()}
    )
    rest.app.add_middleware(CORSMiddleware, allow_origins=["*"])
    config = uvicorn.Config(
        rest.app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
    )
    server = uvicorn.Server(config)

    server_thread = threading.Thread(target=server.run, daemon=False)
    server_thread.start()
    time.sleep(1)  # Give server a moment to start

    return server, server_thread


def _teardown_benchmark_server(server, server_thread):
    """Properly shutdown the benchmark server."""
    server.should_exit = True
    server_thread.join(timeout=5)


def _get_dataset_benchmark_info(ds, dataset_name):
    """Get first variable and colorscale requirements for a dataset."""
    if not ds.data_vars:
        return None, None, None

    first_var = next(iter(ds.data_vars))
    needs_colorscale = (
        "valid_min" not in ds[first_var].attrs or "valid_max" not in ds[first_var].attrs
    )

    dataset_obj = get_dataset_object_for_name(dataset_name)
    if dataset_obj and dataset_obj.benchmark_tiles:
        benchmark_tiles = dataset_obj.benchmark_tiles
    else:
        warnings.warn("Unknown dataset; using global tiles", RuntimeWarning, stacklevel=2)
        benchmark_tiles = GLOBAL_BENCHMARK_TILES

    return first_var, needs_colorscale, benchmark_tiles


def _run_single_dataset_benchmark_subprocess(
    dataset_name, args, use_titiler=False, dataset_arg=None
):
    """Run benchmark for a single dataset in a subprocess. Returns benchmark result or None if failed.

    We use subprocess isolation to avoid asyncio event loop conflicts and library state
    pollution that can occur when running multiple benchmarks sequentially. Some libraries
    (like asyncio semaphores in xarray/dask stack) can get bound to specific event loops
    and cause "bound to a different event loop" errors on subsequent runs.
    """
    try:
        # Build command to run benchmark in subprocess using --spy mode
        # fmt: off
        cmd = [
            "uv", "run", "xpublish-tiles",
            "--dataset", dataset_arg or f"local://{dataset_name}",
            "--port", str(args.port),
            "--concurrency", str(args.concurrency),
            "--where", args.where,
            "--log-level", "debug",
            "--spy"
        ]
        if use_titiler:
            cmd.append("--titiler")
        # fmt: on

        print(f"  Running: {' '.join(cmd)}")

        # Run subprocess and capture output
        env = os.environ.copy()
        env["XPUBLISH_TILES_MAX_RENDERABLE_SIZE"] = "400000000"
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
        # print(result.stdout)

        if result.returncode != 0:
            print(f"  ERROR: Subprocess failed with return code {result.returncode}")
            if result.stderr:
                print(f"  STDERR: {result.stderr[:500]}")
            return None

        # Parse the JSON result from stdout
        for line in result.stdout.split("\n"):
            if line.startswith("BENCHMARK_RESULT_JSON:"):
                json_str = line[len("BENCHMARK_RESULT_JSON:") :].strip()
                return json.loads(json_str)

        print("  ERROR: No benchmark result found in output")
        return None

    except subprocess.TimeoutExpired:
        print("  ERROR: Benchmark timed out after 5 minutes")
        return None
    except Exception as e:
        print(f"  ERROR: Failed to run benchmark subprocess for {dataset_name}: {e}")
        return None


def _run_single_dataset_benchmark(dataset_name, args, ds=None):
    """Run benchmark for a single dataset. Returns benchmark result or None if failed."""
    try:
        if ds is None:
            ds_name = (
                dataset_name
                if dataset_name in ["global", "air"]
                else f"local://{dataset_name}"
            )
            ds = get_dataset_for_name(ds_name, args.branch, args.group, args.cache)

        first_var, needs_colorscale, benchmark_tiles = _get_dataset_benchmark_info(
            ds, dataset_name
        )
        if first_var is None:
            print(
                f"  WARNING: No data variables found in dataset '{dataset_name}', skipping..."
            )
            return None

        # Use titiler.xarray if requested
        if hasattr(args, "titiler") and args.titiler:
            return run_titiler_benchmark(
                dataset=ds,
                dataset_name=dataset_name,
                benchmark_tiles=benchmark_tiles,
                port=args.port,
                concurrency=args.concurrency,
                where=args.where,
                variable_name=first_var,
                needs_colorscale=needs_colorscale,
                return_results=True,
            )

        # Use xpublish-tiles (default)
        if args.where == "local":
            server, server_thread = _setup_benchmark_server(ds, args.port)

            try:
                result = run_benchmark(
                    port=args.port,
                    bench_type="requests",
                    dataset_name=dataset_name,
                    benchmark_tiles=benchmark_tiles,
                    concurrency=args.concurrency,
                    where=args.where,
                    variable_name=first_var,
                    needs_colorscale=needs_colorscale,
                    return_results=True,
                )
                _teardown_benchmark_server(server, server_thread)
                return result

            except Exception as e:
                print(f"  ERROR: Benchmark failed for {dataset_name}: {e}")
                _teardown_benchmark_server(server, server_thread)
                return None
        else:
            return run_benchmark(
                port=args.port,
                bench_type="requests",
                dataset_name=dataset_name,
                benchmark_tiles=benchmark_tiles,
                concurrency=args.concurrency,
                where=args.where,
                variable_name=first_var,
                needs_colorscale=needs_colorscale,
                return_results=True,
            )

    except Exception as e:
        print(f"  ERROR: Failed to benchmark {dataset_name}: {e}")
        return None


def run_bench_suite(args):
    """Run benchmarks for all local datasets and tabulate results."""

    available_datasets = [
        "ifs",
        "hrrr",
        "para_hires",
        "eu3035_hires",
        "utm50s_hires",
        "utm50s_hires_onecrs",
        "global_6km",
        "redgauss_n320",
        # This dataset needs to be updated
        # "sentinel",
    ]

    # Run both xpublish-tiles and titiler benchmarks if titiler requested
    implementations = ["xpublish-tiles"]
    if hasattr(args, "titiler") and args.titiler:
        implementations.append("titiler")

    print(
        f"Running benchmark suite for {len(available_datasets)} datasets at {args.where}..."
    )
    print(f"Implementations: {', '.join(implementations)}")
    print(f"Concurrency: {args.concurrency} requests")
    print(f"Port: {args.port}")
    print("-" * 80)

    results = []

    for impl in implementations:
        use_titiler = impl == "titiler"
        print(f"\n{'=' * 20} {impl.upper()} BENCHMARKS {'=' * 20}")

        for dataset_name in available_datasets:
            print(f"\n=== Benchmarking {dataset_name} with {impl} ===")
            # Don't prefix 'global' and other synthetic datasets with 'local://'
            dataset_arg = (
                dataset_name
                if dataset_name in ["global", "air"]
                else f"local://{dataset_name}"
            )
            result = _run_single_dataset_benchmark_subprocess(
                dataset_name, args, use_titiler, dataset_arg
            )
            if result:
                results.append(result)

    # Print tabulated results
    print("\n" + "=" * 85)
    print("BENCHMARK SUITE RESULTS")
    print("=" * 85)

    if not results:
        print("No benchmarks completed successfully.")
        return

    print(
        f"\n{'Dataset':>23} {'Tiles':>8} {'Success':>8} {'Failed':>8} {'Wall Time':>10} {'Avg Time':>10} {'Req/s':>10}"
    )
    print("-" * 85)

    total_wall_time = 0
    total_failed = 0
    has_failures = False

    for r in results:
        failed_count = r["failed"]
        total_failed += failed_count
        if failed_count > 0:
            has_failures = True

        # Highlight failed requests if any failures, green circle for success
        failed_display = (
            f"🔴 {failed_count}" if failed_count > 0 else f"🟢 {failed_count}"
        )

        dataset = r["dataset"].replace("utm50s_hires_onecrs", "utm50s_1crs")
        print(
            f"{dataset:>23} {r['total_tiles']:>8} {r['successful']:>8} {failed_display:>7} "
            f"{r['total_wall_time']:>10.3f} {r['avg_request_time']:>10.3f} {r['requests_per_second']:>10.2f}"
        )
        total_wall_time += r["total_wall_time"]

    print("-" * 85)
    print(f"Total Benchmark Time: {total_wall_time:.3f}s")

    if has_failures:
        print(f"🔴 WARNING: {total_failed} total failed requests detected!")
    else:
        print("🟢 All requests completed successfully!")


def get_dataset_object_for_name(name: str):
    """Get the Dataset object for benchmark tiles."""
    if name.startswith("local://"):
        local_path = name.removeprefix("local://")
        _, dataset_name = (
            local_path.rsplit("::", 1) if "::" in local_path else (None, local_path)
        )
        # Handle synthetic datasets
        if dataset_name == "utm50s_hires_onecrs":
            return DATASET_LOOKUP.get("utm50s_hires")
        return DATASET_LOOKUP.get(dataset_name)
    return DATASET_LOOKUP.get(name)


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
        "--bench",
        action="store_true",
        help="Run benchmark requests with the specified dataset",
    )
    parser.add_argument(
        "--spy",
        action="store_true",
        help="Run benchmark requests with the specified dataset (alias for --bench)",
    )
    parser.add_argument(
        "--bench-suite",
        action="store_true",
        help="Run benchmarks for all local datasets and tabulate results",
    )
    parser.add_argument(
        "--titiler",
        action="store_true",
        help="Use titiler.xarray instead of xpublish-tiles for benchmarking. With --bench-suite, runs both implementations for comparison",
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
    setup_logging(log_level)

    # Check if we're running bench-suite mode
    if args.bench_suite:
        run_bench_suite(args)
        return

    # Determine dataset to use and benchmarking mode
    dataset_name = args.dataset
    benchmarking = args.bench or args.spy

    # Load dataset and setup server
    ds = get_dataset_for_name(dataset_name, args.branch, args.group, args.cache)

    xr.set_options(keep_attrs=True)
    if args.where == "local":
        rest = xpublish.SingleDatasetRest(
            ds,
            plugins={"tiles": TilesPlugin(), "wms": WMSPlugin()},
        )
        rest.app.add_middleware(CORSMiddleware, allow_origins=["*"])

    if benchmarking:
        if args.spy:
            # For spy mode, run single dataset benchmark and output JSON
            result = _run_single_dataset_benchmark(dataset_name, args, ds)
            if result:
                failed_count = result.get("failed", 0)
                if failed_count > 0:
                    print(f"🔴 WARNING: {failed_count} failed requests detected!")
                else:
                    print("🟢 All requests completed successfully!")
            print("BENCHMARK_RESULT_JSON:", json.dumps(result))
        else:
            # Regular --bench mode (legacy threading approach)
            dataset_obj = get_dataset_object_for_name(dataset_name)
            if dataset_obj and dataset_obj.benchmark_tiles:
                benchmark_tiles = dataset_obj.benchmark_tiles
            else:
                warnings.warn(
                    "Unknown dataset; using global tiles", RuntimeWarning, stacklevel=2
                )
                benchmark_tiles = GLOBAL_BENCHMARK_TILES

            if not ds.data_vars:
                raise ValueError(f"No data variables found in dataset '{dataset_name}'")
            first_var = next(iter(ds.data_vars))

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
            elif args.where in ["local-booth", "prod"]:
                bench_thread.join()
    elif args.where == "local":
        rest.serve(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
