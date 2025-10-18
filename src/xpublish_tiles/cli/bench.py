"""Benchmarking functionality for xpublish-tiles CLI."""

import time

import aiohttp
import requests

from xpublish_tiles.cli.benchmark_common import (
    calculate_benchmark_stats,
    finalize_benchmark_results,
    print_benchmark_results,
    run_concurrent_tile_requests,
)


def run_benchmark(
    port: int,
    bench_type: str,
    dataset_name: str,
    benchmark_tiles: list[str],
    concurrency: int,
    where: str = "local",
    variable_name: str = "foo",
    needs_colorscale: bool = False,
    return_results: bool = False,
) -> dict | None:
    """Run benchmarking requests for the given dataset.

    If return_results is True, returns a dict with benchmark results instead of exiting.
    """

    if bench_type != "requests":
        print(f"Unknown benchmark type: {bench_type}")
        return

    # Define tiles to request based on dataset
    if not benchmark_tiles:
        raise ValueError(f"No benchmark tiles defined for dataset '{dataset_name}'")

    warmup_tiles = [benchmark_tiles[0]]  # Use first tile for warmup

    print(f"Starting benchmark requests for {dataset_name} using endpoint")
    print(f"Warmup tiles: {warmup_tiles}")
    print(f"Benchmark tiles: {len(benchmark_tiles)} tiles")

    # Determine server URL based on where parameter
    if where == "local":
        server_url = f"http://localhost:{port}"
    elif where == "local-booth":
        server_url = f"http://localhost:{port}/services/tiles/earthmover-integration/tiles-icechunk/main/{dataset_name}"
    else:  # prod
        server_url = f"https://compute.earthmover.dev/v1/services/tiles/earthmover-integration/tiles-icechunk/main/{dataset_name}"

    # For local servers, we can use a health check endpoint if available, otherwise warmup with tiles
    if where == "local":
        # For local servers, try a basic warmup with the first tile
        z, x, y = warmup_tiles[0].split("/")
        base_params = (
            f"variables={variable_name}&style=raster/viridis&width=256&height=256"
        )
        if needs_colorscale:
            base_params += "&colorscalerange=-100,100"
        warmup_url = f"{server_url}/tiles/WebMercatorQuad/{z}/{x}/{y}?{base_params}"

        # Custom warmup for xpublish-tiles
        max_retries = 10
        for _i in range(max_retries):
            try:
                response = requests.get(warmup_url, timeout=10)
                if response.status_code == 200:
                    print(
                        f"Server is ready at {server_url} (warmed up with tile {warmup_tiles[0]})"
                    )
                    break
                else:
                    print(
                        f"Warmup request returned status {response.status_code}, retrying..."
                    )
            except Exception as e:
                print(f"Warmup request failed: {e}, retrying...")
            time.sleep(0.5)
        else:
            print(f"ERROR: Server warmup failed after {max_retries} attempts")
            print(f"Failed to get 200 response from: {warmup_url}")
            raise RuntimeError("Server warmup failed - did not receive 200 response")
    else:
        # For remote servers, just assume they're ready
        print(f"Using server at {server_url}")

    # Make requests to benchmark tiles concurrently
    print(f"Making concurrent benchmark tile requests (max {concurrency} at a time)...")

    async def fetch_tile(session: aiohttp.ClientSession, tile: str):
        """Fetch a single tile using xpublish-tiles API."""
        z, x, y = tile.split("/")
        # Build tile URL with required parameters
        tile_params = (
            f"variables={variable_name}&style=raster/viridis&width=256&height=256"
        )
        if needs_colorscale:
            tile_params += "&colorscalerange=-100,100"

        tile_url = f"{server_url}/tiles/WebMercatorQuad/{z}/{x}/{y}?{tile_params}"

        start_time = time.perf_counter()
        try:
            async with session.get(
                tile_url, timeout=aiohttp.ClientTimeout(total=90)
            ) as response:
                duration = time.perf_counter() - start_time
                return {
                    "tile": tile,
                    "status": response.status,
                    "duration": duration,
                    "error": None
                    if response.status == 200
                    else f"HTTP {response.status}",
                }
        except Exception as e:
            duration = time.perf_counter() - start_time
            return {
                "tile": tile,
                "status": None,
                "duration": duration,
                "error": f"{type(e).__name__}: {e}",
            }

    # Run the async tile requests
    print(f"Starting benchmark with {len(benchmark_tiles)} tiles...")
    start_time = time.perf_counter()

    import asyncio

    results = asyncio.run(
        run_concurrent_tile_requests(
            tiles=benchmark_tiles,
            concurrency=concurrency,
            fetch_tile_func=fetch_tile,
        )
    )
    total_time = time.perf_counter() - start_time

    # Print results and calculate statistics
    print_benchmark_results(results, total_time, dataset_name)
    stats = calculate_benchmark_stats(results, total_time)

    return finalize_benchmark_results(
        stats=stats,
        dataset_name=dataset_name,
        return_results=return_results,
    )
