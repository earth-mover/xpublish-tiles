"""Benchmarking functionality for xpublish-tiles CLI."""

import asyncio
import os
import random
import time

import aiohttp
import requests


def run_benchmark(
    port: int,
    bench_type: str,
    dataset_name: str,
    benchmark_tiles: list[str],
    concurrency: int,
    where: str = "local",
    variable_name: str = "foo",
    needs_colorscale: bool = False,
):
    """Run benchmarking requests for the given dataset."""

    if bench_type != "requests":
        print(f"Unknown benchmark type: {bench_type}")
        return

    # Define tiles to request based on dataset
    if not benchmark_tiles:
        raise ValueError(f"No benchmark tiles defined for dataset '{dataset_name}'")

    warmup_tiles = [benchmark_tiles[0]]  # Use first tile for warmup

    # Randomly shuffle the benchmark tiles to avoid ordering bias
    # Use current time as seed for different order each run
    random.seed(int(time.time() * 1000000))
    shuffled_tiles = benchmark_tiles.copy()
    random.shuffle(shuffled_tiles)

    print(f"Starting benchmark requests for {dataset_name} using endpoint")
    print(f"Warmup tiles: {warmup_tiles}")
    print(f"Benchmark tiles: {len(shuffled_tiles)} tiles (randomized order)")

    # Wait for server to start with warmup
    if where == "local":
        server_url = f"http://localhost:{port}"
    elif where == "local-booth":
        server_url = f"http://localhost:{port}/services/tiles/earthmover-integration/tiles-icechunk/main/{dataset_name}"
    else:  # prod
        server_url = f"https://compute.earthmover.dev/v1/services/tiles/earthmover-integration/tiles-icechunk/main/{dataset_name}"
    max_retries = 10
    for _i in range(max_retries):
        try:
            # Use a tile endpoint for health check and warmup
            z, x, y = warmup_tiles[0].split("/")
            # Build base URL with required parameters
            base_params = (
                f"variables={variable_name}&style=raster/viridis&width=256&height=256"
            )
            if needs_colorscale:
                base_params += "&colorscalerange=-100,100"  # Use reasonable default range

            warmup_url = f"{server_url}/tiles/WebMercatorQuad/{z}/{x}/{y}?{base_params}"
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

    # Make requests to benchmark tiles concurrently using async with semaphore
    print(f"Making concurrent benchmark tile requests (max {concurrency} at a time)...")

    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrency)

    async def fetch_tile(session, tile):
        async with semaphore:  # Acquire semaphore before making request
            z, x, y = tile.split("/")
            # The tile endpoint format is /tiles/{tileMatrixSetId}/{tileMatrix}/{tileCol}/{tileRow}
            # Include required query parameters
            # Build tile URL with required parameters
            tile_params = (
                f"variables={variable_name}&style=raster/viridis&width=256&height=256"
            )
            if needs_colorscale:
                tile_params += "&colorscalerange=-100,100"  # Use reasonable default range

            tile_url = f"{server_url}/tiles/WebMercatorQuad/{z}/{x}/{y}?{tile_params}"

            start_time = time.perf_counter()
            try:
                async with session.get(
                    tile_url, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    duration = time.perf_counter() - start_time
                    if response.status != 200:
                        return {
                            "tile": tile,
                            "status": response.status,
                            "duration": duration,
                            "error": None,
                        }
                    return {
                        "tile": tile,
                        "status": 200,
                        "duration": duration,
                        "error": None,
                    }
            except Exception as e:
                duration = time.perf_counter() - start_time
                return {
                    "tile": tile,
                    "status": None,
                    "duration": duration,
                    "error": f"{type(e).__name__}: {e}",
                }

    async def fetch_all_tiles():
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_tile(session, tile) for tile in shuffled_tiles]
            results = await asyncio.gather(*tasks)
            return results

    # Run the async tile requests
    print(f"Starting benchmark with {len(shuffled_tiles)} tiles...")
    start_time = time.perf_counter()
    results = asyncio.run(fetch_all_tiles())
    total_time = time.perf_counter() - start_time

    # Print detailed results and summary
    print("\n=== Benchmark Results ===")
    successful = 0
    failed = 0
    total_duration = 0
    durations = []

    for result in results:
        if result["error"]:
            print(
                f"  Tile {result['tile']}: ERROR - {result['error']} ({result['duration']:.3f}s)"
            )
            failed += 1
        elif result["status"] != 200:
            print(
                f"  Tile {result['tile']}: {result['status']} ({result['duration']:.3f}s)"
            )
            failed += 1
        else:
            # Only include timing for successful requests
            total_duration += result["duration"]
            durations.append(result["duration"])
            successful += 1

    # Calculate statistics
    avg_duration = sum(durations) / len(durations) if durations else 0
    min_duration = min(durations) if durations else 0
    max_duration = max(durations) if durations else 0
    requests_per_second = len(shuffled_tiles) / total_time if total_time > 0 else 0

    print("\n=== Summary ===")
    print(f"Total tiles: {len(shuffled_tiles)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total wall time: {total_time:.3f}s")
    print(f"Avg request time: {avg_duration:.3f}s (successful only)")
    print(f"Min request time: {min_duration:.3f}s (successful only)")
    print(f"Max request time: {max_duration:.3f}s (successful only)")
    print(f"Requests/second: {requests_per_second:.2f}")
    print("Benchmark completed!")

    # Exit the process since this is a benchmarking run
    os._exit(0)
