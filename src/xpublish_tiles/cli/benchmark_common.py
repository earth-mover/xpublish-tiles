"""Common benchmarking functionality shared between xpublish-tiles and titiler benchmarks."""

import asyncio
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import aiohttp


@dataclass
class BenchmarkResult:
    """Represents the result of a single tile request."""

    tile: str
    status: int | None
    duration: float
    error: str | None = None


async def run_concurrent_tile_requests(
    tiles: list[str],
    concurrency: int,
    fetch_tile_func: Callable[[aiohttp.ClientSession, str], Any],
    shuffle_seed: int | None = None,
) -> list[BenchmarkResult]:
    """Run concurrent tile requests using the provided fetch function.

    Args:
        tiles: List of tile coordinates to request
        concurrency: Maximum number of concurrent requests
        fetch_tile_func: Async function to fetch a single tile
        shuffle_seed: Optional seed for shuffling tiles (uses current time if None)

    Returns:
        List of BenchmarkResult objects
    """
    # Randomly shuffle the benchmark tiles to avoid ordering bias
    if shuffle_seed is None:
        shuffle_seed = int(time.time() * 1000000)
    random.seed(shuffle_seed)
    shuffled_tiles = tiles.copy()
    # random.shuffle(shuffled_tiles)  # Currently commented out in original code

    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrency)

    async def fetch_tile_with_semaphore(session: aiohttp.ClientSession, tile: str):
        async with semaphore:
            result_dict = await fetch_tile_func(session, tile)
            # Convert dict result to BenchmarkResult object
            return BenchmarkResult(
                tile=result_dict["tile"],
                status=result_dict["status"],
                duration=result_dict["duration"],
                error=result_dict["error"],
            )

    async def fetch_all_tiles():
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_tile_with_semaphore(session, tile) for tile in shuffled_tiles]
            results = await asyncio.gather(*tasks)
            return results

    return await fetch_all_tiles()


def calculate_benchmark_stats(results: list[BenchmarkResult], total_time: float) -> dict:
    """Calculate benchmark statistics from request results.

    Args:
        results: List of result dictionaries from tile requests
        total_time: Total wall clock time for all requests

    Returns:
        Dictionary with benchmark statistics
    """
    successful = 0
    failed = 0
    durations = []

    for result in results:
        if result.error and result.status is None:
            failed += 1
        elif result.status != 200:
            failed += 1
        else:
            durations.append(result.duration)
            successful += 1

    # Calculate statistics
    avg_duration = sum(durations) / len(durations) if durations else 0
    min_duration = min(durations) if durations else 0
    max_duration = max(durations) if durations else 0
    requests_per_second = len(results) / total_time if total_time > 0 else 0

    return {
        "total_tiles": len(results),
        "successful": successful,
        "failed": failed,
        "total_wall_time": total_time,
        "avg_request_time": avg_duration,
        "min_request_time": min_duration,
        "max_request_time": max_duration,
        "requests_per_second": requests_per_second,
        "durations": durations,  # Keep raw durations for further processing if needed
    }


def print_benchmark_results(
    results: list[BenchmarkResult],
    total_time: float,
    dataset_name: str | None,
    implementation: str = "",
):
    """Print detailed benchmark results to console.

    Args:
        results: List of result dictionaries from tile requests
        total_time: Total wall clock time for all requests
        dataset_name: Name of the dataset being benchmarked
        implementation: Name of the implementation (e.g., "Titiler", "")
    """
    stats = calculate_benchmark_stats(results, total_time)

    print(f"\n=== {implementation + ' ' if implementation else ''}Benchmark Results ===")

    # Print detailed results for failed requests
    for result in results:
        if result.error and result.status is None:
            print(
                f"  Tile {result.tile}: ERROR - {result.error} ({result.duration:.3f}s)"
            )
        elif result.status != 200:
            error_msg = result.error or "No error details"
            print(
                f"  Tile {result.tile}: {result.status} - {error_msg} ({result.duration:.3f}s)"
            )

    # Print summary statistics
    print(f"Total tiles: {stats['total_tiles']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total wall time: {stats['total_wall_time']:.3f}s")
    print(f"Avg request time: {stats['avg_request_time']:.3f}s")
    print(f"Requests/second: {stats['requests_per_second']:.2f}")
    print(f"{implementation + ' ' if implementation else ''}benchmark completed!")


def wait_for_server_ready(
    server_url: str, health_endpoint: str = "/health", max_retries: int = 10
) -> bool:
    """Wait for server to be ready by checking health endpoint.

    Args:
        server_url: Base URL of the server
        health_endpoint: Health check endpoint path
        max_retries: Maximum number of retry attempts

    Returns:
        True if server is ready, False if timeout
    """
    import requests

    for _ in range(max_retries):
        try:
            health_url = f"{server_url}{health_endpoint}"
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                print(f"Server is ready at {server_url}")
                return True
            else:
                print(f"Health check returned status {response.status_code}, retrying...")
        except Exception as e:
            print(f"Health check failed: {e}, retrying...")
        time.sleep(0.5)

    print(f"ERROR: Server warmup failed after {max_retries} attempts")
    return False


def finalize_benchmark_results(
    stats: dict,
    dataset_name: str | None,
    return_results: bool = False,
    implementation_suffix: str = "",
) -> dict | None:
    """Finalize benchmark results - either return them or exit process.

    Args:
        stats: Statistics dictionary from calculate_benchmark_stats
        dataset_name: Name of the dataset
        return_results: Whether to return results or exit
        implementation_suffix: Suffix to add to dataset name (e.g., "_titiler")

    Returns:
        Results dictionary if return_results=True, otherwise exits process
    """
    if return_results:
        return {
            "dataset": f"{dataset_name}{implementation_suffix}",
            **{
                k: v for k, v in stats.items() if k != "durations"
            },  # Exclude raw durations
        }
    else:
        # Exit the process since this is a benchmarking run
        os._exit(0)
