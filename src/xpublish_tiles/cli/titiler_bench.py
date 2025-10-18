"""Titiler.xarray benchmarking functionality for comparison with xpublish-tiles."""

import threading
import time
from collections.abc import Callable

import attr
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from titiler.xarray.factory import TilerFactory
from titiler.xarray.io import Reader

import xarray as xr
from xpublish_tiles.cli.benchmark_common import (
    calculate_benchmark_stats,
    finalize_benchmark_results,
    print_benchmark_results,
    run_concurrent_tile_requests,
    wait_for_server_ready,
)

# Global storage for in-memory datasets
_DATASET_STORE = {}


def in_memory_dataset_opener(src_path: str, **kwargs) -> xr.Dataset:
    """Custom opener that returns datasets from in-memory storage."""
    if src_path in _DATASET_STORE:
        return _DATASET_STORE[src_path]
    else:
        raise ValueError(f"Dataset not found in memory store: {src_path}")


@attr.s
class InMemoryReader(Reader):
    """Custom Reader that uses in-memory datasets."""

    opener: Callable[..., xr.Dataset] = attr.ib(default=in_memory_dataset_opener)


class TitilerXarrayBenchmarkServer:
    """A benchmark server using titiler.xarray for comparison."""

    def __init__(
        self, dataset: xr.Dataset, port: int = 8080, dataset_name: str = "dataset"
    ):
        print(dataset)
        if len(dataset.dims) > 3:
            # titler can't handle >3D datasets directly
            if "time" in dataset.dims:
                dataset = dataset.isel(time=0)
            if "step" in dataset.dims:
                dataset = dataset.isel(step=0)

        self.dataset = dataset
        self.port = port
        self.app = None
        self.server = None
        self.server_thread = None
        self.dataset_name = dataset_name

    def create_app(self) -> FastAPI:
        """Create a FastAPI app with titiler.xarray endpoints."""

        app = FastAPI(
            title="Titiler.xarray Benchmark Server",
            description="Benchmark server using titiler.xarray for performance comparison",
            version="0.1.0",
        )

        # Add CORS middleware
        app.add_middleware(CORSMiddleware, allow_origins=["*"])

        # Store dataset in global store for the custom opener
        _DATASET_STORE[self.dataset_name] = self.dataset

        # Create the tiler factory with custom reader
        tiler = TilerFactory(reader=InMemoryReader)

        # Create a health check endpoint
        @app.get("/health")
        def health_check():
            """Health check endpoint."""
            return {"status": "ok", "dataset_name": self.dataset_name}

        # Create dataset info endpoint
        @app.get("/info")
        def dataset_info():
            """Return dataset information."""
            return {
                "variables": list(self.dataset.data_vars.keys()),
                "dims": dict(self.dataset.dims),
                "coords": list(self.dataset.coords.keys()),
            }

        # Mount the tiler routes
        app.include_router(tiler.router, tags=["Xarray Tiler"])

        return app

    def start(self):
        """Start the titiler server in a separate thread."""
        self.app = self.create_app()

        # Configure uvicorn server
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info",
            access_log=True,
        )
        self.server = uvicorn.Server(config)

        # Start server in a separate thread
        self.server_thread = threading.Thread(target=self.server.run, daemon=False)
        self.server_thread.start()
        time.sleep(3)  # Give server time to start

    def stop(self):
        """Stop the titiler server and cleanup."""
        if self.server:
            self.server.should_exit = True
        if self.server_thread:
            self.server_thread.join(timeout=5)

        # Clean up dataset from global store
        if self.dataset_name in _DATASET_STORE:
            del _DATASET_STORE[self.dataset_name]


def run_titiler_benchmark(
    dataset: xr.Dataset,
    dataset_name: str,
    benchmark_tiles: list[str],
    port: int = 8080,
    concurrency: int = 12,
    where: str = "local",
    variable_name: str = "foo",
    needs_colorscale: bool = False,
    return_results: bool = False,
) -> dict | None:
    """Run benchmark using titiler.xarray server.

    Args:
        dataset: The xarray dataset to serve
        dataset_name: Name of the dataset for reporting
        benchmark_tiles: List of tile coordinates to benchmark
        port: Port to run the server on
        concurrency: Number of concurrent requests
        where: Where to run benchmarks (local, local-booth, prod)
        variable_name: Variable name to request in tiles
        needs_colorscale: Whether colorscale parameter is needed
        return_results: Whether to return results dict or exit process

    Returns:
        Benchmark results dict if return_results=True, otherwise exits process
    """
    server = None

    try:
        if where == "local":
            # Start titiler server
            server = TitilerXarrayBenchmarkServer(dataset, port, dataset_name)
            server.start()
            print(f"Started titiler.xarray server on port {port}")

        # Run the benchmark using titiler-specific request patterns
        result = run_titiler_benchmark_requests(
            dataset_name=server.dataset_name if server else None,
            port=port,
            benchmark_tiles=benchmark_tiles,
            concurrency=concurrency,
            where=where,
            variable_name=variable_name,
            needs_colorscale=needs_colorscale,
            return_results=return_results,
        )

        return result

    finally:
        if server:
            server.stop()


def run_titiler_benchmark_requests(
    dataset_name: str | None,
    port: int,
    benchmark_tiles: list[str],
    concurrency: int,
    where: str = "local",
    variable_name: str = "foo",
    needs_colorscale: bool = False,
    return_results: bool = False,
) -> dict | None:
    """Run benchmarking requests against titiler.xarray server.

    This is adapted for titiler.xarray URL patterns and API endpoints.
    """
    if not benchmark_tiles:
        raise ValueError(f"No benchmark tiles defined for dataset '{dataset_name}'")

    print(f"Starting titiler.xarray benchmark requests for {dataset_name}")
    print(f"Warmup tiles: {[benchmark_tiles[0]]}")
    print(f"Benchmark tiles: {len(benchmark_tiles)} tiles")

    # Wait for server to start with warmup
    if where == "local":
        server_url = f"http://localhost:{port}"
    else:
        raise ValueError("Titiler benchmark only supports 'local' where option")

    # Wait for server to be ready using health endpoint
    if not wait_for_server_ready(server_url, "/health"):
        raise RuntimeError("Titiler server warmup failed")

    # Make requests to benchmark tiles concurrently
    print(f"Making concurrent benchmark tile requests (max {concurrency} at a time)...")

    async def fetch_tile(session, tile):
        """Fetch a single tile using titiler.xarray API."""
        z, x, y = tile.split("/")

        # Build titiler.xarray tile URL - uses tileMatrix/tileRow/tileCol format
        tile_url = f"{server_url}/tiles/WebMercatorQuad/{z}/{y}/{x}"

        # Titiler.xarray parameters
        params = {
            "url": dataset_name,  # Our custom dataset key
            "variable": variable_name,  # titiler uses 'variable' (singular)
            "style": "raster/viridis",  # Format: {style}/{colormap}
            "width": 256,  # Required parameter
            "height": 256,  # Required parameter
            "f": "image/png",  # Output format
        }
        if needs_colorscale:
            params["colorscalerange"] = "-100,100"  # titiler.xarray uses colorscalerange

        start_time = time.perf_counter()
        try:
            import aiohttp

            async with session.get(
                tile_url, params=params, timeout=aiohttp.ClientTimeout(total=90)
            ) as response:
                duration = time.perf_counter() - start_time
                if response.status != 200:
                    # Read the error response to debug errors
                    error_text = await response.text()
                    return {
                        "tile": tile,
                        "status": response.status,
                        "duration": duration,
                        "error": error_text[:200] if error_text else None,
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

    # Run the async tile requests
    print(f"Starting titiler benchmark with {len(benchmark_tiles)} tiles...")
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
    print_benchmark_results(results, total_time, dataset_name, "Titiler")
    stats = calculate_benchmark_stats(results, total_time)

    return finalize_benchmark_results(
        stats=stats,
        dataset_name=dataset_name,
        return_results=return_results,
        implementation_suffix="_titiler",
    )
