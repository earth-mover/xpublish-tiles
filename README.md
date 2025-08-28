# xpublish-tiles

![PyPI - Version](https://img.shields.io/pypi/v/xpublish-tiles)

Web mapping plugins for Xpublish

## Project Overview
This project contains a set of web mapping plugins for Xpublish - a framework for serving xarray datasets via HTTP APIs.

The goal of this project is to transform xarray datasets to raster, vector and other types of tiles, which can then be served via HTTP APIs. To do this, the package implements a set of xpublish plugins:
* `xpublish_tiles.xpublish.tiles.TilesPlugin`: An [OGC Tiles](https://www.ogc.org/standards/ogcapi-tiles/) conformant plugin for serving raster, vector and other types of tiles.
* `xpublish_tiles.xpublish.wms.WMSPlugin`: An [OGC Web Map Service](https://www.ogc.org/standards/wms/) conformant plugin for serving raster, vector and other types of tiles.

## Development

Sync the environment with [`uv`](https://docs.astral.sh/uv/getting-started/)

```sh
uv sync
```

Run the type checker

```sh
uv run ty check
```

Run the tests

```sh
uv run pytest tests
```

Run setup tests (create local datasets, these can be deployed using the CLI)

```sh
uv run pytest --setup
```

## CLI Usage

The package includes a command-line interface for quickly serving datasets with tiles and WMS endpoints:

```sh
uv run xpublish-tiles [OPTIONS]
```

### Options

- `--port PORT`: Port to serve on (default: 8080)
- `--dataset DATASET`: Dataset to serve (default: global)
  - `global`: Generated global dataset with synthetic data
  - `air`: Tutorial air temperature dataset from xarray tutorial
  - `hrrr`: High-Resolution Rapid Refresh dataset
  - `para`: Parameterized dataset
  - `eu3035`: European dataset in ETRS89 / LAEA Europe projection
  - `eu3035_hires`: High-resolution European dataset
  - `ifs`: Integrated Forecasting System dataset
  - `curvilinear`: Curvilinear coordinate dataset
  - `sentinel`: Sentinel-2 dataset (without coordinates)
  - `global-6km`: Global dataset at 6km resolution
  - `xarray://<tutorial_name>`: Load any xarray tutorial dataset (e.g., `xarray://rasm`)
  - `local://<dataset_name>`: Load dataset from local icechunk repository at `/tmp/tiles-icechunk/` (datasets created with `uv run pytest --setup`)
  - `local:///path/to/repo::<dataset_name>`: Load dataset from custom icechunk repository path
  - For Arraylake datasets: specify the dataset name in {arraylake_org}/{arraylake_dataset} format (requires Arraylake credentials)
- `--branch BRANCH`: Branch to use for Arraylake or icechunk datasets (default: main)
- `--group GROUP`: Group to use for Arraylake datasets (default: '')
- `--cache`: Enable icechunk cache for Arraylake and local icechunk datasets (default: enabled)
- `--spy`: Run benchmark requests with the specified dataset for performance testing
- `--concurrency INT`: Number of concurrent requests for benchmarking (default: 12)
- `--where CHOICE`: Where to run benchmark requests (choices: local, local-booth, prod; default: local)
  - `local`: Start server on localhost and run benchmarks against it
  - `local-booth`: Run benchmarks against existing localhost server (no server startup)
  - `prod`: Run benchmarks against production server
- `--log-level LEVEL`: Set the logging level for xpublish_tiles (choices: debug, info, warning, error; default: warning)

> [!TIP]
> To use local datasets (e.g., `local://ifs`, `local://para_hires`), first create them with `uv run pytest --setup`. This creates icechunk repositories at `/tmp/tiles-icechunk/`.

### Examples

```sh
# Serve synthetic global dataset on default port 8080
xpublish-tiles

# Serve air temperature tutorial dataset on port 9000
xpublish-tiles --port 9000 --dataset air

# Serve built-in test datasets
xpublish-tiles --dataset hrrr
xpublish-tiles --dataset para
xpublish-tiles --dataset eu3035_hires

# Load xarray tutorial datasets
xpublish-tiles --dataset xarray://rasm
xpublish-tiles --dataset xarray://ersstv5

# Serve locally stored datasets (first create them with `uv run pytest --setup`)
xpublish-tiles --dataset local://ifs
xpublish-tiles --dataset local://para_hires

# Serve local icechunk data from custom path
xpublish-tiles --dataset local:///path/to/my/repo::my_dataset

# Serve Arraylake dataset with specific branch and group
xpublish-tiles --dataset earthmover-public/aifs-outputs --branch main --group 2025-04-01/12z

# Run benchmark with a specific dataset
xpublish-tiles --dataset local://para_hires --spy

# Run benchmark with custom concurrency and against production
xpublish-tiles --dataset para --spy --concurrency 20 --where prod

# Enable debug logging
xpublish-tiles --dataset hrrr --log-level debug
```

## Benchmarking

The CLI includes a benchmarking feature that can be used to test tile server performance:

```sh
# Run benchmark with a specific dataset (starts server automatically)
xpublish-tiles --dataset local://para_hires --spy

# Run benchmark against existing localhost server
xpublish-tiles --dataset para --spy --where local-booth

# Run benchmark against production server with custom concurrency
xpublish-tiles --dataset para --spy --where prod --concurrency 8
```

The `--spy` flag enables benchmarking mode. The benchmarking behavior depends on the `--where` option:

- **`--where local`** (default): Starts the tile server and automatically runs benchmark requests against it
- **`--where local-booth`**: Runs benchmarks against an existing localhost server (doesn't start a new server)
- **`--where prod`**: Runs benchmarks against a production server

The benchmarking process:
- Warms up the server with initial tile requests
- Makes concurrent tile requests (configurable with `--concurrency`, default: 12) to test performance
- Uses dataset-specific benchmark tiles or falls back to global tiles
- Automatically exits after completing the benchmark run
- Uses appropriate colorscale ranges based on dataset attributes

Once running, the server provides:
- Tiles API at `http://localhost:8080/tiles/`
- WMS API at `http://localhost:8080/wms/`
- Interactive API documentation at `http://localhost:8080/docs`

An example tile url:
```
http://localhost:8080/tiles/WebMercatorQuad/4/4/14?variables=2t&style=raster/viridis&colorscalerange=280,300&width=256&height=256&valid_time=2025-04-03T06:00:00
```

Where `4/4/14` represents the tile coordinates in {z}/{y}/{x}

## Integration Examples

- [Mapbox Usage](./examples/mapbox/)


## Deployment notes

1. Make sure to limit `NUMBA_NUM_THREADS`; this is used for rendering categorical data with datashader.
2. The first invocation of a render will block while datashader functions are JIT-compiled. Our attempts to add a precompilation step to remove this have been unsuccessful.

### Environment variables
1. `XPUBLISH_TILES_ASYNC_LOAD: [0, 1]` - controls whether Xarray's async loading is used.
2. `XPUBLISH_TILES_NUM_THREADS: int` - controls the size of the threadpool
3. `XPUBLISH_TILES_TRANSFORM_CHUNK_SIZE: int` - when transforming coordinates, do so by submitting (NxN) chunks to the threadpool.
