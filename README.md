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
  - `air`: Tutorial air temperature dataset
  - For Arraylake datasets: specify the dataset name in {arraylake_org}/{arraylake_dataset} format (requires Arraylake credentials)
- `--branch BRANCH`: Branch to use for Arraylake datasets (default: main)
- `--group GROUP`: Group to use for Arraylake datasets (default: '')
- `--cache`: Enable icechunk cache for Arraylake datasets
- `--spy`: Run benchmark requests with the specified dataset for performance testing

> [!TIP]
> You can control if the tile servers data loading is async or not with the `XPUBLISH_TILES_ASYNC_LOAD` environment variable (`1` for async mode, `0` for sync mode, async is enabled by default). You can also control the zarr concurrency with the `ZARR_ASYNC__CONCURRENCY` environment variable (default: 10).

### Examples

```sh
# Serve synthetic global dataset on default port 8080
xpublish-tiles

# Serve air temperature tutorial dataset on port 9000
xpublish-tiles --port 9000 --dataset air

# Serve Arraylake dataset with specific branch and group
xpublish-tiles --dataset earthmover-public/aifs-outputs --branch main --group 2025-04-01/12z --cache

# Serve locally stored data created using `uv run pytest --setup`
uv run xpublish-tiles --dataset=local://ifs

# Run benchmark with a specific dataset
xpublish-tiles --dataset=local://para_hires --spy
```

## Benchmarking

The CLI includes a benchmarking feature that can be used to test tile server performance:

```sh
# Run benchmark with a specific dataset
xpublish-tiles --dataset=local://para_hires --spy
```

The `--spy` flag starts the tile server and automatically runs benchmark requests against it. The benchmarking:
- Warms up the server with initial tile requests
- Makes concurrent tile requests (limited to 12 at a time) to test performance
- Automatically exits after completing the benchmark run

For datasets containing "para" in the name, it uses zoom level 9 tiles. For other datasets, it uses lower zoom levels (0-1).

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

Environment variables
1. `XPUBLISH_TILES_ASYNC_LOAD: [0, 1]` - controls whether Xarray's async loading is used.
2. `XPUBLISH_TILES_NUM_THREADS: int` - controls the size of the threadpool
3. `XPUBLISH_TILES_TRANSFORM_CHUNK_SIZE: int` - when transforming coordinates, do so by submitting (NxN) chunks to the threadpool.
4. `XPUBLISH_TILES_LOG_LEVEL: str` - controls the logging level for xpublish_tiles (DEBUG, INFO, WARNING, ERROR). Default is WARNING. Note: numba, datashader, and PIL logs are always set to WARNING level to reduce noise.
