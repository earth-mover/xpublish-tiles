# xpublish-tiles

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
uv run pytest tests --where=local
```

## CLI Usage

The package includes a command-line interface for quickly serving datasets with tiles and WMS endpoints:

```sh
xpublish-tiles [OPTIONS]
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
```

Once running, the server provides:
- Tiles API at `http://localhost:8080/tiles/`
- WMS API at `http://localhost:8080/wms/`
- Interactive API documentation at `http://localhost:8080/docs`

## Examples

- [Mapbox Usage](./examples/mapbox/)
