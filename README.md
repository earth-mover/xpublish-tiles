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
uv run pytest tests
```
