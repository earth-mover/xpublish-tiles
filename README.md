# xpublish-tiles

![PyPI - Version](https://img.shields.io/pypi/v/xpublish-tiles)
![GitHub Actions](https://github.com/earth-mover/xpublish-tiles/actions/workflows/test.yml/badge.svg)
![Codecov](https://codecov.io/gh/earth-mover/xpublish-tiles/branch/main/graph/badge.svg)
[![Xarray](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydata/xarray/refs/heads/main/doc/badge.json)](https://xarray.dev)

Web mapping plugins for [Xpublish](https://github.com/xpublish-community/xpublish)

## Project Overview
This project contains a set of web mapping plugins for Xpublish - a framework for serving xarray datasets via HTTP APIs.

The goal of this project is to transform xarray datasets to raster, vector and other types of tiles, which can then be served via HTTP APIs. To do this, the package implements a set of xpublish plugins:
* `xpublish_tiles.xpublish.tiles.TilesPlugin`: An [OGC Tiles](https://www.ogc.org/standards/ogcapi-tiles/) conformant plugin for serving raster, vector and other types of tiles.
* `xpublish_tiles.xpublish.wms.WMSPlugin`: An [OGC Web Map Service](https://www.ogc.org/standards/wms/) conformant plugin for serving raster, vector and other types of tiles.

> [!NOTE]
> The `TilesPlugin` is feature complete, but the `WMSPlugin` is still in active development.

## Features

### Extensive grid support

`xpublish-tiles` supports handling a wide variety of grids including:
1. Raster grids specified using an Affine transform specified in the `GeoTransform` attribute of the grid mapping variable (`spatial_ref`)
2. Rectilinear grids specified using two 1D orthogonal coordinates `lat[lat], lon[lon]`.
3. Curvilinear grids specified using two 2D coordinates `lat[nlat, nlon], lon[nlat, nlon]`.
4. Unstructured grids specified using two 1D coordinates, interpreted as vertices and triangulated using `scipy.spatial.Delaunay` : `lat[point], lon[point]`.

Here `lat[lat]` means a coordinate variable named `lat` with one dimension named `lat`.

> [!NOTE]
> The library is built to be extensible, and could easily accommodate more grid definitions. Contributions welcome!

We attempt to require as little metadata as possible, and attempts to infer as much as possible. However, it is *always* better
for you to annotate your dataset using the CF & ACDD conventions as well as possible.

### Categorical Data support

By default all data is treated as continuous. Discrete data are assumed to be encoded with the CF flag variable convention i.e., arrays with the `flag_values` and `flag_meanings`
attributes are treated as discrete categorical data by the rendering pipeline.

### Custom Colormaps

> [!IMPORTANT]
> At the moment RGBA colors are not supported in colormaps because of this [upstream datashader issue](https://github.com/holoviz/datashader/issues/1404).

Custom colormaps can be provided using the `colormap` parameter. When using a custom colormap, you must set `style=raster/custom`.

**Continuous data**

The colormap is a JSON-encoded dictionary with:
- **Keys**: String integers from "0" to "255" (not data values)
- **Values**: Hex color codes in the format `#RRGGBB`

> [!IMPORTANT]
> Custom colormaps for continuous data must include both "0" and "255" as keys. These colormaps must have keys that are "0" and "255", not data values. The data value is rescaled by `colorscalerange` to 0→1; the colormap is rescaled from 0→255 to 0→1 and then applied to the scaled 0→1 data.

**Categorical data**

The colormap is a JSON-encoded dictionary with:
- **Keys**: Data values that match the values of the `flag_values` attribute of the array.
- **Values**: Hex color codes in the format `#RRGGBB`

Alternatively the `flag_colors` attribute can be set on the array. Its value must be a string containing space delimited hex colors of the same length
as the corresponding `flag_meanings` and `flag_values` attributes. For example

```
land_cover:flag_values = 1, 2, 3, 4, 5, 6;
land_cover:flag_meanings = "Broadleaf_Woodland Coniferous_Woodland Arable_and_Horticulture Improved_Grassland Rough_Grassland Neutral_Grassland" ;
land_cover:flag_colors = "#FF0000 #006600 #732600 #00FF00 #FAAA00 #7FE57F" ;
```

See the [ncWMS convention docs on Categorical Data](https://web.archive.org/web/20240729161558/https://reading-escience-centre.gitbooks.io/ncwms-user-guide/content/05-data_formats.html#vector) for more.

### Dimension selection with methods

`xpublish-tiles` supports flexible dimension selection using a DSL that allows you to specify selection methods. This is particularly useful for temporal and vertical coordinates where you may want to select the nearest value, or use forward/backward fill.

**Syntax:** `dimension=method::value`

**Supported methods:**
- `nearest` - Select the nearest coordinate value
- `pad` / `ffill` - Forward fill (use the previous valid value)
- `backfill` / `bfill` - Backward fill (use the next valid value)
- `exact` - Exact match (also the default when no method is specified)

**Examples:**

```bash
# Select nearest time to 2000-01-01T04:00
http://localhost:8080/tiles/WebMercatorQuad/4/4/14?variables=temperature&time=nearest::2000-01-01T04:00

# Exact match (implicit)
http://localhost:8080/tiles/WebMercatorQuad/4/4/14?variables=temperature&time=2000-01-01T00:00

# Forward fill for missing timestep
http://localhost:8080/tiles/WebMercatorQuad/4/4/14?variables=temperature&time=ffill::2000-01-01T03:30

# Multiple dimension selections with different methods
http://localhost:8080/tiles/WebMercatorQuad/4/4/14?time=nearest::2000-01-01T04:00&pressure_level=500

# Using timedelta selections
http://localhost:8080/tiles/WebMercatorQuad/4/4/14?variables=temperature&time=nearest::2000-01-01T04:00&step=pad::3h
```

**Key features:**
- Uses `::` separator to avoid ambiguity with datetime colons (e.g., `2000-01-01T12:00:00`)
- Case-insensitive method names
- Works with any dimension type (temporal, vertical, or custom)


### Automatic dimension reduction

Since each tile can only take a 2D DataArray as input, if enough selectors (or indexers; e.g. `step=1h`) are not provided `xpublish-tiles` will index out the last location along each dimension that is not X, Y. Along the "vertical" dimension we index out coordinate location 0. It is recommended that you apply as many selectors as necessary explicitly.

## Integration Examples

- [Maplibre/Mapbox Usage](./examples/maplibre/)

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
  - `zarr:///path/to/zarr/store`: Load standard Zarr store (use `--group` for nested groups)
  - `icechunk:///path/to/repo`: Load Icechunk repository (use `--group` for groups, `--branch` for branches)
  - `local://<dataset_name>`: Convenience alias for `icechunk:///tmp/tiles-icechunk --group <dataset_name>` (datasets created with `uv run pytest --setup`)
  - For Arraylake datasets: specify the dataset name in {arraylake_org}/{arraylake_dataset} format (requires Arraylake credentials)
- `--branch BRANCH`: Branch to use for Arraylake, Icechunk, or local datasets (default: main)
- `--group GROUP`: Group to use for Arraylake, Zarr, or Icechunk datasets (default: '')
- `--cache`: Enable icechunk cache for Arraylake and local icechunk datasets (default: enabled)
- `--spy`: Run benchmark requests with the specified dataset for performance testing
- `--bench-suite`: Run benchmarks for all local datasets and tabulate results (requires `uv run pytest --setup` to create local datasets first)
- `--concurrency INT`: Number of concurrent requests for benchmarking (default: 12)
- `--where CHOICE`: Where to run benchmark requests (choices: local, local-booth, arraylake-prod, arraylake-dev; default: local)
  - `local`: Start server on localhost and run benchmarks against it
  - `local-booth`: Run benchmarks against existing localhost server (no server startup)
  - `arraylake-prod`: Run benchmarks against Arraylake production server (earthmover.io)
  - `arraylake-dev`: Run benchmarks against Arraylake development server (earthmover.dev)
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

# Serve icechunk data from custom path
xpublish-tiles --dataset icechunk:///path/to/my/repo --group my_dataset

# Serve standard Zarr store
xpublish-tiles --dataset zarr:///path/to/data.zarr

# Serve Zarr store with a specific group
xpublish-tiles --dataset zarr:///path/to/data.zarr --group subgroup

# Serve Icechunk repository
xpublish-tiles --dataset icechunk:///path/to/icechunk/repo --group my_dataset

# Serve Arraylake dataset with specific branch and group
xpublish-tiles --dataset earthmover-public/aifs-outputs --branch main --group 2025-04-01/12z

# Run benchmark with a specific dataset
xpublish-tiles --dataset local://para_hires --spy

# Run benchmark with custom concurrency and against Arraylake production
xpublish-tiles --dataset para --spy --concurrency 20 --where arraylake-prod

# Run benchmark suite for all local datasets (creates tabulated results)
xpublish-tiles --bench-suite

# Run benchmark suite for all local datasets and compare with titiler
xpublish-tiles --bench-suite --titiler

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

# Run benchmark against Arraylake production server with custom concurrency
xpublish-tiles --dataset para --spy --where arraylake-prod --concurrency 8

# Run benchmark suite for all local datasets
xpublish-tiles --bench-suite
```

### Benchmark Suite

The `--bench-suite` option runs performance tests on all available local datasets and creates a tabulated summary of results. This is useful for comparing performance across different dataset types and configurations.

**Prerequisites**: You must first create the local test datasets:
```sh
uv run pytest --setup
```

The benchmark suite will test the following local datasets:
- `ifs`: Integrated Forecasting System dataset
- `hrrr`: High-Resolution Rapid Refresh dataset
- `para_hires`: High-resolution parameterized dataset
- `eu3035_hires`: High-resolution European dataset
- `utm50s_hires`: High-resolution UTM Zone 50S dataset
- `sentinel`: Sentinel-2 dataset
- `global-6km`: Global dataset at 6km resolution

The output includes a performance table showing tiles processed, success/failure rates, wall time, average request time, and requests per second for each dataset.

### Individual Benchmarking

The `--spy` flag enables benchmarking mode. The benchmarking behavior depends on the `--where` option:

- **`--where local`** (default): Starts the tile server and automatically runs benchmark requests against it
- **`--where local-booth`**: Runs benchmarks against an existing localhost server (doesn't start a new server)
- **`--where arraylake-prod`**: Runs benchmarks against Arraylake production server (earthmover.io)
- **`--where arraylake-dev`**: Runs benchmarks against Arraylake development server (earthmover.dev)

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

## Deployment notes

1. Make sure to limit `NUMBA_NUM_THREADS`; this is used for rendering categorical data with datashader.
2. The first invocation of a render will block while datashader functions are JIT-compiled. Our attempts to add a precompilation step to remove this have been unsuccessful.

### Configuration
Settings can be configured via environment variables or config files. The async loading setting has been moved to the config system (use `async_load` in config files or `XPUBLISH_TILES_ASYNC_LOAD` environment variable).
1. `XPUBLISH_TILES_NUM_THREADS: int` - controls the size of the threadpool
2. `XPUBLISH_TILES_ASYNC_LOAD: bool` - whether to use Xarray's async loading
3. `XPUBLISH_TILES_TRANSFORM_CHUNK_SIZE: int` - when transforming coordinates, do so by submitting (NxN) chunks to the threadpool.
4. `XPUBLISH_TILES_DETECT_APPROX_RECTILINEAR: bool` - detect whether a curvilinear grid is approximately rectilinear
5. `XPUBLISH_TILES_RECTILINEAR_CHECK_MIN_SIZE: int` - check for rectilinearity if array.shape > (N, N)
6. `XPUBLISH_TILES_MAX_RENDERABLE_SIZE: int` - do not attempt to load or render arrays with size greater than this value
7. `XPUBLISH_TILES_DEFAULT_PAD: int` - how much to pad a selection on either side
8. `XPUBLISH_TILES_GRID_CACHE_MAX_SIZE: int` - maximum number of grid systems to cache (default: 16). **Note:** This must be set via environment variable before importing the module, as the cache is initialized at import time.

## Performance Notes

For context, the rendering pipeline is:
1. Receive dataset `ds` and `QueryParams` from the plugin.
2. Grab `GridSystem` for `ds` and requested DataArray. The inference here is complex and is cached internally using the `ds.attrs['_xpublish_id']` and the requested `DataArray.name`. *Be sure to set this attribute to a unique string.*
3. Based on the grid system, the data are subset to the bounding box using slices. For datasets with a geographic CRS, padding is applied to the slicers if needed to account for the meridian or anti-meridian and depending on the dataset's longitude convention (0→360 or -180→180).
4. This plugin supports parsing multiple "grid mappings" for a single DataArray. If present, we pick coordinates corresponding to the output CRS. If not, we look to see if there are coordinates corresponding to `epsg:4326`, if not, we use the native coordinates.
5. Coordinates are transformed to the output CRS, if needed. This is usually a very slow step. For performance,
   a. We reimplement the `epsg:4326 -> epsg:3857` transformation because it is separable (`x` is fully determined by `longitude`, and `y` is fully determined by latitude). This allows us to preserve the regular or rectilinear nature of the grid if possible.
   b. If (a) is not possible, we broadcast the input coordinates against each other, then cut up the coordinates in to chunks and process them in a threadpool using `pyproj`.
4. Xarray's new `load_async` is used to load the data in to memory.
5. Next we check whether the grid, if curvilinear, may be approximated by a rectilinear grid.
   a. The Rectilinear mesh codepath is datashader can be 3-10X faster than the Curvilinear codepath, so this approximation is worth it.
   b. We replicate the logic in datashader that constructs an array that contains output pixel id for each each input pixel -- this is done for each axis.
   c. If the difference between these arrays, constructed from the curvilinear and rectilinear meshes, differs by one pixel, then we approximate the grid as rectilinear. This threshold is pretty tight, and requires some experimentation to loosen further. If loosening, we will need to pad appropriately.
   d. Realistically this optimization is triggered on high resolution data at zoom levels where the grid distortion isn't very high.


### Performance recommendations:
1. Make sure `_xpublish_id` is set in `Dataset.attrs`.
2. If CRS transformations are a bottleneck,
   1. Assign reprojected coordinates for the desired output CRS using multiple grid mapping variables. This will take reprojection time down to 0.
   1. See if you can approximate the coordinate system with rectilinear coordinates as much as possible. This triggers a much faster rendering pathway in datashader.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details
