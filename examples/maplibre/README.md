# MapLibre Examples

Run the following command to start the server:
```
uv run xpublish-tiles --dataset=earthmover-public/gfs --group=solar
```

Then you can try the examples:

* [MapLibre XYZ Tiles](./tiles.html)
* [MapLibre TileJSON](./tilejson.html)
* [MapLibre WMS Tiles](./wms-tiled.html)

## Vector Tile Examples

Serve the synthetic vector data cube of polygon features:
```
uv run xpublish-tiles --dataset=local://features_polygons
```

* [MapLibre Vector Tiles (MVT)](./mvt.html): styles each feature by the server's `color` property, or recolors it in the browser from its `value` without fetching new tiles.

Any GeoParquet file works too (`--dataset=geoparquet+file:///path/to/file.parquet`); pick the variable and color range with query parameters, e.g. `mvt.html?variables=pop_est&colorscalerange=0,1e9`.

## Categorical Examples

*More Instructions to Come*

* [MapLibre XYZ Tiles with Categorical Data](./tiles-categorical.html)

## Projected Examples

*More Instructions to Come*

* [MapLibre XYZ Tiles with Projected Data](./tiles-projected.html)

### Native EPSG:3035 WMS Image

Run the server with the synthetic EU LAEA dataset:
```
uv run xpublish-tiles --dataset=local://eu3035
```

* [MapLibre WMS Single Image (native EPSG:3035)](./wms-image.html)
