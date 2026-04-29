# MapLibre Examples

Run the following command to start the server:
```
uv run xpublish-tiles --dataset=earthmover-public/gfs --group=solar
```

Then you can try the examples:

* [MapLibre XYZ Tiles](./tiles.html)
* [MapLibre TileJSON](./tilejson.html)
* [MapLibre WMS Tiles](./wms-tiled.html)
* [MapLibre Vector Tiles (MVT)](./tiles-vector.html) — defaults to the GFS forecast dataset (`uv run xpublish-tiles --dataset=earthmover-public/noaa-gfs-forecast`, requires Arraylake credentials); edit `VARIABLE` / `INITIAL_MIN` / `INITIAL_MAX` / `DOMAIN_*` at the top of the script to point at any other dataset.

## Categorical Examples

*More Instructions to Come*

* [MapLibre XYZ Tiles with Categorical Data](./tiles-categorical.html)

## Projected Examples

*More Instructions to Come*

* [MapLibre XYZ Tiles with Projected Data](./tiles-projected.html)
