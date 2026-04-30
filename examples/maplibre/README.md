# MapLibre Examples

Run the following command to start the server:
```
uv run xpublish-tiles --dataset=earthmover-public/gfs --group=solar
```

Then you can try the examples:

* [MapLibre XYZ Tiles](./tiles.html)
* [MapLibre TileJSON](./tilejson.html)
* [MapLibre WMS Tiles](./wms-tiled.html)
* [MapLibre Vector Tiles (MVT) — cells](./tiles-vector.html) — `vector/cells`: one polygon feature per grid cell, value as a typed property. Defaults to GFS `temperature_2m` (`uv run xpublish-tiles --dataset=earthmover-public/noaa-gfs-forecast`, requires Arraylake credentials).
* [MapLibre Vector Tiles (MVT) — wind points + pressure contours](./tiles-vector-points.html) — `vector/points` arrows for `wind_u_10m` + `wind_v_10m` over a `vector/contours` backdrop of mean sea-level pressure. Filled contour bands plus isolines (drawn as polygon outlines on the same MVT source) — three layers from two server tile sources, composited client-side. Same dataset / server invocation as above.

## Categorical Examples

*More Instructions to Come*

* [MapLibre XYZ Tiles with Categorical Data](./tiles-categorical.html)

## Projected Examples

*More Instructions to Come*

* [MapLibre XYZ Tiles with Projected Data](./tiles-projected.html)
