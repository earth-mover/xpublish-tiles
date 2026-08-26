# MapLibre Examples

Run the following command to start the server:
```
uv run xpublish-tiles --dataset=earthmover-public/gfs --group=solar
```

Then you can try the examples:

* [MapLibre XYZ Tiles](./tiles.html)
* [MapLibre TileJSON](./tilejson.html)
* [MapLibre WMS Tiles](./wms-tiled.html)

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
