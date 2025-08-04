# Mapbox Example

Run the following command to start the server:
```
uv run xpublish-tiles --dataset=earthmover-public/aifs-outputs --group=2025-04-01/12z
```

Then you can try the examples:

* [Mapbox XYZ Tiles](./tiles.html) - Direct tile URL template
* [Mapbox WMS Tiles](./wms-tiled.html) - WMS GetMap requests
* [Mapbox TileJSON](./tilejson.html) - TileJSON 3.0.0 specification
* [Advanced TileJSON](./tilejson-advanced.html) - Interactive TileJSON with dynamic parameters

## TileJSON Examples

The TileJSON examples demonstrate how to use the new TileJSON 3.0.0 specification support:

### Basic TileJSON (`tilejson.html`)
- Uses TileJSON URL directly in the source configuration
- Mapbox GL JS automatically fetches and parses the TileJSON specification
- Uses the same parameters as the existing XYZ and WMS examples
- Demonstrates the simplest way to use TileJSON with Mapbox GL JS

### Advanced TileJSON (`tilejson-advanced.html`)
- Interactive controls for changing variables, styles, and parameters
- Dynamic TileJSON loading with different coordinate systems
- Real-time layer switching
- TileJSON specification viewer
- Demonstrates the full power of the TileJSON endpoint

Both examples use the same dataset and show the same weather data (2t variable) as the existing examples, making it easy to compare the different approaches.
