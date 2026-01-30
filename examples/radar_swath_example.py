"""
Example: Serving radar data tiles using SwathGrid and pyresample.

This example demonstrates how to use the SwathGrid class to efficiently
resample radar data to map tiles.

Prerequisites:
    pip install pyresample xradar

Usage:
    python examples/radar_swath_example.py
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyproj.aoi import BBox

# Import the SwathGrid
from xpublish_tiles.grids_swath import SwathGrid, is_swath_data


def create_synthetic_radar_data():
    """Create synthetic radar-like data for testing."""
    import xarray as xr

    # Radar parameters
    n_azimuth = 360
    n_range = 500
    max_range = 250000  # 250 km

    # Radar location (example: somewhere in the US Midwest)
    radar_lat = 41.6
    radar_lon = -88.1

    # Create azimuth and range coordinates
    azimuth = np.linspace(0.5, 359.5, n_azimuth)
    range_m = np.linspace(500, max_range, n_range)

    # Compute lat/lon for each gate (simplified, assumes flat earth for demo)
    az_rad = np.radians(azimuth)

    # Create 2D meshgrid
    az_2d, range_2d = np.meshgrid(az_rad, range_m, indexing="ij")

    # Convert to approximate lat/lon (simplified projection)
    # In reality, you'd use proper geodetic calculations
    meters_per_degree_lat = 111320
    meters_per_degree_lon = 111320 * np.cos(np.radians(radar_lat))

    dx = range_2d * np.sin(az_2d)
    dy = range_2d * np.cos(az_2d)

    lon_2d = radar_lon + dx / meters_per_degree_lon
    lat_2d = radar_lat + dy / meters_per_degree_lat

    # Create synthetic reflectivity data (some "weather" patterns)
    # Simulate a storm cell
    storm_az = 45  # degrees
    storm_range = 100000  # meters
    storm_width = 30000

    az_diff = np.abs(azimuth[:, np.newaxis] - storm_az)
    az_diff = np.minimum(az_diff, 360 - az_diff)
    range_diff = np.abs(range_m[np.newaxis, :] - storm_range)

    # Gaussian-like storm pattern
    dbzh = 50 * np.exp(-((az_diff / 20) ** 2 + (range_diff / storm_width) ** 2))
    dbzh += np.random.normal(0, 2, dbzh.shape)  # Add noise
    dbzh = np.clip(dbzh, -10, 70)

    # Add some clear air return near radar
    clear_air = 10 * np.exp(-((range_m[np.newaxis, :] / 50000) ** 2))
    dbzh = np.maximum(dbzh, clear_air - 5)

    # Create dataset
    ds = xr.Dataset(
        data_vars={
            "DBZH": (["azimuth", "range"], dbzh.astype(np.float32)),
        },
        coords={
            "azimuth": ("azimuth", azimuth),
            "range": ("range", range_m),
            "lat": (["azimuth", "range"], lat_2d),
            "lon": (["azimuth", "range"], lon_2d),
            "latitude": radar_lat,
            "longitude": radar_lon,
        },
    )
    ds["DBZH"].attrs = {
        "long_name": "Equivalent reflectivity factor H",
        "units": "dBZ",
        "standard_name": "radar_equivalent_reflectivity_factor_h",
    }

    return ds


def render_tile_to_image(
    tile_data: np.ndarray,
    colormap: str = "pyart_NWSRef",
    vmin: float = -10,
    vmax: float = 60,
) -> Image.Image:
    """Convert tile data to a colored PIL Image."""
    # Normalize data to 0-1 range
    normalized = (tile_data - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)

    # Handle NaN values
    mask = np.isnan(tile_data)

    # Get colormap
    try:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(colormap)
    except ValueError:
        cmap = plt.get_cmap("viridis")

    # Apply colormap
    rgba = cmap(normalized)

    # Set NaN pixels to transparent
    rgba[mask] = [0, 0, 0, 0]

    # Convert to uint8
    rgba_uint8 = (rgba * 255).astype(np.uint8)

    return Image.fromarray(rgba_uint8, mode="RGBA")


def main():
    print("=" * 70)
    print("Radar SwathGrid Example")
    print("=" * 70)

    # Create or load radar data
    print("\n1. Creating synthetic radar data...")
    ds = create_synthetic_radar_data()
    print(f"   Dataset dimensions: {dict(ds.dims)}")
    print(f"   Lat range: {float(ds.lat.min()):.2f} to {float(ds.lat.max()):.2f}")
    print(f"   Lon range: {float(ds.lon.min()):.2f} to {float(ds.lon.max()):.2f}")

    # Check if it's detected as swath data
    print("\n2. Detecting swath data...")
    is_swath = is_swath_data(ds)
    print(f"   Is swath data: {is_swath}")

    # Create SwathGrid
    print("\n3. Creating SwathGrid...")
    start = time.time()
    grid = SwathGrid.from_dataset(ds, lat_name="lat", lon_name="lon")
    setup_time = time.time() - start
    print(f"   Setup time: {setup_time * 1000:.1f} ms")
    print(f"   Grid bbox: {grid.bbox}")
    print(f"   Grid dims: {grid.dims}")

    # Define a tile bbox (subset of radar coverage)
    radar_lat = float(ds.latitude)
    radar_lon = float(ds.longitude)
    tile_bbox = BBox(
        west=radar_lon - 0.5,
        south=radar_lat - 0.5,
        east=radar_lon + 0.5,
        north=radar_lat + 0.5,
    )

    # Resample to tile (first request - includes neighbour info computation)
    print("\n4. First tile request (includes KDTree setup)...")
    data = ds["DBZH"]

    start = time.time()
    tile_data = grid.resample_to_tile(data, tile_bbox, width=256, height=256)
    first_request_time = time.time() - start
    print(f"   First request time: {first_request_time * 1000:.1f} ms")
    print(f"   Tile shape: {tile_data.shape}")
    print(f"   Value range: {np.nanmin(tile_data):.1f} to {np.nanmax(tile_data):.1f}")

    # Second request (cached - should be fast)
    print("\n5. Cached tile request...")
    start = time.time()
    _ = grid.resample_to_tile(data, tile_bbox, width=256, height=256)
    cached_time = time.time() - start
    print(f"   Cached request time: {cached_time * 1000:.1f} ms")
    print(f"   Speedup: {first_request_time / cached_time:.1f}x")

    # Different tile (new bbox - needs new neighbour info)
    print("\n6. Different tile bbox...")
    tile_bbox_2 = BBox(
        west=radar_lon,
        south=radar_lat,
        east=radar_lon + 1.0,
        north=radar_lat + 1.0,
    )
    start = time.time()
    _ = grid.resample_to_tile(data, tile_bbox_2, width=256, height=256)
    new_bbox_time = time.time() - start
    print(f"   New bbox request time: {new_bbox_time * 1000:.1f} ms")

    # Full radar extent
    print("\n7. Full radar extent (512x512)...")
    start = time.time()
    full_tile = grid.resample_to_tile(data, grid.bbox, width=512, height=512)
    full_time = time.time() - start
    print(f"   Full extent time: {full_time * 1000:.1f} ms")

    # Render to image
    print("\n8. Rendering tile to image...")
    start = time.time()
    _ = render_tile_to_image(tile_data, vmin=-10, vmax=60)
    render_time = time.time() - start
    print(f"   Render time: {render_time * 1000:.1f} ms")

    # Summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"  Grid setup:           {setup_time * 1000:>8.1f} ms")
    print(f"  First tile (w/ KDTree): {first_request_time * 1000:>8.1f} ms")
    print(f"  Cached tile:          {cached_time * 1000:>8.1f} ms")
    print(f"  Image render:         {render_time * 1000:>8.1f} ms")
    print("  ─────────────────────────────────")
    print(f"  Total (cached):       {(cached_time + render_time) * 1000:>8.1f} ms")
    print("=" * 70)

    # Visualize
    print("\n9. Creating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original radar in polar coords
    ax = axes[0]
    ds["DBZH"].plot(ax=ax, cmap="viridis", vmin=-10, vmax=60)
    ax.set_title("Original (azimuth × range)")

    # Tile near radar
    ax = axes[1]
    ax.imshow(
        tile_data,
        origin="lower",
        cmap="viridis",
        vmin=-10,
        vmax=60,
        extent=[tile_bbox.west, tile_bbox.east, tile_bbox.south, tile_bbox.north],
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Tile (256×256) - {cached_time * 1000:.1f}ms")

    # Full extent
    ax = axes[2]
    im = ax.imshow(
        full_tile,
        origin="lower",
        cmap="viridis",
        vmin=-10,
        vmax=60,
        extent=[grid.bbox.west, grid.bbox.east, grid.bbox.south, grid.bbox.north],
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Full Extent (512×512) - {full_time * 1000:.1f}ms")
    plt.colorbar(im, ax=ax, label="DBZH (dBZ)")

    plt.tight_layout()
    plt.savefig("radar_swath_example.png", dpi=150)
    print("   Saved: radar_swath_example.png")
    plt.show()


if __name__ == "__main__":
    main()
