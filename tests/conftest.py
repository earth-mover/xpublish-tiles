import io
import os
import re
from itertools import product

import arraylake as al
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image
from pyproj.aoi import BBox
from syrupy.extensions.image import PNGImageSnapshotExtension

import icechunk
import xarray as xr
from tests.tiles import ETRS89_TILES, HRRR_TILES
from xpublish_tiles.datasets import EU3035, HRRR, create_global_dataset

IS_SNAPSHOT_UPDATE = False


def pytest_addoption(parser):
    parser.addoption(
        "--where",
        action="store",
        choices=["local", "arraylake", "arraylake-dev"],
        default="local",
        help="Storage backend: 'local' for local filesystem, 'arraylake' for Arraylake prod deployment, or 'arraylake-dev' for Arraylake dev deployment (default: local)",
    )
    parser.addoption(
        "--prefix",
        action="store",
        help="Prefix for the repository/storage path (defaults: local=/tmp/tiles-icechunk/, arraylake=earthmover-integration/tiles-icechunk/, arraylake-dev=earthmover-integration/tiles-icechunk/)",
    )
    parser.addoption(
        "--setup",
        action="store",
        nargs="?",
        const="",
        default=None,
        help="Run setup tests (test_create). Use --setup=force to always recreate datasets.",
    )
    parser.addoption(
        "--debug-visual",
        action="store_true",
        help="Show visual difference plots in matplotlib window when PNG snapshots don't match (automatically disables parallelization)",
    )
    parser.addoption(
        "--debug-visual-save",
        action="store_true",
        help="Save visual difference plots to PNG files and auto-open them (automatically disables parallelization)",
    )
    parser.addoption(
        "--visualize",
        action="store_true",
        help="Show matplotlib visualization windows during tests",
    )


def pytest_configure(config):
    """Configure pytest settings based on command line options."""
    # Disable parallelization when debug visual options are used
    if config.getoption("--debug-visual") or config.getoption("--debug-visual-save"):
        # Check if pytest-xdist is being used and disable it
        if hasattr(config.option, "numprocesses") and config.option.numprocesses != 0:
            config.option.numprocesses = 0
            print(
                "🔍 Debug visual mode enabled - disabling parallel execution for better visualization"
            )

        # Also disable dist mode completely
        if hasattr(config.option, "dist") and config.option.dist:
            config.option.dist = "no"


@pytest.fixture(scope="session")
def air_dataset():
    ds = xr.tutorial.load_dataset("air_temperature")
    ds.air.attrs["valid_min"] = 271
    ds.air.attrs["valid_max"] = 317.4
    return ds


@pytest.fixture(scope="session")
def where(request):
    return request.config.getoption("--where")


@pytest.fixture(scope="session")
def prefix(request, where):
    provided_prefix = request.config.getoption("--prefix")
    if provided_prefix:
        return provided_prefix

    # Use defaults based on storage backend
    if where == "local":
        return "/tmp/tiles-icechunk/"
    elif where == "arraylake":
        return "earthmover-integration/tiles-icechunk"
    elif where == "arraylake-dev":
        return "earthmover-integration/tiles-icechunk"
    else:
        raise ValueError(f"No default prefix available for storage backend: {where}")


def generate_repo(where: str, prefix: str):
    """Generate an icechunk Repository based on storage backend choice.

    Args:
        where: Storage backend - 'local', 'arraylake', or 'arraylake-dev'
        prefix: Prefix for the repository/storage path

    Returns:
        icechunk.Repository: Repository object for the specified backend
    """
    if where == "local":
        storage = icechunk.local_filesystem_storage(prefix)
        try:
            # Try to open existing repository
            return icechunk.Repository.open(storage)
        except Exception:
            # Create new repository if it doesn't exist
            return icechunk.Repository.create(storage)
    elif where == "arraylake":
        client = al.Client()
        repo = client.get_or_create_repo(prefix)
        return repo
    elif where == "arraylake-dev":
        client = al.Client(
            service_uri="https://dev.api.earthmover.io",
            token=os.environ.get("ARRAYLAKE_TOKEN"),
        )
        repo = client.get_or_create_repo(prefix, bucket_config_nickname="earthmover")
        return repo
    else:
        raise ValueError(f"Unsupported storage backend: {where}")


@pytest.fixture(scope="session")
def setup_option(request):
    """Get the --setup option value for the session."""
    return request.config.getoption("--setup")


@pytest.fixture(scope="session")
def session_repo(where, prefix, setup_option):
    """Create repository once per session for force-create scenarios."""
    if setup_option == "force":
        # For force create, ensure we have a clean repo
        if where == "local":
            import os
            import shutil

            # Clean up existing repo for force-create
            try:
                if os.path.exists(prefix):
                    shutil.rmtree(prefix, ignore_errors=True)
            except Exception:
                # If cleanup fails, continue anyway
                pass

    return generate_repo(where, prefix)


@pytest.fixture
def repo(session_repo):
    """Return the session repo for individual tests."""
    return session_repo


@pytest.fixture(
    params=tuple(map(",".join, product(["-90->90", "90->-90"], ["-180->180", "0->360"])))
)
def global_datasets(request):
    param = request.param

    # Parse parameters to determine coordinate ordering
    lat_ascending = "-90->90" in param
    lon_0_360 = "0->360" in param

    yield create_global_dataset(lat_ascending=lat_ascending, lon_0_360=lon_0_360)


# Create the product of datasets and their appropriate tiles
def _get_projected_dataset_tile_params():
    params = []
    for dataset_class, tiles in [
        (EU3035, ETRS89_TILES),
        (HRRR, HRRR_TILES),
    ]:
        for tile_param in tiles:
            tile, tms = tile_param.values
            param_id = f"{dataset_class.name}_{tile_param.id}"
            params.append(pytest.param((dataset_class, tile, tms), id=param_id))
    return params


@pytest.fixture(params=_get_projected_dataset_tile_params())
def projected_dataset_and_tile(request):
    dataset_class, tile, tms = request.param
    ds = dataset_class.create()

    # Validate that tile overlaps with dataset bounding box
    dataset_bbox = ds.attrs["bbox"]
    tile_bounds = tms.bounds(tile)
    tile_bbox = BBox(
        west=tile_bounds.left,
        south=tile_bounds.bottom,
        east=tile_bounds.right,
        north=tile_bounds.top,
    )

    # Check if dataset bbox intersects with tile bounds
    if not dataset_bbox.intersects(tile_bbox):
        pytest.skip(f"Tile {tile} does not overlap with dataset bbox {dataset_bbox}")

    return (ds, tile, tms)


@pytest.fixture
def png_snapshot(snapshot, pytestconfig, request):
    """PNG snapshot with custom numpy array comparison and optional debug visualization."""

    IS_SNAPSHOT_UPDATE = pytestconfig.getoption("--snapshot-update", default=False)
    DEBUG_VISUAL = pytestconfig.getoption("--debug-visual", default=False)
    DEBUG_VISUAL_SAVE = pytestconfig.getoption("--debug-visual-save", default=False)

    def create_debug_visualization(
        actual_array, expected_array, test_name, tile_info=None
    ):
        """Create a 3-panel debug visualization: Expected | Actual | Differences"""

        def extract_tile_info(test_name, tile_info):
            """Extract tile coordinates and TMS info from tile parameter."""
            # Use tile and tms info directly from test parameters
            tile, tms = tile_info
            # Extract coordinate system info from test name
            coord_pattern = r"\[([-\d>]+,[-\d>]+)-"
            coord_match = re.search(coord_pattern, test_name)
            coord_info = coord_match.group(1) if coord_match else "unknown"

            return {
                "tms_name": tms.id,
                "z": tile.z,
                "x": tile.x,
                "y": tile.y,
                "coord_info": coord_info,
                "tms": tms,
            }

        # Create difference map
        def create_difference_map(expected, actual):
            # Calculate absolute differences for RGB channels (ignore alpha)
            diff_rgb = np.abs(
                expected[:, :, :3].astype(np.float32)
                - actual[:, :, :3].astype(np.float32)
            )

            # Calculate magnitude of difference (L2 norm across RGB channels)
            diff_magnitude = np.sqrt(np.sum(diff_rgb**2, axis=2))

            # Normalize to 0-255 range for visualization
            if diff_magnitude.max() > 0:
                diff_normalized = (diff_magnitude / diff_magnitude.max() * 255).astype(
                    np.uint8
                )
            else:
                diff_normalized = np.zeros_like(diff_magnitude, dtype=np.uint8)

            # Create a heatmap: black = no difference, red = maximum difference
            diff_map = np.zeros((*diff_normalized.shape, 4), dtype=np.uint8)
            diff_map[:, :, 0] = diff_normalized  # Red channel
            diff_map[:, :, 3] = 255  # Full alpha

            return diff_map

        # Extract tile information and calculate bbox
        extracted_tile_info = extract_tile_info(test_name, tile_info)
        bbox_info = ""
        try:
            # Use TMS directly from the extracted tile info
            tms = extracted_tile_info["tms"]

            from morecantile import Tile

            tile = Tile(
                x=extracted_tile_info["x"],
                y=extracted_tile_info["y"],
                z=extracted_tile_info["z"],
            )
            xy_bounds = tms.xy_bounds(tile)
            geo_bounds = tms.bounds(tile)

            bbox_info = f"""Tile Information:
Tile: z={extracted_tile_info['z']}, x={extracted_tile_info['x']}, y={extracted_tile_info['y']} ({extracted_tile_info['tms_name']})
Coordinate System: {extracted_tile_info['coord_info']}

Geographic Bounds (WGS84):
West: {geo_bounds.west:.3f}°, East: {geo_bounds.east:.3f}°
South: {geo_bounds.south:.3f}°, North: {geo_bounds.north:.3f}°

Projected Bounds ({tms.crs}):
X: {xy_bounds[0]:.0f} to {xy_bounds[2]:.0f}
Y: {xy_bounds[1]:.0f} to {xy_bounds[3]:.0f}

"""
        except Exception as e:
            bbox_info = f"Tile: z={extracted_tile_info['z']}, x={extracted_tile_info['x']}, y={extracted_tile_info['y']}\nBounds calculation failed: {e}\n\n"

        # Calculate difference statistics
        expected_transparent = np.sum(expected_array[:, :, 3] == 0)
        actual_transparent = np.sum(actual_array[:, :, 3] == 0)

        diff_pixels = np.sum(np.any(expected_array != actual_array, axis=2))
        total_pixels = expected_array.shape[0] * expected_array.shape[1]
        diff_pct = (diff_pixels / total_pixels) * 100

        rgb_diff = np.abs(
            expected_array[:, :, :3].astype(np.float32)
            - actual_array[:, :, :3].astype(np.float32)
        )
        max_diff = rgb_diff.max()
        mean_diff = rgb_diff[rgb_diff > 0].mean() if np.any(rgb_diff > 0) else 0

        # Create 3-panel visualization: Expected | Actual | Differences
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1: Expected output (snapshot)
        axes[0].imshow(expected_array)
        axes[0].set_title(f"Expected (Snapshot)\n{test_name}")
        axes[0].axis("off")

        # Panel 2: Actual output
        axes[1].imshow(actual_array)
        axes[1].set_title(f"Actual (Current)\n{test_name}")
        axes[1].axis("off")

        # Panel 3: Difference map
        diff_map = create_difference_map(expected_array, actual_array)
        axes[2].imshow(diff_map)
        axes[2].set_title("Differences\n(Black=Same, Red=Different)")
        axes[2].axis("off")

        # Add difference statistics as text
        diff_text = f"""{bbox_info}Difference Statistics:

Different pixels: {diff_pixels:,} / {total_pixels:,}
Percentage different: {diff_pct:.3f}%

Max RGB difference: {max_diff:.1f} / 255
Mean RGB difference: {mean_diff:.1f} / 255

Transparency Comparison:
Expected: {expected_transparent:,} transparent pixels
Actual: {actual_transparent:,} transparent pixels
Change: {actual_transparent - expected_transparent:+,} pixels

{'✓ Visual differences are minimal' if diff_pct < 0.5 else '⚠ Noticeable visual differences'}
"""

        # Add text box with statistics
        fig.text(
            0.02,
            0.02,
            diff_text,
            fontfamily="monospace",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        )

        plt.tight_layout()

        if DEBUG_VISUAL_SAVE:
            # Save visualization
            debug_path = f"debug_visual_diff_{test_name.replace('/', '_').replace('[', '_').replace(']', '_')}.png"
            plt.savefig(debug_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"\n🔍 Debug visualization saved to: {debug_path}")
            print(
                f"   Different pixels: {diff_pixels:,} / {total_pixels:,} ({diff_pct:.3f}%)"
            )
            print(f"   Max RGB difference: {max_diff:.1f} / 255")
            print(
                f"   Transparency change: {actual_transparent - expected_transparent:+,} pixels"
            )

            # Try to open the image automatically using the system's default viewer
            import subprocess
            import sys

            try:
                if sys.platform == "darwin":  # macOS
                    subprocess.run(["open", debug_path], check=False)
                elif sys.platform == "linux":  # Linux
                    subprocess.run(["xdg-open", debug_path], check=False)
                elif sys.platform == "win32":  # Windows
                    subprocess.run(["start", debug_path], shell=True, check=False)
            except Exception:
                # If opening fails, just continue - the path is already printed
                pass
        else:
            # Show in matplotlib window
            print("\n🔍 Showing debug visualization in matplotlib window...")
            print(
                f"   Different pixels: {diff_pixels:,} / {total_pixels:,} ({diff_pct:.3f}%)"
            )
            print(f"   Max RGB difference: {max_diff:.1f} / 255")
            print(
                f"   Transparency change: {actual_transparent - expected_transparent:+,} pixels"
            )
            plt.show()

    class RobustPNGSnapshotExtension(PNGImageSnapshotExtension):
        def matches(self, *, serialized_data: bytes, snapshot_data: bytes) -> bool:
            """
            Compare PNG images as numpy arrays instead of raw bytes.
            This is more robust against compression differences and platform variations.
            Generates debug visualization when --debug-visual flag is used.
            """
            # Convert both images to numpy arrays
            actual_img = Image.open(io.BytesIO(serialized_data))
            expected_img = Image.open(io.BytesIO(snapshot_data))

            actual_array = np.array(actual_img)
            expected_array = np.array(expected_img)

            # Check if arrays are equal
            arrays_equal = np.array_equal(actual_array, expected_array)

            # Check if we're in snapshot update mode
            if IS_SNAPSHOT_UPDATE:
                return arrays_equal
            else:
                # Generate debug visualization if arrays don't match and debug flag is set
                if not arrays_equal and (DEBUG_VISUAL or DEBUG_VISUAL_SAVE):
                    test_name = request.node.name

                    # Try to get tile and tms from test parameters
                    tile_info = None
                    try:
                        # Look for tile and tms in the request's fixturenames and cached values
                        if hasattr(request, "_pyfuncitem"):
                            callspec = getattr(request._pyfuncitem, "callspec", None)
                            if callspec and hasattr(callspec, "params"):
                                params = callspec.params
                                # Check for individual tile/tms params (test_pipeline_tiles)
                                if "tile" in params and "tms" in params:
                                    tile_info = (params["tile"], params["tms"])
                                # Check for projected_dataset_and_tile fixture (test_projected_coordinate_data)
                                elif "projected_dataset_and_tile" in params:
                                    ds, tile, tms = params["projected_dataset_and_tile"]
                                    tile_info = (tile, tms)
                    except Exception:
                        # Parameter extraction failed
                        pass

                    # Only create debug visualization if we have tile info
                    if tile_info:
                        create_debug_visualization(
                            actual_array, expected_array, test_name, tile_info
                        )
                    else:
                        print(
                            f"Warning: Could not extract tile info for debug visualization: {test_name}"
                        )

                # Normal test run - better error messages
                np.testing.assert_array_equal(actual_array, expected_array)
                return True

    return snapshot.use_extension(RobustPNGSnapshotExtension)
