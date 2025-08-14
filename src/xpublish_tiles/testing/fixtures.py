"""Test fixtures for xpublish-tiles with optional pytest dependencies."""

import io
import re
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def compare_image_buffers(buffer1: io.BytesIO, buffer2: io.BytesIO) -> bool:
    """Compare two image BytesIO buffers by converting them to numpy arrays."""
    buffer1.seek(0)
    buffer2.seek(0)

    # Convert both images to numpy arrays
    img1 = Image.open(buffer1)
    img2 = Image.open(buffer2)

    array1 = np.array(img1)
    array2 = np.array(img2)

    # Compare arrays using numpy array equality
    return np.array_equal(array1, array2)


def create_debug_visualization(
    actual_array: np.ndarray,
    expected_array: np.ndarray,
    test_name: str,
    tile_info: Optional[tuple] = None,
    debug_visual_save: bool = False,
) -> None:
    """Create a 3-panel debug visualization: Expected | Actual | Differences."""

    def extract_tile_info(test_name: str, tile_info: Optional[tuple]) -> dict:
        """Extract tile coordinates and TMS info from tile parameter."""
        if tile_info is None:
            return {
                "tms_name": "unknown",
                "z": 0,
                "x": 0,
                "y": 0,
                "coord_info": "unknown",
                "tms": None,
            }

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
    def create_difference_map(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        # Calculate absolute differences for RGB channels (ignore alpha)
        diff_rgb = np.abs(
            expected[:, :, :3].astype(np.float32) - actual[:, :, :3].astype(np.float32)
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
        if tms is not None:
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

    if debug_visual_save:
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


def _create_png_snapshot_fixture():
    """Create the png_snapshot fixture. Only available when pytest is installed."""
    try:
        import pytest
        from syrupy.extensions.image import PNGImageSnapshotExtension
    except ImportError as e:
        raise ImportError(
            "pytest and syrupy are required for png_snapshot fixture. "
            "Install with: uv add --group testing pytest syrupy"
        ) from e

    @pytest.fixture
    def png_snapshot(snapshot, pytestconfig, request):
        """PNG snapshot with custom numpy array comparison and optional debug visualization."""

        IS_SNAPSHOT_UPDATE = pytestconfig.getoption("--snapshot-update", default=False)
        DEBUG_VISUAL = pytestconfig.getoption("--debug-visual", default=False)
        DEBUG_VISUAL_SAVE = pytestconfig.getoption("--debug-visual-save", default=False)

        class RobustPNGSnapshotExtension(PNGImageSnapshotExtension):
            def matches(self, *, serialized_data: bytes, snapshot_data: bytes) -> bool:
                """
                Compare PNG images as numpy arrays instead of raw bytes.
                This is more robust against compression differences and platform variations.
                Generates debug visualization when --debug-visual flag is used.
                """
                # Use the helper function to compare images
                actual_buffer = io.BytesIO(serialized_data)
                expected_buffer = io.BytesIO(snapshot_data)
                arrays_equal = compare_image_buffers(expected_buffer, actual_buffer)

                if IS_SNAPSHOT_UPDATE:
                    return arrays_equal

                # Convert both images to numpy arrays for debug visualization
                actual_img = Image.open(actual_buffer)
                expected_img = Image.open(expected_buffer)
                actual_array = np.array(actual_img)
                expected_array = np.array(expected_img)

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
                                    _, tile, tms = params["projected_dataset_and_tile"]
                                    tile_info = (tile, tms)
                    except Exception:
                        # Parameter extraction failed
                        pass

                    # Only create debug visualization if we have tile info
                    if tile_info:
                        create_debug_visualization(
                            actual_array,
                            expected_array,
                            test_name,
                            tile_info,
                            DEBUG_VISUAL_SAVE,
                        )
                    else:
                        print(
                            f"Warning: Could not extract tile info for debug visualization: {test_name}"
                        )

                    # Normal test run - better error messages
                    np.testing.assert_array_equal(actual_array, expected_array)

                return arrays_equal

        return snapshot.use_extension(RobustPNGSnapshotExtension)

    return png_snapshot


# Create the fixture when pytest is available
try:
    png_snapshot = _create_png_snapshot_fixture()
except ImportError:
    # Define a placeholder that will raise helpful error when accessed
    def png_snapshot(*args, **kwargs):
        raise ImportError(
            "pytest and syrupy are required for png_snapshot fixture. "
            "Install with: uv add --group testing pytest syrupy"
        )


# Export the fixture name for easier importing
__all__ = ["compare_image_buffers", "create_debug_visualization", "png_snapshot"]
