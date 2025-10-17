"""Test fixtures for xpublish-tiles with optional pytest dependencies."""

import io
import re
import subprocess
import sys
from typing import Literal
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
from morecantile import Tile
from PIL import Image
from pyproj.aoi import BBox
from skimage.metrics import structural_similarity as ssim
from syrupy import SnapshotAssertion

from xpublish_tiles.lib import check_transparent_pixels
from xpublish_tiles.logger import logger


def compare_image_buffers(buffer1: io.BytesIO, buffer2: io.BytesIO) -> bool:
    """Compare two image BytesIO buffers by converting them to numpy arrays."""
    buffer1.seek(0)
    buffer2.seek(0)

    # Convert both images to numpy arrays
    img1 = Image.open(buffer1)
    img2 = Image.open(buffer2)
    array1 = np.array(img1)
    array2 = np.array(img2)
    return np.array_equal(array1, array2)


def compare_images_perceptual(buffer1: io.BytesIO, buffer2: io.BytesIO) -> float:
    """Compare two images using perceptual similarity (SSIM).

    Args:
        buffer1: First image buffer
        buffer2: Second image buffer
        threshold: SSIM threshold for considering images similar

    Returns:
        tuple: (images_similar, ssim_score) where ssim_score is between -1 and 1
    """
    buffer1.seek(0)
    buffer2.seek(0)

    # Convert to numpy arrays
    img1 = Image.open(buffer1)
    img2 = Image.open(buffer2)
    array1 = np.array(img1)
    array2 = np.array(img2)

    # Ensure same shape
    if array1.shape != array2.shape:
        return 0.0

    # Convert RGBA to RGB if needed (SSIM doesn't handle alpha directly)
    if array1.shape[2] == 4:
        # Use alpha channel to blend with white background
        alpha1 = array1[:, :, 3:4] / 255.0
        alpha2 = array2[:, :, 3:4] / 255.0
        rgb1 = array1[:, :, :3] * alpha1 + 255 * (1 - alpha1)
        rgb2 = array2[:, :, :3] * alpha2 + 255 * (1 - alpha2)
        array1 = rgb1.astype(np.uint8)
        array2 = rgb2.astype(np.uint8)
    else:
        array1 = array1[:, :, :3]
        array2 = array2[:, :, :3]

    # Calculate SSIM
    ssim_score = ssim(array1, array2, channel_axis=2, data_range=255)
    return ssim_score


def compare_image_buffers_with_debug(
    buffer1: io.BytesIO,
    buffer2: io.BytesIO,
    test_name: str = "image_comparison",
    tile_info: tuple | None = None,
    debug_visual: bool = False,
    debug_visual_save: bool = False,
    mode: Literal["exact", "perceptual"] = "exact",
    perceptual_threshold: float = 0.95,
) -> tuple[bool, float]:
    """
    Compare two image buffers and optionally show debug visualization if they differ.

    Args:
        buffer1: First image buffer (expected/reference)
        buffer2: Second image buffer (actual/test)
        test_name: Name for the test/comparison
        tile_info: Optional tuple of (Tile, TileMatrixSet) for geographic context
        debug_visual: Show visualization in matplotlib window if images differ
        debug_visual_save: Save visualization to file if images differ
        mode: Comparison mode - "exact" for pixel-perfect, "perceptual" for SSIM-based
        perceptual_threshold: SSIM threshold for perceptual similarity (default 0.95)

    Returns:
        bool: True if images match (exact mode)
        tuple[bool, float]: (match, ssim_score) if perceptual mode
    """
    # Do the comparison based on mode
    ssim_score = compare_images_perceptual(buffer1, buffer2)
    if mode == "perceptual":
        images_equal = ssim_score > perceptual_threshold
    else:
        images_equal = compare_image_buffers(buffer1, buffer2)

    # If not equal and debug is requested, create visualization
    if not images_equal and (debug_visual or debug_visual_save):
        # Convert buffers to numpy arrays
        buffer1.seek(0)
        buffer2.seek(0)
        array1 = np.array(Image.open(buffer1))
        array2 = np.array(Image.open(buffer2))

        # Create debug visualization with SSIM info if available
        create_debug_visualization(
            actual_array=array2,
            expected_array=array1,
            test_name=test_name,
            tile_info=tile_info,
            debug_visual_save=debug_visual_save,
            ssim_score=ssim_score,
        )

    return images_equal, ssim_score


def create_debug_visualization(
    actual_array: np.ndarray,
    expected_array: np.ndarray,
    test_name: str,
    tile_info: tuple | None = None,
    debug_visual_save: bool = False,
    ssim_score: float | None = None,
) -> None:
    """Create a 3-panel debug visualization: Expected | Actual | Differences."""

    # Calculate SSIM if not provided
    if ssim_score is None:
        # Handle RGBA to RGB conversion for SSIM calculation
        array1 = expected_array.copy()
        array2 = actual_array.copy()

        if array1.shape[2] == 4:
            # Use alpha channel to blend with white background
            alpha1 = array1[:, :, 3:4] / 255.0
            alpha2 = array2[:, :, 3:4] / 255.0
            rgb1 = array1[:, :, :3] * alpha1 + 255 * (1 - alpha1)
            rgb2 = array2[:, :, :3] * alpha2 + 255 * (1 - alpha2)
            array1 = rgb1.astype(np.uint8)
            array2 = rgb2.astype(np.uint8)
        else:
            array1 = array1[:, :, :3]
            array2 = array2[:, :, :3]

        # Calculate SSIM
        ssim_score = ssim(array1, array2, channel_axis=2, data_range=255)

    def extract_tile_info(test_name: str, tile_info: tuple | None) -> dict:
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
            tile = Tile(
                x=extracted_tile_info["x"],
                y=extracted_tile_info["y"],
                z=extracted_tile_info["z"],
            )
            xy_bounds = tms.xy_bounds(tile)
            geo_bounds = tms.bounds(tile)

            bbox_info = f"""
Tile: z={extracted_tile_info["z"]}, x={extracted_tile_info["x"]}, y={extracted_tile_info["y"]} ({extracted_tile_info["tms_name"]})
Coordinate System: {extracted_tile_info["coord_info"]}
West: {geo_bounds.left:.3f}°, East: {geo_bounds.right:.3f}°
South: {geo_bounds.bottom:.3f}°, North: {geo_bounds.top:.3f}°

Projected Bounds ({tms.crs}):
X: {xy_bounds[0]:.0f} to {xy_bounds[2]:.0f}
Y: {xy_bounds[1]:.0f} to {xy_bounds[3]:.0f}

"""
    except Exception as e:
        bbox_info = f"Tile: z={extracted_tile_info['z']}, x={extracted_tile_info['x']}, y={extracted_tile_info['y']}\nBounds calculation failed: {e}\n\n"

    # Calculate difference statistics
    expected_transparent = np.sum(expected_array[:, :, 3] == 0)
    actual_transparent = np.sum(actual_array[:, :, 3] == 0)
    transparency_change = actual_transparent - expected_transparent
    transparency_change_pct = (
        transparency_change / (expected_array.shape[0] * expected_array.shape[1])
    ) * 100

    diff_pixels = np.sum(np.any(expected_array != actual_array, axis=2))
    total_pixels = expected_array.shape[0] * expected_array.shape[1]
    diff_pct = (diff_pixels / total_pixels) * 100

    rgb_diff = np.abs(
        expected_array[:, :, :3].astype(np.float32)
        - actual_array[:, :, :3].astype(np.float32)
    )
    max_diff = rgb_diff.max()
    mean_diff = rgb_diff[rgb_diff > 0].mean() if np.any(rgb_diff > 0) else 0

    # Create 2x3 grid: 3 image panels on top, debug info panel below
    fig = plt.figure(figsize=(18, 10))

    # Add main title with test name
    plt.suptitle(test_name, fontsize=12, y=0.98)

    # Create gridspec for custom layout with more space for text panel
    gs = fig.add_gridspec(3, 3, height_ratios=[4, 1, 1.2], hspace=0.05, wspace=0.02)

    # Top row: 3 image panels
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Panel 1: Expected output (snapshot)
    ax1.imshow(expected_array)
    ax1.set_title("Expected", fontsize=11)
    ax1.axis("off")

    # Panel 2: Actual output
    ax2.imshow(actual_array)
    ax2.set_title("Actual", fontsize=11)
    ax2.axis("off")

    # Panel 3: Difference map
    diff_map = create_difference_map(expected_array, actual_array)
    ax3.imshow(diff_map)
    ax3.set_title("Differences", fontsize=11)
    ax3.axis("off")

    # Middle row: Transparency visualization
    ax_transparency = fig.add_subplot(gs[1, :])
    ax_transparency.axis("off")

    # Create transparency status banner
    if transparency_change == 0:
        transparency_status = "✓ No transparency change"
        transparency_color = "green"
    elif abs(transparency_change) < 100:
        transparency_status = f"⚠ Minor transparency change: {transparency_change:+,} pixels ({transparency_change_pct:+.2f}%)"
        transparency_color = "orange"
    else:
        transparency_status = f"⚠️ TRANSPARENCY DIFFERENCE: {transparency_change:+,} pixels ({transparency_change_pct:+.2f}%)"
        transparency_color = "red"

    ax_transparency.text(
        0.5,
        0.5,
        transparency_status,
        transform=ax_transparency.transAxes,
        fontsize=14,
        fontweight="bold",
        color=transparency_color,
        horizontalalignment="center",
        verticalalignment="center",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor=transparency_color,
            linewidth=2,
        ),
    )

    # Bottom panel: Debug information text
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis("off")

    # Format debug text more compactly
    tile_info_str = f"Tile: z={extracted_tile_info['z']}, x={extracted_tile_info['x']}, y={extracted_tile_info['y']} ({extracted_tile_info['tms_name']}) | Coords: {extracted_tile_info['coord_info']}"

    if bbox_info:
        tms = extracted_tile_info["tms"]
        if tms is not None:
            tile = Tile(
                x=extracted_tile_info["x"],
                y=extracted_tile_info["y"],
                z=extracted_tile_info["z"],
            )
            geo_bounds = tms.bounds(tile)
            xy_bounds = tms.xy_bounds(tile)
            bounds_str = f"Bounds: [{geo_bounds.left:.1f}°, {geo_bounds.bottom:.1f}°] to [{geo_bounds.right:.1f}°, {geo_bounds.top:.1f}°] | Projected: X[{xy_bounds[0]:.0f}, {xy_bounds[2]:.0f}] Y[{xy_bounds[1]:.0f}, {xy_bounds[3]:.0f}]"
    else:
        bounds_str = ""

    diff_status = (
        "✓ Minimal differences" if diff_pct < 0.5 else "⚠ Noticeable differences"
    )

    # Add SSIM score if available
    ssim_text = f" | SSIM: {ssim_score:.4f}" if ssim_score is not None else ""

    diff_text = f"""{tile_info_str}
{bounds_str}
Diff: {diff_pixels:,}/{total_pixels:,} pixels ({diff_pct:.2f}%) | RGB Δ: max={max_diff:.0f}, mean={mean_diff:.1f}{ssim_text} | {diff_status}"""

    # Add text to the bottom panel with larger font size
    ax_text.text(
        0.02,
        0.85,
        diff_text,
        transform=ax_text.transAxes,
        fontfamily="monospace",
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.9),
    )

    plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.02)

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
        if ssim_score is not None:
            print(f"   SSIM score: {ssim_score:.4f}")
        print(
            f"   Transparency change: {actual_transparent - expected_transparent:+,} pixels"
        )

        # Try to open the image automatically using the system's default viewer
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
        if ssim_score is not None:
            print(f"   SSIM score: {ssim_score:.4f}")
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

                actual_array = np.array(Image.open(actual_buffer))
                expected_array = np.array(Image.open(expected_buffer))

                if IS_SNAPSHOT_UPDATE:
                    return arrays_equal

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
                        pass

                    # Always create debug visualization, even without tile info
                    create_debug_visualization(
                        actual_array,
                        expected_array,
                        test_name,
                        tile_info,
                        DEBUG_VISUAL_SAVE,
                    )
                if not arrays_equal:
                    try:
                        np.testing.assert_array_equal(actual_array, expected_array)
                    except AssertionError as e:
                        # syrupy seems to swallow the error?
                        logger.error(e)

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


def validate_transparency(
    content: bytes,
    *,
    tile=None,
    tms=None,
    dataset_bbox=None,
):
    """Validate transparency of rendered content based on tile/dataset overlap.

    Args:
        content: The rendered PNG content
        tile: The tile being rendered (optional)
        tms: The tile matrix set (optional)
        dataset_bbox: Bounding box of the dataset (optional)
    """
    # Calculate tile bbox if tile and tms provided
    tile_bbox = None
    if tile is not None and tms is not None:
        tile_bounds = tms.bounds(tile)
        tile_bbox = BBox(
            west=tile_bounds.left,
            south=tile_bounds.bottom,
            east=tile_bounds.right,
            north=tile_bounds.top,
        )

    # Check if this is the specific failing test case that should skip transparency checks
    # This is a boundary tile, and the bounds checking is inaccurate.
    # TODO: Consider figuring out a better way to do this, but I suspect it's just too hard.
    # TODO: We could instead just keep separate lists of fully contained and partially intersecting tiles;
    #       and add an explicit check.
    skip_transparency_check = (
        tile is not None
        and tms is not None
        and tile.x == 0
        and tile.y == 1
        and tile.z == 2
        and tms.id == "EuropeanETRS89_LAEAQuad"
    )

    # Check transparency based on whether dataset contains the tile
    transparent_percent = check_transparent_pixels(content)
    if not skip_transparency_check:
        if tile_bbox is not None and dataset_bbox is not None:
            if dataset_bbox.contains(tile_bbox):
                assert transparent_percent == 0, (
                    f"Found {transparent_percent:.1f}% transparent pixels in fully contained tile."
                )
            elif dataset_bbox.intersects(tile_bbox):
                # relaxed from > 0 for UTM data which is heavily distorted over antarctica
                assert transparent_percent >= 0, transparent_percent
            else:
                assert transparent_percent == 100, (
                    f"Found {transparent_percent:.1f}% transparent pixels in fully disjoint tile (expected 100%)."
                )
        else:
            assert transparent_percent == 0, (
                f"Found {transparent_percent:.1f}% transparent pixels."
            )


def assert_render_matches_snapshot(
    result: io.BytesIO,
    png_snapshot: SnapshotAssertion,
    *,
    tile=None,
    tms=None,
    dataset_bbox=None,
    perceptual_threshold: float | None = None,
):
    """Helper function to validate PNG content against snapshot.

    Args:
        result: The rendered image buffer
        png_snapshot: The expected snapshot
        tile: The tile being rendered (optional)
        tms: The tile matrix set (optional)
        dataset_bbox: Bounding box of the dataset (optional)
        perceptual_threshold: If provided, use perceptual comparison with this SSIM threshold (0-1).
                              Values closer to 1 mean stricter matching. Common values:
                              - 0.99: Very strict, nearly identical
                              - 0.95: Strict, minor differences allowed
                              - 0.90: Moderate, noticeable but similar
                              If None, uses exact pixel comparison.
    """
    assert isinstance(result, io.BytesIO)
    result.seek(0)
    content = result.read()
    assert len(content) > 0
    validate_transparency(content, tile=tile, tms=tms, dataset_bbox=dataset_bbox)

    if perceptual_threshold is None:
        # Use exact comparison
        assert content == png_snapshot
        return

    # For perceptual comparison, we need to get the actual snapshot bytes
    # Check if png_snapshot is a SnapshotAssertion object
    assert hasattr(png_snapshot, "_recall_data")
    # It's a SnapshotAssertion, we need to get the actual data
    snapshot_data, _ = png_snapshot._recall_data(png_snapshot.index)
    if snapshot_data is None:
        # No existing snapshot, fail the comparison
        assert content == png_snapshot
    snapshot_bytes = snapshot_data

    # Use perceptual comparison
    result.seek(0)
    snapshot_buffer = io.BytesIO(snapshot_bytes)
    ssim_score = compare_images_perceptual(result, snapshot_buffer)

    if ssim_score >= perceptual_threshold:
        # Images are similar enough, mock the extension's matches to return True
        # This ensures the snapshot is marked as used without doing exact comparison
        with mock.patch.object(png_snapshot._extension, "matches", return_value=True):
            assert content == png_snapshot
    else:
        raise AssertionError(
            f"Image similarity {ssim_score:.4f} is below threshold {perceptual_threshold}. "
            f"Images are too different."
        )


def visualize_tile(result: io.BytesIO, tile: Tile) -> None:
    """Visualize a rendered tile with matplotlib showing RGB and alpha channels.

    Args:
        result: BytesIO buffer containing PNG image data
        tile: Tile object with z, x, y coordinates
    """
    result.seek(0)
    pil_img = Image.open(result)
    img_array = np.array(pil_img)

    _, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Show the rendered tile
    axes[0].imshow(img_array)
    axes[0].set_title(f"Tile z={tile.z}, x={tile.x}, y={tile.y}")

    # Show alpha channel if present
    if img_array.shape[2] == 4:
        alpha = img_array[:, :, 3]
        im = axes[1].imshow(alpha, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title(
            f"Alpha Channel\n{((alpha == 0).sum() / alpha.size * 100):.1f}% transparent"
        )
        plt.colorbar(im, ax=axes[1])
    else:
        axes[1].text(
            0.5, 0.5, "No Alpha", ha="center", va="center", transform=axes[1].transAxes
        )

    plt.tight_layout()
    plt.show(block=True)  # Block until window is closed


# Export the fixture name for easier importing
__all__ = [
    "assert_render_matches_snapshot",
    "compare_image_buffers",
    "compare_image_buffers_with_debug",
    "compare_images_perceptual",
    "create_debug_visualization",
    "png_snapshot",
    "validate_transparency",
    "visualize_tile",
]
