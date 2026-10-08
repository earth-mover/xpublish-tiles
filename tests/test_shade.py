import io

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pytest
from PIL import Image

from xpublish_tiles.lib import apply_range_colors
from xpublish_tiles.render.shade import (
    continuous_index,
    continuous_palette,
    discrete_index,
    indexed_image,
)

RANGE_COLORS = [None, "extend", "#ff8000", "transparent"]


def _continuous_field(lo: float, hi: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    interior = rng.uniform(lo, hi, 4000)
    edges = lo + (hi - lo) * np.arange(254) / 253
    special = [np.nan, lo - 1, hi + 1, lo, hi, -np.inf, np.inf]
    return np.concatenate([interior, edges, special]).reshape(-1, 1)


@pytest.mark.parametrize("cmap_name", ["viridis", "RdBu"])
@pytest.mark.parametrize("above", RANGE_COLORS)
@pytest.mark.parametrize("below", RANGE_COLORS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_continuous_matches_matplotlib(cmap_name, above, below, dtype):
    lo, hi = -0.5, 0.7
    x = _continuous_field(lo, hi).astype(dtype)
    cmap = apply_range_colors(mpl.colormaps[cmap_name], above, below)
    c, palette = continuous_palette(cmap)
    assert palette.shape == (256, 4)
    actual = palette[continuous_index(x, lo, hi, c)]
    expected = c((x - lo) / (hi - lo), bytes=True)
    np.testing.assert_array_equal(actual, expected)


def test_continuous_transparent_range_colors():
    cmap = apply_range_colors(mpl.colormaps["viridis"], "transparent", "transparent")
    c, palette = continuous_palette(cmap)
    x = np.array([[np.nan, -2.0, 2.0, 0.0]])
    rgba = palette[continuous_index(x, -1.0, 1.0, c)]
    np.testing.assert_array_equal(rgba[0, :3, 3], [0, 0, 0])
    assert rgba[0, 3, 3] == 255


def test_continuous_keeps_colormap_alpha():
    cmap = mcolors.ListedColormap([(1, 0, 0, 0.5), (0, 0, 1, 0.5)])
    c, palette = continuous_palette(cmap)
    rgba = palette[continuous_index(np.array([[0.1, 0.9]]), 0.0, 1.0, c)]
    np.testing.assert_array_equal(rgba[0, :, 3], [127, 127])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_discrete_index(dtype):
    color_key = {1: "#ff0000", 3: "#00ff00", 2: (0, 0, 1, 0.5)}
    data = np.array([[1, 2, 3, 7]], dtype=dtype)
    idx, palette = discrete_index(data, color_key)
    rgba = palette[idx]
    np.testing.assert_array_equal(
        rgba[0], [[255, 0, 0, 255], [0, 0, 255, 127], [0, 255, 0, 255], [0, 0, 0, 0]]
    )


def test_discrete_index_nan_is_transparent():
    idx, palette = discrete_index(np.array([[np.nan, 0.0]]), {0: "#ff0000"})
    np.testing.assert_array_equal(palette[idx][0], [[0, 0, 0, 0], [255, 0, 0, 255]])


def test_indexed_image_flips_rows_and_writes_trns():
    palette = np.array([[255, 0, 0, 255], [0, 0, 0, 0]], dtype=np.uint8)
    idx = np.array([[0, 0], [1, 1]])
    im = indexed_image(idx, palette)
    assert im.mode == "P"
    buf = io.BytesIO()
    im.save(buf, format="png")
    decoded = Image.open(io.BytesIO(buf.getvalue()))
    assert decoded.mode == "P"
    assert "transparency" in decoded.info
    np.testing.assert_array_equal(np.asarray(decoded.convert("RGBA")), palette[idx[::-1]])


def test_indexed_image_large_palette_is_rgba():
    palette = np.zeros((300, 4), dtype=np.uint8)
    palette[299] = (1, 2, 3, 4)
    im = indexed_image(np.array([[299, 0]]), palette)
    assert im.mode == "RGBA"
    np.testing.assert_array_equal(np.asarray(im)[0], [[1, 2, 3, 4], [0, 0, 0, 0]])
