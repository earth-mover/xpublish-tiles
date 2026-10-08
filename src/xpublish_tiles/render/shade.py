"""Colormap aggregates to indexed (palette) images."""

import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

# 253 colours + under, over, bad fill the 256-entry PNG palette.
PALETTE_COLORS = 253


def continuous_palette(cmap: mcolors.Colormap) -> tuple[mcolors.Colormap, np.ndarray]:
    """``_lut`` rows are already in index order: N colours, then under, over, bad."""
    cmap = cmap.resampled(PALETTE_COLORS)
    cmap._init()  # ty: ignore[unresolved-attribute]
    # Same truncation as ``Colormap.__call__(..., bytes=True)``.
    return cmap, (cmap._lut * 255).astype(np.uint8)  # ty: ignore[unresolved-attribute]


def continuous_index(
    data: np.ndarray, lo: float, hi: float, cmap: mcolors.Colormap
) -> np.ndarray:
    """matplotlib's ``Colormap.__call__`` index rule (colors.py, ``_get_rgba_and_mask``)."""
    n = cmap.N
    with np.errstate(invalid="ignore"):
        # Same float order as datashader's ``scaled_data`` then mpl's ``xa *= N``.
        xa = (data - lo) / (hi - lo)
        xa *= n
        xa[xa == n] = n - 1
        idx = np.clip(xa, 0, n - 1).astype(np.uint8)
    idx[xa < 0] = cmap._i_under  # ty: ignore[unresolved-attribute]
    idx[xa >= n] = cmap._i_over  # ty: ignore[unresolved-attribute]
    idx[np.isnan(xa)] = cmap._i_bad  # ty: ignore[unresolved-attribute]
    return idx


def discrete_index(data: np.ndarray, color_key: dict) -> tuple[np.ndarray, np.ndarray]:
    """Index of each flag value in ``color_key``; values not in it are transparent."""
    flags = np.asarray(list(color_key), dtype=data.dtype)
    palette = np.array([mcolors.to_rgba(c) for c in color_key.values()] + [(0, 0, 0, 0)])
    order = np.argsort(flags)
    sorted_flags = flags[order]
    pos = np.clip(np.searchsorted(sorted_flags, data), 0, flags.size - 1)
    idx = np.where(sorted_flags[pos] == data, order[pos], flags.size)
    return idx, (palette * 255).astype(np.uint8)


def indexed_image(idx: np.ndarray, palette: np.ndarray) -> Image.Image:
    """Aggregates are y-up; images are top-down (as datashader's ``to_pil``)."""
    idx = idx[::-1]
    if len(palette) > 256:
        return Image.fromarray(palette[idx], mode="RGBA")
    im = Image.fromarray(idx.astype(np.uint8), mode="P")
    im.putpalette(palette.tobytes(), rawmode="RGBA")
    return im
