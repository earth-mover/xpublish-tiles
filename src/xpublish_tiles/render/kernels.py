"""Numba kernels for the raster renderer."""

import numba
import numpy as np

from xpublish_tiles.utils import NUMBA_PARALLEL


@numba.njit(parallel=NUMBA_PARALLEL, nogil=True, cache=True, boundscheck=False)
def footprint_mode(
    source, ystart, ystop, xstart, xstop, ynear, xnear, counts_span, offset, out
):
    """Mode of ``source`` over each output pixel's [start, stop) footprint.

    An empty footprint falls back to the nearest cell, or NaN if there is none.
    """
    ny, nx = out.shape
    for j in numba.prange(ny):  # ty: ignore[not-iterable]
        # ``stamp`` tags counts by pixel, so no per-pixel clear of the span
        counts = np.zeros(counts_span, dtype=np.int64)
        stamp = np.full(counts_span, -1, dtype=np.int64)
        for i in range(nx):
            y0, y1, x0, x1 = ystart[j], ystop[j], xstart[i], xstop[i]
            if y1 <= y0 or x1 <= x0:
                jy, ix = ynear[j], xnear[i]
                out[j, i] = np.nan if jy < 0 or ix < 0 else source[jy, ix]
                continue
            pixel = j * nx + i
            best_code = -1
            best_count = 0
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    shifted = source[yy, xx] - offset
                    # NaN fails both comparisons, so fill values drop out here
                    if not (shifted >= 0 and shifted < counts_span):
                        continue
                    code = np.int64(shifted)
                    if stamp[code] != pixel:
                        stamp[code] = pixel
                        counts[code] = 0
                    counts[code] += 1
                    if counts[code] > best_count:
                        best_count = counts[code]
                        best_code = code
            out[j, i] = best_code + offset if best_code >= 0 else np.nan


@numba.njit(parallel=NUMBA_PARALLEL, nogil=True, cache=True, boundscheck=False)
def offdisk_quad_mask(xc, yc):
    """Mark cells whose quad touches a non-finite (off-disk) vertex.

    Any non-finite coordinate in the 3x3 neighbourhood marks the cell.
    """
    ny, nx = xc.shape
    bad = np.zeros((ny, nx), dtype=np.bool_)
    for j in numba.prange(ny):  # ty: ignore[not-iterable]
        for i in range(nx):
            found = False
            for dj in range(-1, 2):
                jj = j + dj
                if jj < 0 or jj >= ny:
                    continue
                for di in range(-1, 2):
                    ii = i + di
                    if ii < 0 or ii >= nx:
                        continue
                    if not (np.isfinite(xc[jj, ii]) and np.isfinite(yc[jj, ii])):
                        found = True
                        break
                if found:
                    break
            bad[j, i] = found
    return bad
