"""Numba kernels for the raster renderer."""

import numba
import numpy as np

from xpublish_tiles.utils import NUMBA_PARALLEL


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
