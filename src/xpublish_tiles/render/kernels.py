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


@numba.njit(nogil=True, cache=True, boundscheck=False)
def _corner_row(xc, yc, j, west, south, xscale, yscale, out_x, out_y):
    """Pixel-space quad corners between centre rows ``j`` and ``j + 1``."""
    for i in range(out_x.size):
        x = 0.25 * (xc[j, i] + xc[j + 1, i] + xc[j, i + 1] + xc[j + 1, i + 1])
        y = 0.25 * (yc[j, i] + yc[j + 1, i] + yc[j, i + 1] + yc[j + 1, i + 1])
        out_x[i] = (x - west) * xscale
        out_y[i] = (y - south) * yscale


@numba.njit(nogil=True, cache=True, boundscheck=False)
def _half_step(xc, yc, j, i, dj, di):
    """Half the centre spacing from cell ``(j, i)`` along ``(dj, di)``."""
    ny, nx = xc.shape
    x0, y0 = xc[j, i], yc[j, i]
    jn, in_, jp, ip = j + dj, i + di, j - dj, i - di
    fn = jn < ny and in_ < nx and np.isfinite(xc[jn, in_] + yc[jn, in_])
    fp = jp >= 0 and ip >= 0 and np.isfinite(xc[jp, ip] + yc[jp, ip])
    if fn and fp:
        return True, (xc[jn, in_] - xc[jp, ip]) / 4, (yc[jn, in_] - yc[jp, ip]) / 4
    if fn:
        return True, (xc[jn, in_] - x0) / 2, (yc[jn, in_] - y0) / 2
    if fp:
        return True, (x0 - xc[jp, ip]) / 2, (y0 - yc[jp, ip]) / 2
    return False, 0.0, 0.0


@numba.njit(nogil=True, cache=True, boundscheck=False)
def _parallelogram(xc, yc, j, i, west, south, xscale, yscale):
    """Pixel-space quad of cell ``(j, i)`` from half-steps to its finite neighbours."""
    if not np.isfinite(xc[j, i] + yc[j, i]):
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ok, ux, uy = _half_step(xc, yc, j, i, 0, 1)
    ok2, vx, vy = _half_step(xc, yc, j, i, 1, 0)
    if not (ok and ok2):
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    px = (xc[j, i] - west) * xscale
    py = (yc[j, i] - south) * yscale
    ux, uy, vx, vy = ux * xscale, uy * yscale, vx * xscale, vy * yscale
    return (
        True,
        px - ux - vx,
        py - uy - vy,
        px + ux - vx,
        py + uy - vy,
        px + ux + vx,
        py + uy + vy,
        px - ux + vx,
        py - uy + vy,
    )


@numba.njit(nogil=True, cache=True, boundscheck=False)
def _paint_quad(ax, ay, bx, by, cx, cy, dx, dy, value, offset, counts_span, covered, out):
    """Give ``value`` to each uncovered pixel whose centre lies in the quad."""
    height, width = out.shape
    # pixel centres sit at k + 0.5
    col0 = max(np.int64(np.ceil(min(ax, bx, cx, dx) - 0.5)), 0)
    col1 = min(np.int64(np.floor(max(ax, bx, cx, dx) - 0.5)), width - 1)
    row0 = max(np.int64(np.ceil(min(ay, by, cy, dy) - 0.5)), 0)
    row1 = min(np.int64(np.floor(max(ay, by, cy, dy) - 0.5)), height - 1)
    shifted = value - offset
    countable = shifted >= 0 and shifted < counts_span
    for row in range(row0, row1 + 1):
        for col in range(col0, col1 + 1):
            if covered[row, col]:
                continue
            qx, qy = col + 0.5, row + 0.5
            e0 = _cross(ax, ay, bx, by, qx, qy)
            e1 = _cross(bx, by, cx, cy, qx, qy)
            e2 = _cross(cx, cy, dx, dy, qx, qy)
            e3 = _cross(dx, dy, ax, ay, qx, qy)
            if (e0 >= 0 and e1 >= 0 and e2 >= 0 and e3 >= 0) or (
                e0 <= 0 and e1 <= 0 and e2 <= 0 and e3 <= 0
            ):
                covered[row, col] = True
                if countable:
                    out[row, col] = value


@numba.njit(nogil=True, cache=True, boundscheck=False)
def _cross(ax, ay, bx, by, px, py):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


@numba.njit(nogil=True, cache=True, boundscheck=False)
def scattered_footprint_mode(
    source, xc, yc, west, south, xscale, yscale, counts_span, offset, out
):
    """``footprint_mode`` for 2D cell centres: each cell counts once, in its centre's pixel.

    A pixel with no centre takes the cell whose finite quad covers the pixel centre.
    """
    ny, nx = source.shape
    height, width = out.shape
    counts = np.zeros((height, width, counts_span), dtype=np.int32)
    covered = np.zeros((height, width), dtype=np.bool_)
    for j in range(ny):
        for i in range(nx):
            px = (xc[j, i] - west) * xscale
            py = (yc[j, i] - south) * yscale
            # NaN fails both comparisons, so off-disk centres drop out here
            if not (px >= 0 and px < width and py >= 0 and py < height):
                continue
            col, row = np.int64(px), np.int64(py)
            covered[row, col] = True
            shifted = source[j, i] - offset
            if shifted >= 0 and shifted < counts_span:
                counts[row, col, np.int64(shifted)] += 1

    for row in range(height):
        for col in range(width):
            best_code = -1
            best_count = 0
            for code in range(counts_span):
                if counts[row, col, code] > best_count:
                    best_count = counts[row, col, code]
                    best_code = code
            out[row, col] = best_code + offset if best_code >= 0 else np.nan

    for j in range(ny):
        for i in range(nx):
            if 0 < j < ny - 1 and 0 < i < nx - 1:
                continue
            ok, ax, ay, bx, by, cx, cy, dx, dy = _parallelogram(
                xc, yc, j, i, west, south, xscale, yscale
            )
            if ok:
                _paint_quad(
                    ax,
                    ay,
                    bx,
                    by,
                    cx,
                    cy,
                    dx,
                    dy,
                    source[j, i],
                    offset,
                    counts_span,
                    covered,
                    out,
                )
    if ny < 3 or nx < 3:
        return
    top_x = np.empty(nx - 1)
    top_y = np.empty(nx - 1)
    bottom_x = np.empty(nx - 1)
    bottom_y = np.empty(nx - 1)
    _corner_row(xc, yc, 0, west, south, xscale, yscale, top_x, top_y)
    for j in range(1, ny - 1):
        _corner_row(xc, yc, j, west, south, xscale, yscale, bottom_x, bottom_y)
        for i in range(1, nx - 1):
            ax, ay = top_x[i - 1], top_y[i - 1]
            bx, by = top_x[i], top_y[i]
            cx, cy = bottom_x[i], bottom_y[i]
            dx, dy = bottom_x[i - 1], bottom_y[i - 1]
            if not (
                np.isfinite(ax + ay)
                and np.isfinite(bx + by)
                and np.isfinite(cx + cy)
                and np.isfinite(dx + dy)
            ):
                # A limb cell has off-disk corners; span it with half-steps to
                # its finite neighbours instead.
                ok, ax, ay, bx, by, cx, cy, dx, dy = _parallelogram(
                    xc, yc, j, i, west, south, xscale, yscale
                )
                if not ok:
                    continue
            _paint_quad(
                ax,
                ay,
                bx,
                by,
                cx,
                cy,
                dx,
                dy,
                source[j, i],
                offset,
                counts_span,
                covered,
                out,
            )
        top_x, bottom_x = bottom_x, top_x
        top_y, bottom_y = bottom_y, top_y
