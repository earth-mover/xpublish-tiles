import numba
import numpy as np
import numpy.typing as npt

from xpublish_tiles.utils import NUMBA_PARALLEL


@numba.njit(nogil=True, cache=True)
def _in_quad(x, y, corner_x, corner_y, a, b):
    # crossing-number test with a half-open rule: a shared edge belongs to one cell
    inside = False
    for k in range(4):
        i0 = a + (1 if k in (1, 2) else 0)
        j0 = b + (1 if k in (2, 3) else 0)
        i1 = a + (1 if (k + 1) % 4 in (1, 2) else 0)
        j1 = b + (1 if (k + 1) % 4 in (2, 3) else 0)
        ax, ay = corner_x[i0, j0], corner_y[i0, j0]
        bx, by = corner_x[i1, j1], corner_y[i1, j1]
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return inside


@numba.jit(nopython=True, parallel=NUMBA_PARALLEL, nogil=True, cache=True)
def _locate_in_quads(
    px, py, corner_x, corner_y, cell_x_min, cell_x_max, cell_y_min, cell_y_max, period
):
    n0, n1 = cell_x_min.shape
    out = np.full(px.shape[0], -1, dtype=np.int64)
    for p in numba.prange(px.shape[0]):  # ty: ignore[not-iterable]
        y = py[p]
        for a in range(n0):
            for b in range(n1):
                if not (cell_y_min[a, b] <= y <= cell_y_max[a, b]):
                    continue
                x = px[p]
                for shift in range(2 if period > 0 else 1):
                    xs = x + shift * period
                    if cell_x_min[a, b] <= xs <= cell_x_max[a, b] and _in_quad(
                        xs, y, corner_x, corner_y, a, b
                    ):
                        out[p] = a * n1 + b
                        break
                if out[p] >= 0:
                    break
    return out


def locate_in_quads(
    px: np.ndarray,
    py: np.ndarray,
    corner_x: np.ndarray,
    corner_y: np.ndarray,
    cell_x_min: np.ndarray,
    cell_x_max: np.ndarray,
    cell_y_min: np.ndarray,
    cell_y_max: np.ndarray,
    period: float,
) -> npt.NDArray[np.int64]:
    """Flat C-order position in ``cell_x_min.shape`` of the quad that contains each point; -1 if none.

    ``corner_*`` have shape ``(n0 + 1, n1 + 1)``; cell ``(a, b)`` is the ring
    ``(a, b), (a + 1, b), (a + 1, b + 1), (a, b + 1)``. ``cell_*`` have shape ``(n0, n1)``.
    With ``period > 0`` a point also matches at ``x + period``. The lowest position wins.
    A linear scan per point: CellTree2d would need about 64 bytes per cell.
    """
    px = np.asarray(px, dtype=np.float64)
    out = _locate_in_quads(
        np.ravel(px),
        np.ravel(np.asarray(py, dtype=np.float64)),
        *(
            np.asarray(a, dtype=np.float64)
            for a in (corner_x, corner_y, cell_x_min, cell_x_max, cell_y_min, cell_y_max)
        ),
        float(period),
    )
    return out.reshape(px.shape)
