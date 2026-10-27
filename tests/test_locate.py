import numpy as np
import numpy.testing as npt

from xpublish_tiles.locate import locate_in_quads

Arrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _cell_bounds(corner_x, corner_y):
    xs = np.stack(
        [corner_x[:-1, :-1], corner_x[1:, :-1], corner_x[1:, 1:], corner_x[:-1, 1:]]
    )
    ys = np.stack(
        [corner_y[:-1, :-1], corner_y[1:, :-1], corner_y[1:, 1:], corner_y[:-1, 1:]]
    )
    return xs.min(0), xs.max(0), ys.min(0), ys.max(0)


def _mesh(nx, ny, dx=1.0, dy=1.0, x0=0.0, y0=0.0, dtype=np.float64) -> Arrays:
    """Test only: corners and cell bounds of a regular mesh."""
    corner_y, corner_x = np.meshgrid(
        y0 + dy * np.arange(ny + 1, dtype=dtype),
        x0 + dx * np.arange(nx + 1, dtype=dtype),
        indexing="ij",
    )
    x_min, x_max, y_min, y_max = _cell_bounds(corner_x, corner_y)
    return corner_x, corner_y, x_min, x_max, y_min, y_max


def test_locate_quads_regular():
    args = _mesh(4, 3)
    px = np.array([0.5, 3.5, 1.2, -0.1, 4.1, 2.5])
    py = np.array([0.5, 2.5, 1.7, 0.5, 0.5, -0.2])
    npt.assert_array_equal(locate_in_quads(px, py, *args, 0.0), [0, 11, 5, -1, -1, -1])


def test_locate_quads_shared_edge_assigned_once():
    args = _mesh(4, 3)
    (pos,) = locate_in_quads(np.array([2.0]), np.array([1.5]), *args, 0.0)
    assert pos in (5, 6)


def test_locate_quads_rotated():
    # 45-degree rotated unit square: its bbox corners lie outside the quad
    corner_x = np.array([[0.0, 1.0], [-1.0, 0.0]])
    corner_y = np.array([[0.0, 1.0], [1.0, 2.0]])
    x_min, x_max, y_min, y_max = _cell_bounds(corner_x, corner_y)
    out = locate_in_quads(
        np.array([0.0, 0.9]),
        np.array([1.0, 0.1]),
        corner_x,
        corner_y,
        x_min,
        x_max,
        y_min,
        y_max,
        0.0,
    )
    npt.assert_array_equal(out, [0, -1])


def test_locate_quads_period_wrap():
    args = _mesh(4, 1, dx=90.0, x0=10.0)
    px, py = np.array([5.0, 365.0]), np.array([0.5, 0.5])
    npt.assert_array_equal(locate_in_quads(px, py, *args, 360.0), [3, 3])
    npt.assert_array_equal(locate_in_quads(px[:1], py[:1], *args, 0.0), [-1])


def test_locate_quads_nonfinite():
    args = _mesh(4, 3)
    out = locate_in_quads(
        np.array([np.nan, np.inf, 0.5]), np.array([0.5, 0.5, np.nan]), *args, 360.0
    )
    npt.assert_array_equal(out, [-1, -1, -1])


def test_locate_quads_float32():
    args = _mesh(4, 3, dtype=np.float32)
    npt.assert_array_equal(
        locate_in_quads(np.array([3.5]), np.array([2.5]), *args, 0.0), [11]
    )


def test_locate_quads_overlap_lowest_row_wins():
    # rows 0 and 2 both contain the point; the lowest position must win
    corner_x, corner_y = np.meshgrid([0.0, 1.0], [0.0, 1.0, 0.0, 1.0], indexing="xy")
    x_min, x_max, y_min, y_max = _cell_bounds(corner_x, corner_y)
    args: Arrays = (corner_x, corner_y, x_min, x_max, y_min, y_max)
    out = locate_in_quads(np.array([0.5]), np.array([0.5]), *args, 0.0)
    npt.assert_array_equal(out, [0])
