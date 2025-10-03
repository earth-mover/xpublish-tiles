import numpy as np
import pytest

from xpublish_tiles.expressions import ValidatedExpression


def test_valid_expression():
    expr = ValidatedExpression("b1 + b2 / 2")
    assert expr.band_indexes == [1, 2]
    assert expr.band_names == ["b1", "b2"]

    expr = ValidatedExpression("cos(b100)")
    assert expr.band_indexes == [100]
    assert expr.band_names == ["b100"]


def test_invalid_expression():
    with pytest.raises(ValueError):
        ValidatedExpression("b1 + b2 / c")


def test_evaluate():
    expr = ValidatedExpression("b1 + b2 / 2")
    arrays = {
        "b1": np.array([[1, 2], [3, 4]]),
        "b2": np.array([[10, 20], [30, 40]]),
    }
    result = expr.evaluate(arrays)
    expected = np.array([[6, 12], [18, 24]])
    np.testing.assert_array_equal(result, expected)


def test_evaluate_from_array():
    expr = ValidatedExpression("b0 + b2 * 3 - b1 / 2")
    array = np.array(
        [
            [[1, 2], [3, 4]],  # b0
            [[10, 20], [30, 40]],  # b1
            [[100, 200], [300, 400]],  # b2
        ]
    )
    result = expr.evaluate_from_array(array)
    expected = array[0] + array[2] * 3 - array[1] / 2
    np.testing.assert_array_equal(result, expected)
