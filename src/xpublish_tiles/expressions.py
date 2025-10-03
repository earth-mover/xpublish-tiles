import re

import numexpr as ne
import numpy as np

BAND_NAME_PATTERN = re.compile(r"b(\d+)")


class ValidatedExpression:
    """A class to validate and evaluate mathematical expressions using numexpr.

    Attributes:
        expression (str): The mathematical expression to be validated and evaluated.
    """

    expression: str
    band_indexes: list[int]
    __slots__ = ["band_indexes", "expression"]

    def __init__(self, expression: str):
        self.expression = expression
        expr = ne.NumExpr(expression)
        input_names = expr.input_names
        matches = [BAND_NAME_PATTERN.fullmatch(name) for name in input_names]
        if any(match is None for match in matches):
            raise ValueError(
                f"Invalid band names in expression: {input_names}. "
                "Band names must be of the form 'b{n}' where n is an integer."
            )
        self.band_indexes = [int(match.group(1)) for match in matches]

    @property
    def band_names(self) -> list[str]:
        """Get the band names used in the expression.

        Returns:
            list[str]: A list of band names (e.g., 'b1', 'b2').
        """
        return [f"b{index}" for index in self.band_indexes]

    def evaluate(self, arrays: dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate the expression using the provided arrays.

        Args:
            arrays (dict): A dictionary mapping band names (e.g., 'b1', 'b2') to numpy arrays.

        Returns:
            np.ndarray: The result of evaluating the expression.
        """
        return ne.evaluate(self.expression, local_dict=arrays)
