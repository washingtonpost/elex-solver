import logging
import warnings
from abc import ABC

import numpy as np

from elexsolver.logging import initialize_logging

initialize_logging()

LOG = logging.getLogger(__name__)


class LinearSolverException(Exception):
    pass


class IllConditionedMatrixException(LinearSolverException):
    pass


class LinearSolver(ABC):
    """
    An abstract base class for a linear solver
    """

    CONDITION_WARNING_MIN = 50  # arbitrary
    CONDITION_ERROR_MIN = 1e8  # based on scipy

    def __init__(self):
        self.coefficients = None

    @classmethod
    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None, lambda_: float = 0.0, *kwargs):
        """
        Fits model
        """
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Use coefficients to predict
        """
        self._check_any_element_nan_or_inf(x)

        return x @ self.coefficients

    def get_coefficients(self) -> np.ndarray:
        """
        Returns model coefficients
        """
        return self.coefficients

    def _check_matrix_condition(self, x):
        """
        Check condition number of the design matrix as a check for multicolinearity.
        This is equivalent to the ratio between the largest and the smallest singular value of the design matrix.
        """
        condition_number = np.linalg.cond(x)
        if condition_number >= self.CONDITION_ERROR_MIN:
            raise IllConditionedMatrixException(
                f"Ill-conditioned matrix detected. Matrix condition number >= {self.CONDITION_ERROR_MIN}"
            )
        elif condition_number >= self.CONDITION_WARNING_MIN:
            warnings.warn("Warning: Ill-conditioned matrix detected. result is not guaranteed to be accurate")

    def _check_any_element_nan_or_inf(self, x):
        """
        Check whether any element in a matrix or vector is NaN or infinity
        """
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            raise ValueError("Array contains NaN or Infinity")

    def _check_intercept(self, x):
        """
        Check whether the first column is all 1s (normal intercept) otherwise raises a warning.
        """
        if ~np.all(x[:, 0] == 1):
            warnings.warn("Warning: fit_intercept=True and not all elements of the first columns are 1s")
