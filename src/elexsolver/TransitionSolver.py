import logging
from abc import ABC

import numpy as np

from elexsolver.logging import initialize_logging

initialize_logging()

LOG = logging.getLogger(__name__)


class TransitionSolver(ABC):
    """
    Abstract class for (voter) transition solvers.
    """

    def fit_predict(self, X: np.ndarray, Y: np.ndarray):
        raise NotImplementedError

    def get_prediction_interval(self, pi: float):
        raise NotImplementedError

    def mean_absolute_error(self, Y_expected: np.ndarray, Y_pred: np.ndarray):
        absolute_errors = np.abs(Y_pred - Y_expected)
        error_sum = np.sum(absolute_errors)
        return error_sum / len(absolute_errors)

    def _check_any_element_nan_or_inf(self, A: np.ndarray):
        """
        Check whether any element in a matrix or vector is NaN or infinity
        """
        if np.any(np.isnan(A)) or np.any(np.isinf(A)):
            raise ValueError("Matrix contains NaN or Infinity.")

    def _check_data_type(self, A: np.ndarray):
        if not np.all(A.astype("int64") == A):
            raise ValueError("Matrix must contain integers.")

    def _check_dimensions(self, A: np.ndarray):
        """
        Ensure that in our (things x units) matrix, the number of units is
        at least twice as large as the number of things.
        """
        if A.shape[1] <= A.shape[0] or (A.shape[1] // 2) <= A.shape[0]:
            raise ValueError(f"Not enough units ({A.shape[1]}) relative to the number of things ({A.shape[0]}).")

    def _rescale(self, A: np.ndarray):
        """
        Rescale columns (units) to ensure they sum to 1 (100%).
        """
        if isinstance(A, np.ndarray):
            for j in range(0, A.shape[1]):
                A[:, j] = A[:, j] / A[:, j].sum()
            return np.nan_to_num(A, nan=0, posinf=0, neginf=0)
        # pandas.DataFrame()
        for col in A.columns:
            A[col] /= A[col].sum()
        return A.fillna(0).replace(np.inf, 0).replace(-np.inf, 0)
