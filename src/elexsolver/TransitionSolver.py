import logging
import warnings
from abc import ABC

import numpy as np

from elexsolver.logging import initialize_logging

initialize_logging()

LOG = logging.getLogger(__name__)


def mean_absolute_error(Y_expected: np.ndarray, Y_pred: np.ndarray):
    if isinstance(Y_expected, list):
        Y_expected = np.array(Y_expected)
    if isinstance(Y_pred, list):
        Y_pred = np.array(Y_pred)

    absolute_errors = np.abs(Y_pred - Y_expected)
    error_sum = np.sum(absolute_errors)
    return error_sum / len(absolute_errors)


class TransitionSolver(ABC):
    """
    Abstract class for (voter) transition solvers.
    """

    def __init__(self):
        self._mae = None

    def fit_predict(self, X: np.ndarray, Y: np.ndarray):
        raise NotImplementedError

    def get_prediction_interval(self, pi: float):
        raise NotImplementedError

    @property
    def MAE(self):
        return self._mae

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
        Ensure that in our (units x things) matrix, the number of units is
        at least twice as large as the number of things.
        """
        if A.shape[0] <= A.shape[1] or (A.shape[0] // 2) <= A.shape[1]:
            raise ValueError(f"Not enough units ({A.shape[0]}) relative to the number of things ({A.shape[1]}).")

    def _check_for_zero_units(self, A: np.ndarray):
        """
        If we have at least one unit whose columns are all zero, we can't continue.
        """
        if np.any(np.sum(A, axis=1) == 0):
            raise ValueError("Matrix cannot contain any rows (units) where all columns (things) are zero.")

    def _rescale(self, A: np.ndarray):
        """
        Rescale columns (things) to ensure they sum to 1 (100%).
        """
        A = A.copy().astype(float)

        if isinstance(A, np.ndarray):
            with warnings.catch_warnings():
                # Zeros are completely ok here;
                # means the thing (e.g. candidate) received zero votes.
                warnings.filterwarnings(
                    "ignore", category=RuntimeWarning, message="invalid value encountered in divide"
                )
                for j in range(0, A.shape[1]):
                    A[:, j] = A[:, j] / A[:, j].sum()
                return np.nan_to_num(A, nan=0, posinf=0, neginf=0)
        # pandas.DataFrame()
        for col in A.columns:
            A[col] /= A[col].sum()
        return A.fillna(0).replace(np.inf, 0).replace(-np.inf, 0)
