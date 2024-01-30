import logging
import warnings
from abc import ABC

import numpy as np

from elexsolver.logging import initialize_logging

initialize_logging()

LOG = logging.getLogger(__name__)


class TransitionSolver(ABC):
    """
    Abstract class for (voter) transition solvers.
    """

    def __init__(self):
        self._transitions = None

    def fit_predict(self, X: np.ndarray, Y: np.ndarray, weights: np.ndarray | None = None):
        raise NotImplementedError

    def get_prediction_interval(self, pi: float):
        raise NotImplementedError

    @property
    def transitions(self) -> np.ndarray:
        return self._transitions

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

    def _rescale(self, A: np.ndarray) -> np.ndarray:
        """
        Rescale rows (units) to ensure they sum to 1 (100%).
        """
        A = A.copy().astype(float)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
            A = (A.T / A.sum(axis=1)).T

        return np.nan_to_num(A, nan=0, posinf=0, neginf=0)

    def _check_and_prepare_weights(self, X: np.ndarray, Y: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
        """
        If `weights` is not None, and `weights` has the same number of rows in both matrices `X` and `Y`,
        we'll rescale the weights by taking the square root after dividing them by their sum,
        then return a diagonal matrix containing these now-normalized weights.
        If `weights` is None, return a diagonal matrix of ones.

        Parameters
        ----------
        `X` : np.ndarray matrix of int (same number of rows as `Y`)
        `Y` : np.ndarray matrix of int (same number of rows as `X`)
        `weights` : np.ndarray of int of the shape (number of rows in `X` and `Y`, 1), optional
        """

        if weights is not None:
            if len(weights) != X.shape[0] and len(weights) != Y.shape[0]:
                raise ValueError("weights must be the same length as the number of rows in X and Y.")
            if isinstance(weights, list):
                weights = np.array(weights).copy()
            elif not isinstance(weights, np.ndarray):
                # pandas.Series
                weights = weights.values.copy()
            return np.diag(np.sqrt(weights.flatten() / weights.sum()))

        return np.diag(np.ones((Y.shape[0],)))
