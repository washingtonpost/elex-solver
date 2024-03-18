import logging
import warnings
from abc import ABC

import numpy as np

from elexsolver.logging import initialize_logging

initialize_logging()

LOG = logging.getLogger(__name__)


class TransitionSolver(ABC):
    """
    Abstract class for transition solvers.
    """

    def __init__(self):
        self._transitions = None

    def fit_predict(self, X: np.ndarray, Y: np.ndarray, weights: np.ndarray | None = None):
        """
        After this method finishes, transitions will be available in the `transitions` class member.

        Parameters
        ----------
        `X` : np.ndarray matrix or pandas.DataFrame of int
            Must have the same number of rows as `Y` but can have any number of columns greater than the number of rows.
        `Y` : np.ndarray matrix or pandas.DataFrame of int
            Must have the same number of rows as `X` but can have any number of columns greater than the number of rows.
        `weights` : list, np.ndarray, or pandas.Series of int, optional
            Must have the same length (number of rows) as both `X` and `Y`.

        Returns
        -------
        np.ndarray matrix of float of shape (number of columns in `X`) x (number of columns in `Y`).
        Each float represents the percent of how much of row x is part of column y.
        """
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
