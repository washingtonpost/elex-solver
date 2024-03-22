import logging
import warnings

import numpy as np

from elexsolver.LinearSolver import LinearSolver
from elexsolver.logging import initialize_logging

initialize_logging()

LOG = logging.getLogger(__name__)


class TransitionSolver(LinearSolver):
    """
    Abstract class for transition solvers.
    """

    def __init__(self):
        """
        After model-fit, `self.coefficients` will contain
        the solved coefficients, an np.ndarray matrix of float of shape
        (number of columns in `X`) x (number of columns in `Y`).
        Each float represents the percent of how much of row x is part of column y.
        """
        super().__init__()

    def fit(self, X: np.ndarray, Y: np.ndarray, sample_weight: np.ndarray | None = None):
        """
        Parameters
        ----------
        X : np.ndarray matrix or pandas.DataFrame of int
            Must have the same number of rows as `Y` but can have any number of columns greater than the number of rows.
        Y : np.ndarray matrix or pandas.DataFrame of int
            Must have the same number of rows as `X` but can have any number of columns greater than the number of rows.
        sample_weight : list or np.ndarray or pandas.Series of int, optional
            Must have the same length (number of rows) as both `X` and `Y`.

        Returns
        -------
        `self` and populates `betas` with the beta coefficients determined by this solver.
        `betas` is an np.ndarray matrix of float of shape (number of columns in `X`) x (number of columns in `Y`).
        Each float represents the percent of how much of row x is part of column y.
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray matrix or pandas.DataFrame of int
            Must have the same dimensions as the `X` supplied to `fit()`.

        Returns
        -------
        `Y_hat`, np.ndarray of float of the same shape as Y.
        """
        if self.coefficients is None:
            raise RuntimeError("Solver must be fit before prediction can be performed.")
        return X @ self.coefficients

    def _check_data_type(self, A: np.ndarray):
        """
        Make sure we're starting with count data which we'll standardize to percentages
        by calling `self._rescale(A)` later.
        """
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
        X : np.ndarray matrix of int (same number of rows as `Y`)
        Y : np.ndarray matrix of int (same number of rows as `X`)
        weights : np.ndarray of int of the shape (number of rows in `X` and `Y`, 1), optional
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
