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

    def mean_absolute_error(self, X: np.ndarray, Y: np.ndarray):
        raise NotImplementedError

    def get_prediction_interval(self, pi: float):
        raise NotImplementedError

    def _get_expected_totals(self, A: np.ndarray):
        output = np.sum(A, axis=0)
        # rescaling in case any columns had been dropped previously
        return output / sum(output)

    def _check_any_element_nan_or_inf(self, A: np.ndarray):
        """
        Check whether any element in a matrix or vector is NaN or infinity
        """
        if np.any(np.isnan(A)) or np.any(np.isinf(A)):
            raise ValueError("Matrix contains NaN or Infinity")

    def _check_percentages(self, A: np.ndarray):
        """
        Verify that every element in matrix A is >= 0 and <= 1.
        """
        if not np.all((A >= 0) & (A <= 1)):
            raise ValueError("Matrix contains values less than 0 or greater than 1.")

    def _check_and_rescale(self, A: np.ndarray):
        """
        Rescale columns (units) so that they sum to 1 (100%).
        """
        if not np.all(A.sum(axis=0) == 1):
            LOG.warn("Each column (unit) needs to sum to 1.  Rescaling...")
            if isinstance(A, np.ndarray):
                for j in range(0, A.shape[1]):
                    A[:, j] /= A[:, j].sum()
            else:
                # pandas.DataFrame()
                for col in A.columns:
                    A[col] /= A[col].sum()
        return A