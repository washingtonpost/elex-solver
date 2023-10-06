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

    def fit_predict(self, X: np.ndarray, Y: np.ndarray):
        raise NotImplementedError

    def mean_absolute_error(self, X: np.ndarray, Y: np.ndarray):
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
