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
        self.rng = np.random.default_rng(seed=0)

    @classmethod
    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None, lambda_: float = 0.0, cache: bool = True, **kwargs):
        """
        Fits model
        """
        raise NotImplementedError

    def predict(self, x: np.ndarray, coefficients: np.ndarray | None = None) -> np.ndarray:
        """
        Use coefficients to predict
        """
        self._check_any_element_nan_or_inf(x)

        if coefficients is None:
            return x @ self.coefficients
        return x @ coefficients

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

    def residuals(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None, K: int | None = None, center: bool = True, **kwargs) -> np.ndarray:
        """
        Computes residuals for the model
        """
        if K is None:
            # when K is None we are just using the training residuals
            y_hat = self.predict(x).reshape(*y.shape)
            residuals = y - y_hat
        else:
            if weights is None:
                weights = np.ones_like(y)

            # create indices for k-fold crossfiting
            indices = np.arange(x.shape[0])
            # shuffle for random order of datapoints
            self.rng.shuffle(indices)
            x_shuffled, y_shuffled, weights_shuffled = x[indices], y[indices], weights[indices]
    
            # create folds
            x_folds = np.array_split(x_shuffled, K)
            y_folds = np.array_split(y_shuffled, K)
            weights_folds = np.array_split(weights_shuffled, K)

            residuals = []
            for k in range(K):
                # extract test points
                x_test, y_test, weights_test, = x_folds[k], y_folds[k], weights_folds[k]
                
                # extract training points
                x_train = np.concatenate([x_folds[j] for j in range(K) if j != k])
                y_train = np.concatenate([y_folds[j] for j in range(K) if j != k])
                weights_train = np.concatenate([weights_folds[j] for j in range(K) if j != k])

                # fit k-th model
                coefficients_k = self.fit(x_train, y_train, weights=weights_train, cache=False, **kwargs)
                y_hat_k = self.predict(x_test, coefficients=coefficients_k)

                # k-th residuals
                residuals_k = y_test - y_hat_k
                residuals.append(residuals_k)
            
            residuals = np.concatenate(residuals)
            # undo shuffling of residuals, to put them in the original dataset order
            residuals = residuals[np.argsort(indices)]

        if center:
            residuals -= np.mean(residuals, axis=0)

        return residuals
