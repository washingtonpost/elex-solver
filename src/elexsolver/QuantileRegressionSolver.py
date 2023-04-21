import logging
import warnings

import cvxpy as cp
import numpy as np

from elexsolver.logging import initialize_logging

initialize_logging()

LOG = logging.getLogger(__name__)


class QuantileRegressionSolverException(Exception):
    pass


class IllConditionedMatrixException(QuantileRegressionSolverException):
    pass


class QuantileRegressionSolver:

    VALID_SOLVERS = {"SCS", "ECOS", "MOSEK", "OSQP", "CVXOPT", "GLPK"}
    KWARGS = {"ECOS": {"max_iters": 10000}}

    CONDITION_WARNING_MIN = 50  # arbitrary
    CONDITION_ERROR_MIN = 1e8  # based on scipy

    def __init__(self, solver="ECOS"):
        if solver not in self.VALID_SOLVERS:
            raise ValueError(f"solver must be in {self.VALID_SOLVERS}")
        self.tau = cp.Parameter()
        self.coefficients = None
        self.problem = None
        self.solver = solver

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

    def get_loss_function(self, x, y, coefficients, weights):
        """
        Get the quantile regression loss function
        """
        y_hat = x @ coefficients
        residual = y - y_hat
        return cp.sum(cp.multiply(weights, 0.5 * cp.abs(residual) + (self.tau.value - 0.5) * residual))

    def get_regularizer(self, coefficients, fit_intercept):
        """
        Get regularization component of the loss function. Note that this is L2 (ridge) regularization.
        """
        # if we are fitting an intercept in the model, then that coefficient should not be regularized.
        # NOTE: assumes that if fit_intercept=True, that the intercept is in the first column
        coefficients_to_regularize = coefficients
        if fit_intercept:
            coefficients_to_regularize = coefficients[1:]
        return cp.pnorm(coefficients_to_regularize, p=2) ** 2

    def __solve(self, x, y, weights, lambda_, fit_intercept, verbose):
        """
        Sets up the optimization problem and solves it
        """
        self._check_matrix_condition(x)
        coefficients = cp.Variable((x.shape[1],))
        loss_function = self.get_loss_function(x, y, coefficients, weights)
        loss_function += lambda_ * self.get_regularizer(coefficients, fit_intercept)
        objective = cp.Minimize(loss_function)
        problem = cp.Problem(objective)
        problem.solve(solver=self.solver, verbose=verbose, **self.KWARGS.get(self.solver, {}))
        return coefficients, problem

    def fit(
        self,
        x,
        y,
        tau_value=0.5,
        weights=None,
        lambda_=0,
        fit_intercept=True,
        verbose=False,
        save_problem=False,
        normalize_weights=True,
    ):
        """
        Fit the (weighted) quantile regression problem.
        Weights should not sum to one.
        If fit_intercept=True then intercept is assumed to be the first column in `x`
        """

        self._check_any_element_nan_or_inf(x)
        self._check_any_element_nan_or_inf(y)

        if fit_intercept:
            self._check_intercept(x)

        if weights is None:  # if weights are none, give unit weights
            weights = [1] * x.shape[0]
        if normalize_weights:
            weights_sum = np.sum(weights)
            if weights_sum == 0:
                # This should not happen
                raise ZeroDivisionError
            weights = weights / weights_sum

        self.tau.value = tau_value
        coefficients, problem = self.__solve(x, y, weights, lambda_, fit_intercept, verbose)
        self.coefficients = coefficients.value
        if save_problem:
            self.problem = problem
        else:
            self.problem = None

    def predict(self, x):
        """
        Returns predictions
        """
        self._check_any_element_nan_or_inf(x)

        return self.coefficients @ x.T
