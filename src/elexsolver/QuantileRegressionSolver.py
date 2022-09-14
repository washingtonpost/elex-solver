import logging

import cvxpy as cp
import numpy as np

from elexsolver.logging import initialize_logging

initialize_logging()

LOG = logging.getLogger(__name__)

class QuantileRegressionSolverException(Exception):
    pass

class IllConditionedMatrixException(QuantileRegressionSolverException):
    pass

class QuantileRegressionSolver():

    VALID_SOLVERS = {'SCS', 'ECOS', 'MOSEK', 'OSQP', 'CVXOPT', 'GLPK'}
    KWARGS = {
        "ECOS": {
            "max_iters": 10000
        }
    }

    CONDITION_WARNING_MIN = 50 # arbitrary
    CONDITION_ERROR_MIN = 1e+8 # based on scipy

    def __init__(self, solver='ECOS'):
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
            LOG.warn(f"Ill-conditioned matrix detected. result is not guaranteed to be accurate")
            return False
        return True

    def __solve(self, x, y, weights, verbose):
        """
        Sets up the optimization problem and solves it
        """
        self._check_matrix_condition(x)
        coefficients = cp.Variable((x.shape[1], ))
        y_hat = x @ coefficients
        residual = y - y_hat
        loss_function = cp.sum(cp.multiply(weights, 0.5 * cp.abs(residual) + (self.tau.value - 0.5) * residual))
        objective = cp.Minimize(loss_function)
        problem = cp.Problem(objective)
        problem.solve(solver=self.solver, verbose=verbose, **self.KWARGS.get(self.solver, {}))
        return coefficients, problem

    def fit(self, x, y, tau_value=0.5, weights=None, verbose=False, save_problem=False, normalize_weights=True):
        """
        Fit the (weighted) quantile regression problem.
        Weights should not sum to one.
        """
        if weights is None: # if weights are none, give unit weights
            weights = [1] * x.shape[0]
        if normalize_weights:
            weights_sum = np.sum(weights)
            if weights_sum == 0:
                # This should not happen
                raise ZeroDivisionError
            weights = weights / weights_sum
        
        self.tau.value = tau_value
        coefficients, problem = self.__solve(x, y, weights, verbose)
        self.coefficients = coefficients.value
        if save_problem:
            self.problem = problem
        else:
            self.problem = None

    def predict(self, x):
        """
        Returns predictions
        """
        return self.coefficients @ x.T
