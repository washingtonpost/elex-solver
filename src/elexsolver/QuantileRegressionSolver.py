import logging

import cvxpy as cp
import numpy as np
from scipy.optimize import linprog

from elexsolver.LinearSolver import LinearSolver
from elexsolver.logging import initialize_logging

initialize_logging()

LOG = logging.getLogger(__name__)


class QuantileRegressionSolver(LinearSolver):
    def __init__(self):
        super().__init__()
        self.coefficients = []

    def _fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, tau: float) -> np.ndarray:
        """
        Fits the dual problem of a quantile regression, for more information see appendix 6 here: https://arxiv.org/pdf/2305.12616.pdf
        """
        S = y
        Phi = x
        zeros = np.zeros((Phi.shape[1],))
        N = y.shape[0]
        bounds = weights.reshape(-1, 1) * np.asarray([(tau - 1, tau)] * N)

        # A_eq are the equality constraint matrix
        # b_eq is the equality constraint vector (ie. A_eq @ x = b_eq)
        # bounds are the (min, max) possible values of every element of x
        res = linprog(-1 * S, A_eq=Phi.T, b_eq=zeros, bounds=bounds, method="highs", options={"presolve": False})
        # marginal are the dual values, since we are solving the dual this is equivalent to the primal
        return -1 * res.eqlin.marginals

    def _get_regularizer(
        self, coefficients: cp.expressions.variable.Variable, regularize_intercept: bool, n_feat_ignore_reg: int
    ) -> cp.atoms.norm:
        """
        Get regularization component of the loss function. Note that this is L2 (ridge) regularization.
        """
        # this assumes that if regularize_intercept=True that the intercept is the first column
        # also note that even if regularize_intercept is True BUT n_feat_ignore_req > 0 and fit_intercept
        # is true that we are NOT regularizing the intercept
        coefficients_to_regularize = coefficients[n_feat_ignore_reg:]
        if not regularize_intercept:
            coefficients_to_regularize = coefficients[1 + n_feat_ignore_reg :]  # noqa: E203
        return cp.pnorm(coefficients_to_regularize, p=2) ** 2

    def _fit_with_regularization(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        tau: float,
        lambda_: float,
        regularize_intercept: bool,
        n_feat_ignore_reg: int,
    ):
        """
        Fits quantile regression with regularization
        TODO: convert this problem to use the dual like in the non regularization case
        """
        arguments = {"ECOS": {"max_iters": 10000}}
        coefficients = cp.Variable((x.shape[1],))
        y_hat = x @ coefficients
        residual = y - y_hat
        loss_function = cp.sum(cp.multiply(weights, 0.5 * cp.abs(residual) + (tau - 0.5) * residual))
        loss_function += lambda_ * self._get_regularizer(coefficients, regularize_intercept, n_feat_ignore_reg)
        objective = cp.Minimize(loss_function)
        problem = cp.Problem(objective)
        problem.solve(solver="ECOS", **arguments.get("ECOS", {}))
        return coefficients.value

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        taus: list | float = 0.5,
        weights: np.ndarray | None = None,
        lambda_: float = 0.0,
        fit_intercept: bool = True,
        regularize_intercept: bool = False,
        n_feat_ignore_reg: int = 0,
        normalize_weights: bool = True,
    ):
        """
        Fits quantile regression
        """
        self._check_any_element_nan_or_inf(x)
        self._check_any_element_nan_or_inf(y)

        if fit_intercept:
            self._check_intercept(x)

        # if weights are none, all rows should be weighted equally
        if weights is None:
            weights = np.ones((y.shape[0],))

        # normalize weights. default to true
        if normalize_weights:
            weights_sum = np.sum(weights)
            if weights_sum == 0:
                raise ZeroDivisionError
            weights = weights / weights_sum

        # _fit assumes that taus is list, so if we want to do one value of tau then turn into a list
        if isinstance(taus, float):
            taus = [taus]

        for tau in taus:
            if lambda_ > 0:
                coefficients = self._fit_with_regularization(
                    x, y, weights, tau, lambda_, regularize_intercept, n_feat_ignore_reg
                )
            else:
                coefficients = self._fit(x, y, weights, tau)
            self.coefficients.append(coefficients)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Use coefficients to predict
        """
        self._check_any_element_nan_or_inf(x)

        return self.coefficients @ x.T
