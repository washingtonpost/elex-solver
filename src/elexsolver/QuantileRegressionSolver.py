import logging

from scipy.optimize import linprog
import cvxpy as cp
import numpy as np

from elexsolver.LinearSolver import LinearSolver
from elexsolver.logging import initialize_logging

initialize_logging()

LOG = logging.getLogger(__name__)


class QuantileRegressionSolver(LinearSolver):
    
    def __init__(self):
        super().__init__()
        self.coefficients = []

    def _fit(
        self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, tau: float
    ) -> np.ndarray:
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

    def _fit_with_regularization(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, tau: float, lambda_: float):
        S = cp.Constant(y.reshape(-1, 1))
        N = y.shape[0]
        Phi = cp.Constant(x)
        radius = 1 / lambda_
        gamma = cp.Variable()
        C = radius / (N + 1)
        eta = cp.Variable(name="weights", shape=N)
        constraints = [
            C * (tau - 1) <= eta,
            C * tau >= eta,
            eta.T @ Phi == gamma
        ]
        prob = cp.Problem(
            cp.Minimize(0.5 * cp.sum_squares(Phi.T @ eta) - cp.sum(cp.multiply(eta, cp.vec(S)))),
            constraints
        )
        prob.solve()

        return prob.constraints[-1].dual_value
        """
    S -> scores
    tau -> quantile
    eta -> optimization_variables

        eta = cp.Variable(name="weights", shape=n_calib)
        
        scores = cp.Constant(scores_calib.reshape(-1,1))
    
        Phi = cp.Constant(phi_calib)

            radius = 1 / infinite_params.get('lambda', FUNCTION_DEFAULTS['lambda'])
    
        C = radius / (n_calib + 1)

        constraints = [
            C * (quantile - 1) <= eta,
            C * quantile >= eta,
            eta.T @ Phi == 0]
        prob = cp.Problem(
                    cp.Minimize(0.5 * cp.sum_squares(eta) - cp.sum(cp.multiply(eta, cp.vec(scores)))),
                    constraints
                )
    
        coefficients = prob.constraints[-1].dual_value
    """

    def fit(self, x: np.ndarray, y: np.ndarray, taus: list | float = 0.5, weights: np.ndarray | None = None, lambda_: float = 0.0, fit_intercept: bool = True) -> np.ndarray:
        """
        Fits quantile regression
        """
        self._check_any_element_nan_or_inf(x)
        self._check_any_element_nan_or_inf(y)
        
        if fit_intercept:
            self._check_intercept(x)

        # if weights are none, all rows should be weighted equally
        if weights is None:
            weights = np.ones((y.shape[0], ))

        # normalize weights
        weights = weights / np.sum(weights)

        # _fit assumes that taus is list, so if we want to do one value of tau then turn into a list
        if isinstance(taus, float):
            taus = [taus]

        for tau in taus:
            if lambda_ > 0:
                coefficients = self._fit_with_regularization(x, y, weights, tau, lambda_)
            else:
                coefficients = self._fit(x, y, weights, tau)
            self.coefficients.append(coefficients)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Use coefficients to predict
        """
        self._check_any_element_nan_or_inf(x)

        return self.coefficients @ x.T