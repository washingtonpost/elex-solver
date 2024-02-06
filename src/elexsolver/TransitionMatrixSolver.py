import logging
import warnings

import cvxpy as cp
import numpy as np

from elexsolver.logging import initialize_logging
from elexsolver.TransitionSolver import TransitionSolver

initialize_logging()

LOG = logging.getLogger(__name__)


class TransitionMatrixSolver(TransitionSolver):
    """
    Matrix regression transition solver using CVXPY.
    """

    def __init__(self, strict: bool = True, lam: float | None = None):
        """
        Parameters
        ----------
        `strict` : bool, default True
            If True, solution will be constrainted so that all coefficients are >= 0,
            <= 1, and the sum of each row equals 1.
        `lam` : float, optional
            `lam` != 0 will enable L2 regularization (Ridge).
        """
        super().__init__()
        self._strict = strict
        self._lambda = lam

    @staticmethod
    def __get_constraints(coef: np.ndarray, strict: bool) -> list:
        if strict:
            return [0 <= coef, coef <= 1, cp.sum(coef, axis=1) == 1]
        return [cp.sum(coef, axis=1) <= 1.1, cp.sum(coef, axis=1) >= 0.9]

    def __standard_objective(self, A: np.ndarray, B: np.ndarray, beta: np.ndarray) -> cp.Minimize:
        loss_function = cp.norm(A @ beta - B, "fro")
        return cp.Minimize(loss_function)

    def __ridge_objective(self, A: np.ndarray, B: np.ndarray, beta: np.ndarray) -> cp.Minimize:
        # Based on https://www.cvxpy.org/examples/machine_learning/ridge_regression.html
        lam = cp.Parameter(nonneg=True, value=self._lambda)
        loss_function = cp.pnorm(A @ beta - B, p=2) ** 2
        regularizer = cp.pnorm(beta, p=2) ** 2
        return cp.Minimize(loss_function + lam * regularizer)

    def __solve(self, A: np.ndarray, B: np.ndarray, weights: np.ndarray) -> np.ndarray:
        transition_matrix = cp.Variable((A.shape[1], B.shape[1]), pos=True)
        Aw = np.dot(weights, A)
        Bw = np.dot(weights, B)

        if self._lambda is None or self._lambda == 0:
            objective = self.__standard_objective(Aw, Bw, transition_matrix)
        else:
            objective = self.__ridge_objective(Aw, Bw, transition_matrix)

        constraints = TransitionMatrixSolver.__get_constraints(transition_matrix, self._strict)
        problem = cp.Problem(objective, constraints)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                problem.solve(solver=cp.CLARABEL)
            except (UserWarning, cp.error.SolverError) as e:
                raise RuntimeError(e) from e

        return transition_matrix.value

    def fit_predict(self, X: np.ndarray, Y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        self._check_data_type(X)
        self._check_data_type(Y)
        self._check_any_element_nan_or_inf(X)
        self._check_any_element_nan_or_inf(Y)

        # matrices should be (units x things), where the number of units is > the number of things
        if X.shape[1] > X.shape[0]:
            X = X.T
        if Y.shape[1] > Y.shape[0]:
            Y = Y.T

        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Number of units in X ({X.shape[0]}) != number of units in Y ({Y.shape[0]}).")

        self._check_dimensions(X)
        self._check_dimensions(Y)
        self._check_for_zero_units(X)
        self._check_for_zero_units(Y)

        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        if not isinstance(Y, np.ndarray):
            Y = Y.to_numpy()

        X_expected_totals = X.sum(axis=0) / X.sum(axis=0).sum()

        X = self._rescale(X)
        Y = self._rescale(Y)

        weights = self._check_and_prepare_weights(X, Y, weights)

        percentages = self.__solve(X, Y, weights)
        self._transitions = np.diag(X_expected_totals) @ percentages
        return percentages
