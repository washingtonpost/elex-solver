import logging

import cvxpy as cp
import numpy as np

from elexsolver.logging import initialize_logging
from elexsolver.TransitionSolver import TransitionSolver

initialize_logging()

LOG = logging.getLogger(__name__)


class TransitionMatrixSolver(TransitionSolver):
    def __init__(self, strict=True):
        super().__init__()
        self._transition_matrix = None
        self._strict = strict

    @staticmethod
    def __get_constraint(coef, strict):
        if strict:
            return [0 <= coef, coef <= 1, cp.sum(coef, axis=1) == 1]
        return [cp.sum(coef, axis=1) <= 1.1, cp.sum(coef, axis=1) >= 0.9]

    def __solve(self, A, B):
        transition_matrix = cp.Variable((A.shape[1], B.shape[1]))
        loss_function = cp.norm(A @ transition_matrix - B, "fro")
        objective = cp.Minimize(loss_function)
        constraint = TransitionMatrixSolver.__get_constraint(transition_matrix, self._strict)
        problem = cp.Problem(objective, constraint)
        # preferring cvxpy's prior default solver, ECOS, over its new default, Clarabel
        # because sometimes Clarabel produces negative-valued results for our problem
        problem.solve(solver=cp.ECOS)
        return transition_matrix.value

    def fit_predict(self, X, Y):
        """
        X and Y are matrixes of integers.
        """
        self._check_any_element_nan_or_inf(X)
        self._check_any_element_nan_or_inf(Y)

        # matrices should be (units x things), where the number of units is > the number of things
        if X.shape[1] > X.shape[0]:
            X = X.T
        if Y.shape[1] > Y.shape[0]:
            Y = Y.T

        self._check_dimensions(X.T)
        self._check_dimensions(Y.T)

        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        if not isinstance(Y, np.ndarray):
            Y = Y.to_numpy()

        X_expected_totals = X.sum(axis=0) / X.sum(axis=0).sum()
        Y_expected_totals = Y.sum(axis=0) / Y.sum(axis=0).sum()

        X = self._rescale(X.T).T
        Y = self._rescale(Y.T).T

        self._transition_matrix = self.__solve(X, Y)
        transitions = np.diag(X_expected_totals) @ self._transition_matrix
        Y_pred_totals = np.sum(transitions, axis=0) / np.sum(transitions, axis=0).sum()
        LOG.info("MAE = %s", np.around(self.mean_absolute_error(Y_expected_totals, Y_pred_totals), 4))

        return transitions
