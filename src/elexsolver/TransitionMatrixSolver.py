import logging

import cvxpy as cp
import numpy as np

from elexsolver.logging import initialize_logging
from elexsolver.TransitionSolver import TransitionSolver, mean_absolute_error

initialize_logging()

LOG = logging.getLogger(__name__)


class TransitionMatrixSolver(TransitionSolver):
    def __init__(self, strict=True):
        super().__init__()
        self._strict = strict

        # class members that are instantiated during model-fit
        # for bootstrapping
        self._residuals = None
        self._X = None
        self._Y = None
        self._X_expected_totals = None
        self._Y_expected_totals = None

    @staticmethod
    def __get_constraint(coef, strict):
        if strict:
            return [0 <= coef, coef <= 1, cp.sum(coef, axis=1) == 1]
        return [cp.sum(coef, axis=1) <= 1.1, cp.sum(coef, axis=1) >= 0.9]

    def _solve(self, A, B, weights):
        transition_matrix = cp.Variable((A.shape[1], B.shape[1]), pos=True)
        Aw = np.dot(weights, A)
        Bw = np.dot(weights, B)
        loss_function = cp.norm(Aw @ transition_matrix - Bw, "fro")
        objective = cp.Minimize(loss_function)
        constraint = TransitionMatrixSolver.__get_constraint(transition_matrix, self._strict)
        problem = cp.Problem(objective, constraint)
        problem.solve(solver=cp.CLARABEL)
        return transition_matrix.value

    def fit_predict(self, X, Y, weights=None):
        """
        X and Y are matrixes of integers.
        weights is a list or numpy array with the same length as both X and Y.
        """
        self._check_data_type(X)
        self._check_data_type(Y)
        self._check_any_element_nan_or_inf(X)
        self._check_any_element_nan_or_inf(Y)

        # matrices should be (units x things), where the number of units is > the number of things
        if X.shape[1] > X.shape[0]:
            X = X.T
        if Y.shape[1] > Y.shape[0]:
            Y = Y.T

        self._check_dimensions(X)
        self._check_dimensions(Y)
        self._check_for_zero_units(X)
        self._check_for_zero_units(Y)

        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        if not isinstance(Y, np.ndarray):
            Y = Y.to_numpy()

        self._X_expected_totals = X.sum(axis=0) / X.sum(axis=0).sum()
        self._Y_expected_totals = Y.sum(axis=0) / Y.sum(axis=0).sum()

        self._X = self._rescale(X)
        self._Y = self._rescale(Y)

        weights = self._check_and_prepare_weights(self._X, self._Y, weights)

        transition_matrix = self._solve(self._X, self._Y, weights)
        transitions = np.diag(self._X_expected_totals) @ transition_matrix
        Y_pred_totals = np.sum(transitions, axis=0) / np.sum(transitions, axis=0).sum()
        self._mae = mean_absolute_error(self._Y_expected_totals, Y_pred_totals)
        LOG.info("MAE = %s", np.around(self._mae, 4))
        self._residuals = Y_pred_totals - self._Y_expected_totals

        return transitions


class BootstrapTransitionMatrixSolver(TransitionSolver):
    def __init__(self, B=1000, strict=True):
        super().__init__()
        self._strict = strict
        self._B = B

    def _constrained_random_numbers(self, n, M, seed=None):
        """
        Generate n random numbers that sum to M.
        Based on: https://stackoverflow.com/a/30659457/224912
        """
        rng = np.random.default_rng(seed=seed)
        splits = [0] + [rng.random() for _ in range(0, n - 1)] + [1]
        splits.sort()
        diffs = [x - splits[i - 1] for (i, x) in enumerate(splits)][1:]
        result = list(map(lambda x: x * M, diffs))
        rng.shuffle(result)
        return result

    def fit_predict(self, X, Y):
        tm = TransitionMatrixSolver(strict=self._strict)
        transitions = tm.fit_predict(X, Y)

        from sklearn.utils import resample  # to be replaced

        maes = []

        for b in range(0, self._B):
            residuals_hat = resample(tm._residuals, replace=True, random_state=b)
            Y_hat = tm._Y.copy()
            for j in range(0, Y_hat.shape[1]):
                residuals_j = self._constrained_random_numbers(len(Y_hat), residuals_hat[j], seed=j)
                Y_hat[:, j] = Y_hat[:, j] + residuals_j

            transition_matrix_hat = tm._solve(tm._X, Y_hat)
            transitions_hat = np.diag(tm._X_expected_totals) @ transition_matrix_hat
            transitions = transitions + transitions_hat

            Y_pred_totals = np.sum(transitions_hat, axis=0) / np.sum(transitions_hat, axis=0).sum()
            this_mae = mean_absolute_error(tm._Y_expected_totals, Y_pred_totals)
            maes.append(this_mae)
            LOG.info("MAE = %s", np.around(this_mae, 4))

        self._mae = np.mean(maes)
        return transitions / self._B
