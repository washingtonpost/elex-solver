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

    @staticmethod
    def __get_constraint(coef, strict):
        if strict:
            return [0 <= coef, coef <= 1, cp.sum(coef, axis=1) == 1]
        return [cp.sum(coef, axis=1) <= 1.1, cp.sum(coef, axis=1) >= 0.9]

    def __solve(self, A, B, weights):
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
        Y_expected_totals = Y.sum(axis=0) / Y.sum(axis=0).sum()

        X = self._rescale(X)
        Y = self._rescale(Y)

        weights = self._check_and_prepare_weights(X, Y, weights)

        transition_matrix = self.__solve(X, Y, weights)
        transitions = np.diag(X_expected_totals) @ transition_matrix
        Y_pred_totals = np.sum(transitions, axis=0) / np.sum(transitions, axis=0).sum()
        self._mae = mean_absolute_error(Y_expected_totals, Y_pred_totals)
        LOG.info("MAE = %s", np.around(self._mae, 4))

        return transitions


class BootstrapTransitionMatrixSolver(TransitionSolver):
    def __init__(self, B=1000, strict=True):
        super().__init__()
        self._strict = strict
        self._B = B

    def fit_predict(self, X, Y, weights=None):
        maes = []
        predicted_transitions = []

        # assuming pandas.DataFrame
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        if not isinstance(Y, np.ndarray):
            Y = Y.to_numpy()
        # assuming pandas.Series
        if weights is not None and not isinstance(weights, np.ndarray):
            weights = weights.values

        tm = TransitionMatrixSolver(strict=self._strict)
        predicted_transitions.append(tm.fit_predict(X, Y, weights=weights))
        maes.append(tm.MAE)

        from sklearn.utils import resample  # to be replaced

        for b in range(0, self._B - 1):
            X_resampled = []
            Y_resampled = []
            weights_resampled = []
            for i in resample(range(0, len(X)), replace=True, random_state=b):
                X_resampled.append(X[i])
                Y_resampled.append(Y[i])
                if weights is not None:
                    weights_resampled.append(weights[i])
            if weights is None:
                weights_resampled = None
            else:
                weights_resampled = np.array(weights_resampled)
            predicted_transitions.append(
                tm.fit_predict(np.array(X_resampled), np.array(Y_resampled), weights=weights_resampled)
            )
            maes.append(tm.MAE)

        self._mae = np.mean(maes)
        return np.mean(predicted_transitions, axis=0)
