import logging
import warnings

import cvxpy as cp
import numpy as np
from tqdm import tqdm

from elexsolver.logging import initialize_logging
from elexsolver.TransitionSolver import TransitionSolver

initialize_logging()

LOG = logging.getLogger(__name__)


class TransitionMatrixSolver(TransitionSolver):
    def __init__(self, strict=True, lam=None):
        """
        `lam` > 0 will enable L2 regularization (Ridge).
        """
        super().__init__()
        self._strict = strict
        self._lambda = lam

    @staticmethod
    def __get_constraints(coef: np.ndarray, strict: bool):
        if strict:
            return [0 <= coef, coef <= 1, cp.sum(coef, axis=1) == 1]
        return [cp.sum(coef, axis=1) <= 1.1, cp.sum(coef, axis=1) >= 0.9]

    def __standard_objective(self, A: np.ndarray, B: np.ndarray, beta: np.ndarray):
        loss_function = cp.norm(A @ beta - B, "fro")
        return cp.Minimize(loss_function)

    def __ridge_objective(self, A: np.ndarray, B: np.ndarray, beta: np.ndarray):
        # Based on https://www.cvxpy.org/examples/machine_learning/ridge_regression.html
        lam = cp.Parameter(nonneg=True, value=self._lambda)
        loss_function = cp.pnorm(A @ beta - B, p=2) ** 2
        regularizer = cp.pnorm(beta, p=2) ** 2
        return cp.Minimize(loss_function + lam * regularizer)

    def __solve(self, A: np.ndarray, B: np.ndarray, weights: np.ndarray):
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
                LOG.error(e)
                return np.zeros((A.shape[1], B.shape[1]))

        return transition_matrix.value

    def fit_predict(self, X: np.ndarray, Y: np.ndarray, weights: np.ndarray | None = None):
        """
        X and Y are matrixes (numpy or pandas.DataFrame) of integers.
        weights is a list, numpy array, or pandas.Series with the same length as both X and Y.
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

        X = self._rescale(X)
        Y = self._rescale(Y)

        weights = self._check_and_prepare_weights(X, Y, weights)

        percentages = self.__solve(X, Y, weights)
        self._transitions = np.diag(X_expected_totals) @ percentages
        return percentages


class BootstrapTransitionMatrixSolver(TransitionSolver):
    def __init__(self, B: int = 1000, strict: bool = True, verbose: bool = True, lam: int | None = None):
        super().__init__()
        self._strict = strict
        self._B = B
        self._verbose = verbose
        self._lambda = lam

        # class members that are instantiated during model-fit
        self._predicted_percentages = None

    def fit_predict(self, X: np.ndarray, Y: np.ndarray, weights: np.ndarray | None = None):
        self._predicted_percentages = []
        predicted_transitions = []

        # assuming pandas.DataFrame
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        if not isinstance(Y, np.ndarray):
            Y = Y.to_numpy()

        tm = TransitionMatrixSolver(strict=self._strict, lam=self._lambda)
        self._predicted_percentages.append(tm.fit_predict(X, Y, weights=weights))
        predicted_transitions.append(tm.transitions)

        for b in tqdm(range(0, self._B - 1), desc="Bootstrapping", disable=not self._verbose):
            rng = np.random.default_rng(seed=b)
            X_resampled = rng.choice(
                X, len(X), replace=True, axis=0, p=(weights / weights.sum() if weights is not None else None)
            )
            indices = [np.where((X == x).all(axis=1))[0][0] for x in X_resampled]
            Y_resampled = Y[indices]
            self._predicted_percentages.append(tm.fit_predict(X_resampled, Y_resampled, weights=None))
            predicted_transitions.append(tm.transitions)

        self._transitions = np.mean(predicted_transitions, axis=0)
        return np.mean(self._predicted_percentages, axis=0)

    def get_confidence_interval(self, alpha: float):
        # TODO: option to get this in transition form
        if alpha > 1:
            alpha = alpha / 100
        if alpha < 0 or alpha >= 1:
            raise ValueError(f"Invalid confidence interval {alpha}.")

        p_lower = ((1.0 - alpha) / 2.0) * 100
        p_upper = ((1.0 + alpha) / 2.0) * 100
        return (
            np.percentile(self._predicted_percentages, p_lower, axis=0),
            np.percentile(self._predicted_percentages, p_upper, axis=0),
        )
