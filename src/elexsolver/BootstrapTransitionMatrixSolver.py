import logging

import numpy as np
from tqdm import tqdm

from elexsolver.logging import initialize_logging
from elexsolver.TransitionMatrixSolver import TransitionMatrixSolver
from elexsolver.TransitionSolver import TransitionSolver

initialize_logging()

LOG = logging.getLogger(__name__)


class BootstrapTransitionMatrixSolver(TransitionSolver):
    """
    Bootstrap version of the matrix regression transition solver.
    TODO: consider moving this into the same module as the TransitionMatrixSolver.
    """

    def __init__(self, B: int = 1000, strict: bool = True, verbose: bool = True, lam: int | None = None):
        """
        Parameters
        ----------
        `B` : int, default 1000
            Number of bootstrap samples to draw and matrix solver models to fit/predict.
        `strict` : bool, default True
            If `True`, solution will be constrainted so that all coefficients are >= 0,
            <= 1, and the sum of each row equals 1.
        `verbose` : bool, default True
            If `False`, this will reduce the amount of logging produced for each of the `B` bootstrap samples.
        `lam` : float, optional
            `lam` != 0 will enable L2 regularization (Ridge).
        """
        super().__init__()
        self._strict = strict
        self._B = B
        self._verbose = verbose
        self._lambda = lam

        # class members that are instantiated during model-fit
        self._predicted_percentages = None
        self._X_expected_totals = None

    def fit_predict(self, X: np.ndarray, Y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        self._predicted_percentages = []

        # assuming pandas.DataFrame
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        if not isinstance(Y, np.ndarray):
            Y = Y.to_numpy()

        self._X_expected_totals = X.sum(axis=0) / X.sum(axis=0).sum()

        tm = TransitionMatrixSolver(strict=self._strict, lam=self._lambda)
        self._predicted_percentages.append(tm.fit_predict(X, Y, weights=weights))

        for b in tqdm(range(0, self._B - 1), desc="Bootstrapping", disable=not self._verbose):
            rng = np.random.default_rng(seed=b)
            X_resampled = rng.choice(
                X, len(X), replace=True, axis=0, p=(weights / weights.sum() if weights is not None else None)
            )
            indices = [np.where((X == x).all(axis=1))[0][0] for x in X_resampled]
            Y_resampled = Y[indices]
            self._predicted_percentages.append(tm.fit_predict(X_resampled, Y_resampled, weights=None))

        percentages = np.mean(self._predicted_percentages, axis=0)
        self._transitions = np.diag(self._X_expected_totals) @ percentages
        return percentages

    def get_confidence_interval(self, alpha: float, transitions: bool = False) -> (np.ndarray, np.ndarray):
        """
        Parameters
        ----------
        `alpha` : float
            Value between [0, 1).  If greater than 1, will be divided by 100.
        `transitions` : bool, default False
            If True, the returned matrices will represent transitions, not percentages.

        Returns
        -------
        A tuple of two np.ndarray matrices of float.  Element 0 has the lower bound and 1 has the upper bound.
        """
        if alpha > 1:
            alpha = alpha / 100
        if alpha < 0 or alpha >= 1:
            raise ValueError(f"Invalid confidence interval {alpha}.")

        p_lower = ((1.0 - alpha) / 2.0) * 100
        p_upper = ((1.0 + alpha) / 2.0) * 100

        percentages = (
            np.percentile(self._predicted_percentages, p_lower, axis=0),
            np.percentile(self._predicted_percentages, p_upper, axis=0),
        )

        if transitions:
            return (
                np.diag(self._X_expected_totals) @ percentages[0],
                np.diag(self._X_expected_totals) @ percentages[1],
            )
        return percentages
