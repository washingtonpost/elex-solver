import logging

import numpy as np
import pymc as pm

from elexsolver.logging import initialize_logging
from elexsolver.TransitionSolver import TransitionSolver

initialize_logging()

LOG = logging.getLogger(__name__)


class EITransitionSolver(TransitionSolver):
    """
    A (voter) transition solver based on RxC ecological inference.
    Largely adapted from version 1.0.1 of
    Knudson et al., (2021). PyEI: A Python package for ecological inference.
    Journal of Open Source Software, 6(64), 3397, https://doi.org/10.21105/joss.03397
    """

    def __init__(self, n: np.ndarray, sigma=1, sampling_chains=2, random_seed=None, draws=300):
        super().__init__()
        self._n = n
        self._sigma = sigma
        self._chains = int(sampling_chains)
        self._seed = random_seed
        self._draws = draws
        self._tune = draws // 2

        # class members that are instantiated during model-fit
        self._sampled = None
        self._X_totals = None

    def mean_absolute_error(self, X, Y):
        y_pred = self._get_expected_totals(X)
        y = self._get_expected_totals(Y.T)
        absolute_errors = np.abs(y_pred - y)
        error_sum = np.sum(absolute_errors)
        mae = error_sum / len(absolute_errors)
        return mae

    def fit_predict(self, X, Y):
        self._check_any_element_nan_or_inf(X)
        self._check_any_element_nan_or_inf(Y)
        self._check_percentages(X)
        self._check_percentages(Y)

        # matrices should be (things x units), where the number of units is > the number of things
        if X.shape[0] > X.shape[1]:
            X = X.T
        if Y.shape[0] > Y.shape[1]:
            Y = Y.T

        if X.shape[1] != Y.shape[1]:
            raise ValueError(f"Number of units in X ({X.shape[1]}) != number of units in Y ({Y.shape[1]}).")
        if Y.shape[1] != len(self._n):
            raise ValueError(f"Number of units in Y ({Y.shape[1]}) != number of units in n ({len(self._n)}).")

        X = self._check_and_rescale(X)
        Y = self._check_and_rescale(Y)

        num_units = len(self._n)  # should be the same as the number of units in Y
        num_rows = X.shape[0]  # number of things in X that are being transitioned "from"
        num_cols = Y.shape[0]  # number of things in Y that are being transitioned "to"

        # reshaping and rounding
        Y_obs = np.transpose(Y * self._n).round()
        X_extended = np.expand_dims(X, axis=2)
        X_extended = np.repeat(X_extended, num_cols, axis=2)
        X_extended = np.swapaxes(X_extended, 0, 1)

        with pm.Model(check_bounds=False) as model:
            conc_params = pm.HalfNormal("conc_params", sigma=self._sigma, shape=(num_rows, num_cols))
            beta = pm.Dirichlet("beta", a=conc_params, shape=(num_units, num_rows, num_cols))
            theta = (X_extended * beta).sum(axis=1)
            pm.Multinomial(
                "result_fractions",
                n=self._n,
                p=theta,
                observed=Y_obs,
                shape=(num_units, num_cols),
            )
            try:
                # TODO: keep trying to tune this for performance and speed
                model_trace = pm.sample(
                    chains=self._chains,
                    random_seed=self._seed,
                    nuts_sampler="numpyro",
                    cores=self._chains,
                    draws=self._draws,
                    tune=self._tune,
                )
            except Exception as e:
                LOG.debug(model.debug())
                raise e

        b_values = np.transpose(
            model_trace["posterior"]["beta"].stack(all_draws=["chain", "draw"]).values, axes=(3, 0, 1, 2)
        )
        samples_converted = np.transpose(b_values, axes=(3, 0, 1, 2)) * X.T.values
        samples_summed_across = samples_converted.sum(axis=2)
        self._sampled = np.transpose(samples_summed_across / X.T.sum(axis=0).values, axes=(1, 2, 0))

        posterior_mean_rxc = self._sampled.mean(axis=0)
        self._X_totals = self._get_expected_totals(np.transpose(X))
        transitions = self._get_transitions(posterior_mean_rxc)
        LOG.info("MAE = %s", np.around(self.mean_absolute_error(transitions, Y), 4))
        return transitions

    def _get_transitions(self, A: np.ndarray):
        # to go from inferences to transitions
        transitions = []
        for col in A.T:
            transitions.append(col * self._X_totals)
        return np.array(transitions).T

    def get_prediction_interval(self, pi):
        """
        Note: this is actually a credible interval, not a prediction interval.
        """
        if pi <= 1:
            pi = pi * 100
        if pi < 0 or pi > 100:
            raise ValueError(f"Invalid prediction interval {pi}.")

        lower = (100 - pi) / 2
        upper = pi + lower
        A_dict = {
            lower: np.zeros((self._sampled.shape[1], self._sampled.shape[2])),
            upper: np.zeros((self._sampled.shape[1], self._sampled.shape[2])),
        }

        for ci in [lower, upper]:
            for i in range(0, self._sampled.shape[1]):
                for j in range(0, self._sampled.shape[2]):
                    A_dict[ci][i][j] = np.percentile(self._sampled[:, i, j], ci)

        return (self._get_transitions(A_dict[lower]), self._get_transitions(A_dict[upper]))
