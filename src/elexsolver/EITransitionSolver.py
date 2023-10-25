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
    Somewhat adapted from version 1.0.1 of
    Knudson et al., (2021). PyEI: A Python package for ecological inference.
    Journal of Open Source Software, 6(64), 3397, https://doi.org/10.21105/joss.03397
    See also:
    Ori Rosen, Wenxin Jiang, Gary King, and Martin A Tanner. 2001.
    “Bayesian and Frequentist Inference for Ecological Inference: The RxC Case.”
    Statistica Neerlandica, 55, Pp. 134–156. Copy at https://tinyurl.com/yajkae6n
    """

    def __init__(self, sigma=1, sampling_chains=2, random_seed=None, n_samples=300):
        super().__init__()
        self._sigma = sigma
        self._chains = int(sampling_chains)
        self._seed = random_seed
        self._draws = n_samples
        self._tune = n_samples // 2

        # class members that are instantiated during model-fit
        self._sampled = None
        self._X_totals = None

    def fit_predict(self, X, Y):
        """
        X and Y are matrixes of integers.
        """
        self._check_data_type(X)
        self._check_data_type(Y)
        self._check_any_element_nan_or_inf(X)
        self._check_any_element_nan_or_inf(Y)

        # matrices should be (things x units), where the number of units is > the number of things
        if X.shape[0] > X.shape[1]:
            X = X.T
        if Y.shape[0] > Y.shape[1]:
            Y = Y.T

        self._check_dimensions(X)
        self._check_dimensions(Y)

        if X.shape[1] != Y.shape[1]:
            raise ValueError(f"Number of units in X ({X.shape[1]}) != number of units in Y ({Y.shape[1]}).")

        self._X_totals = X.sum(axis=1) / X.sum(axis=1).sum()
        Y_expected_totals = Y.sum(axis=1) / Y.sum(axis=1).sum()
        n = Y.sum(axis=0)

        X = self._rescale(X)
        Y = self._rescale(Y)

        num_units = len(n)  # should be the same as the number of units in Y
        num_rows = X.shape[0]  # number of things in X that are being transitioned "from"
        num_cols = Y.shape[0]  # number of things in Y that are being transitioned "to"

        # reshaping and rounding
        Y_obs = np.transpose(Y * n).round()
        X_extended = np.expand_dims(X, axis=2)
        X_extended = np.repeat(X_extended, num_cols, axis=2)
        X_extended = np.swapaxes(X_extended, 0, 1)

        with pm.Model(check_bounds=False) as model:
            conc_params = pm.HalfNormal("conc_params", sigma=self._sigma, shape=(num_rows, num_cols))
            beta = pm.Dirichlet("beta", a=conc_params, shape=(num_units, num_rows, num_cols))
            theta = (X_extended * beta).sum(axis=1)
            pm.Multinomial(
                "result_fractions",
                n=n,
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
        transitions = self._get_transitions(posterior_mean_rxc)
        Y_pred_totals = np.sum(transitions, axis=0) / np.sum(transitions, axis=0).sum()
        LOG.info("MAE = %s", np.around(self.mean_absolute_error(Y_pred_totals, Y_expected_totals), 4))
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
