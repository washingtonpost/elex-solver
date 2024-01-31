import logging

import numpy as np
import pymc as pm

from elexsolver.logging import initialize_logging
from elexsolver.TransitionSolver import TransitionSolver

initialize_logging()

LOG = logging.getLogger(__name__)
logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("jax").setLevel(logging.ERROR)


class EITransitionSolver(TransitionSolver):
    """
    A transition solver based on RxC ecological inference.
    Somewhat adapted from version 1.0.1 of
    Knudson et al., (2021). PyEI: A Python package for ecological inference.
    Journal of Open Source Software, 6(64), 3397, https://doi.org/10.21105/joss.03397
    See also:
    Ori Rosen, Wenxin Jiang, Gary King, and Martin A Tanner. 2001.
    “Bayesian and Frequentist Inference for Ecological Inference: The RxC Case.”
    Statistica Neerlandica, 55, Pp. 134–156. Copy at https://tinyurl.com/yajkae6n
    """

    def __init__(self, sigma: int = 1, sampling_chains: int = 2, random_seed: int | None = None, n_samples: int = 300):
        """
        Parameters
        ----------
        `sigma` : int, default 1
            Standard deviation of the half-normal distribution that provides alphas to the Dirichlet distribution.
        `sampling_chains` : int, default 2
            The number of sampling chains to run in parallel, each of which will draw `n_samples`.
        `random_seed` : int, optional
            For seeding the NUTS sampler.
        `n_samples` : int, default 300
            The number of samples to draw.  Before sampling, the NUTS sampler will be tuned using `n_samples // 2` samples.
        """
        super().__init__()
        self._sigma = sigma
        self._chains = int(sampling_chains)
        self._seed = random_seed
        self._draws = n_samples
        self._tune = n_samples // 2

        # class members that are instantiated during model-fit
        self._sampled = None
        self._X_totals = None

    def fit_predict(self, X: np.ndarray, Y: np.ndarray, weights: np.ndarray | None = None):
        """
        NOTE: weighting is not currently implemented.
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

        self._X_totals = X.sum(axis=0) / X.sum(axis=0).sum()
        n = Y.sum(axis=1)

        num_units = len(n)  # should be the same as the number of units in Y
        num_rows = X.shape[1]  # number of things in X that are being transitioned "from"
        num_cols = Y.shape[1]  # number of things in Y that are being transitioned "to"

        # rescaling and reshaping
        X = self._rescale(X)
        X_extended = np.expand_dims(X, axis=2)
        X_extended = np.repeat(X_extended, num_cols, axis=2)

        with pm.Model(check_bounds=False) as model:
            conc_params = pm.HalfNormal("conc_params", sigma=self._sigma, shape=(num_rows, num_cols))
            beta = pm.Dirichlet("beta", a=conc_params, shape=(num_units, num_rows, num_cols))
            theta = (X_extended * beta).sum(axis=1)
            pm.Multinomial(
                "result_fractions",
                n=n,
                p=theta,
                observed=Y,
                shape=(num_units, num_cols),
            )
            try:
                # DO NOT USE THE NUMPYRO NUTS SAMPLER
                # IT IS UNSTABLE
                model_trace = pm.sample(
                    chains=self._chains,
                    random_seed=self._seed,
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
        samples_converted = np.transpose(b_values, axes=(3, 0, 1, 2)) * X
        samples_summed_across = samples_converted.sum(axis=2)
        self._sampled = np.transpose(samples_summed_across / X.sum(axis=0), axes=(1, 2, 0))

        posterior_mean_rxc = self._sampled.mean(axis=0)
        self._transitions = self.__get_transitions(posterior_mean_rxc)
        return posterior_mean_rxc

    def __get_transitions(self, A: np.ndarray):
        # to go from inferred percentages to transitions
        transitions = []
        for col in A.T:
            transitions.append(col * self._X_totals)
        return np.array(transitions).T

    def get_credible_interval(self, ci: float, transitions: bool = False):
        """
        Parameters
        ----------
        `ci` : float
            Size of the credible interval [0, 100).  If <= 1, will be multiplied by 100.
        `transitions` : bool, default False
            If True, the returned matrices will represent transitions, not percentages.

        Returns
        -------
        A tuple of two np.ndarray matrices of float.  Element 0 has the lower bound and 1 has the upper bound.
        """
        if ci <= 1:
            ci = ci * 100
        if ci < 0 or ci > 100:
            raise ValueError(f"Invalid credible interval {ci}.")

        lower = (100 - ci) / 2
        upper = ci + lower
        A_dict = {
            lower: np.zeros((self._sampled.shape[1], self._sampled.shape[2])),
            upper: np.zeros((self._sampled.shape[1], self._sampled.shape[2])),
        }

        for interval in [lower, upper]:
            for i in range(0, self._sampled.shape[1]):
                for j in range(0, self._sampled.shape[2]):
                    A_dict[interval][i][j] = np.percentile(self._sampled[:, i, j], interval)

        if transitions:
            return (self.__get_transitions(A_dict[lower]), self.__get_transitions(A_dict[upper]))
        return (A_dict[lower], A_dict[upper])
