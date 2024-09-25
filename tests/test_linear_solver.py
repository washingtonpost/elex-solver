import numpy as np
import pytest

from elexsolver.LinearSolver import LinearSolver
from elexsolver.QuantileRegressionSolver import QuantileRegressionSolver


def test_fit():
    solver = LinearSolver()
    with pytest.raises(NotImplementedError):
        solver.fit(np.ndarray((5, 3)), np.ndarray((1, 3)))


##################
# Test residuals #
##################
def test_residuals_without_weights(rng):
    x = rng.normal(size=(100, 5))
    beta = rng.normal(size=(5, 1))
    y = x @ beta

    # we need an a subclass of LinearSolver to actually run a fit
    reg = QuantileRegressionSolver()
    reg.fit(x, y, fit_intercept=False)
    reg.predict(x)

    reg.residuals(x, y, K=None, center=False)
    reg.residuals(x, y, K=10, center=False)
