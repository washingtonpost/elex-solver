import numpy as np
import pytest

from elexsolver.OLSRegressionSolver import OLSRegressionSolver

# relatively high tolerance, since different implementation.
TOL = 1e-3

# the outputs are compared against lm

###############
# Basic tests #
###############

def test_basic_median_1():
    lm = OLSRegressionSolver()
    x = np.asarray([[1], [1], [1], [2]])
    y = np.asarray([3, 8, 9, 15])
    lm.fit(x, y, fit_intercept=False)
    preds = lm.predict(x)
    assert all(np.abs(preds - [7.142857, 7.142857, 7.142857, 14.285714]) <= TOL)

def test_basic_median_2():
    lm = OLSRegressionSolver()
    x = np.asarray([[1, 1], [1, 1], [1, 1], [1, 2]])
    y = np.asarray([3, 8, 9, 15])
    lm.fit(x, y, fit_intercept=True)
    preds = lm.predict(x)
    assert all(np.abs(preds - [6.666667, 6.666667, 6.666667, 15]) <= TOL)

######################
# Intermediate tests #
######################


def test_random_median(random_data_no_weights):
    lm = OLSRegressionSolver()
    x = random_data_no_weights[["x0", "x1", "x2", "x3", "x4"]].values
    y = random_data_no_weights["y"].values.reshape(-1, 1)
    lm.fit(x, y, fit_intercept=False)
    lm.predict(x)
    assert all(np.abs(lm.coefficients - [[1.037], [7.022], [4.794], [4.776], [4.266]]) <= TOL)


######################
# Tests with weights #
######################

def test_random_median_weights(random_data_weights):
    lm = OLSRegressionSolver()
    tau = 0.5
    x = random_data_weights[["x0", "x1", "x2", "x3", "x4"]].values
    y = random_data_weights["y"].values.reshape(-1, 1)
    weights = random_data_weights["weights"].values
    lm.fit(x, y, weights=weights, fit_intercept=False)
    lm.predict(x)
    assert all(np.abs(lm.coefficients - [[1.455], [2.018], [4.699], [3.342], [9.669]]) <= TOL)