import numpy as np
import pytest

from elexsolver.QuantileRegressionSolver import QuantileRegressionSolver

# relatively high tolerance, since different implementation.
TOL = 1e-3

# the outputs are compared against quantreg package in R

###############
# Basic tests #
###############


def test_basic_median_1():
    quantreg = QuantileRegressionSolver()
    tau = 0.5
    x = np.asarray([[1], [1], [1], [2]])
    y = np.asarray([3, 8, 9, 15])
    quantreg.fit(x, y, tau, fit_intercept=False)
    preds = quantreg.predict(x)
    # you'd think it would be 8 instead of 7.5, but run quantreg in R to confirm
    # has to do with missing intercept
    np.testing.assert_array_equal(preds, [[7.5, 7.5, 7.5, 15]])


def test_basic_median_2():
    quantreg = QuantileRegressionSolver()
    tau = 0.5
    x = np.asarray([[1, 1], [1, 1], [1, 1], [1, 2]])
    y = np.asarray([3, 8, 9, 15])
    quantreg.fit(x, y, tau)
    preds = quantreg.predict(x)
    np.testing.assert_array_equal(preds, [[8, 8, 8, 15]])


def test_basic_lower():
    quantreg = QuantileRegressionSolver()
    tau = 0.1
    x = np.asarray([[1, 1], [1, 1], [1, 1], [1, 2]])
    y = np.asarray([3, 8, 9, 15])
    quantreg.fit(x, y, tau)
    preds = quantreg.predict(x)
    np.testing.assert_array_equal(preds, [[3, 3, 3, 15]])


def test_basic_upper():
    quantreg = QuantileRegressionSolver()
    tau = 0.9
    x = np.asarray([[1, 1], [1, 1], [1, 1], [1, 2]])
    y = np.asarray([3, 8, 9, 15])
    quantreg.fit(x, y, tau)
    preds = quantreg.predict(x)
    np.testing.assert_array_equal(preds, [[9, 9, 9, 15]])


######################
# Intermediate tests #
######################


def test_random_median(random_data_no_weights):
    quantreg = QuantileRegressionSolver()
    tau = 0.5
    x = random_data_no_weights[["x0", "x1", "x2", "x3", "x4"]].values
    y = random_data_no_weights["y"].values
    quantreg.fit(x, y, tau, fit_intercept=False)
    quantreg.predict(x)
    np.testing.assert_allclose(quantreg.coefficients, [[1.57699, 6.74906, 4.40175, 4.85346, 4.51814]], rtol=TOL)

def test_random_lower(random_data_no_weights):
    quantreg = QuantileRegressionSolver()
    tau = 0.1
    x = random_data_no_weights[["x0", "x1", "x2", "x3", "x4"]].values
    y = random_data_no_weights["y"].values
    quantreg.fit(x, y, tau, fit_intercept=False)
    quantreg.predict(x)
    np.testing.assert_allclose(quantreg.coefficients, [[0.17759, 6.99588, 4.18896, 4.83906, 3.22546]], rtol=TOL)


def test_random_upper(random_data_no_weights):
    quantreg = QuantileRegressionSolver()
    tau = 0.9
    x = random_data_no_weights[["x0", "x1", "x2", "x3", "x4"]].values
    y = random_data_no_weights["y"].values
    quantreg.fit(x, y, tau, fit_intercept=False)
    quantreg.predict(x)
    np.testing.assert_allclose(quantreg.coefficients, [[1.85617, 6.81286, 6.05586, 5.51965, 4.19864]], rtol=TOL)


######################
# Tests with weights #
######################


def test_basic_median_weights():
    quantreg = QuantileRegressionSolver()
    tau = 0.5
    x = np.asarray([[1, 1], [1, 1], [1, 1], [1, 2]])
    y = np.asarray([3, 8, 9, 15])
    weights = np.asarray([1, 1, 100, 3])
    quantreg.fit(x, y, tau, weights)
    preds = quantreg.predict(x)
    np.testing.assert_array_equal(preds, [[9, 9, 9, 15]])


def test_random_median_weights(random_data_weights):
    quantreg = QuantileRegressionSolver()
    tau = 0.5
    x = random_data_weights[["x0", "x1", "x2", "x3", "x4"]].values
    y = random_data_weights["y"].values
    weights = random_data_weights["weights"].values
    quantreg.fit(x, y, tau, weights=weights, fit_intercept=False)
    quantreg.predict(x)
    np.testing.assert_allclose(quantreg.coefficients, [[1.59521, 2.17864, 4.68050, 3.10920, 9.63739]], rtol=TOL)


def test_random_lower_weights(random_data_weights):
    quantreg = QuantileRegressionSolver()
    tau = 0.1
    x = random_data_weights[["x0", "x1", "x2", "x3", "x4"]].values
    y = random_data_weights["y"].values
    weights = random_data_weights["weights"].values
    quantreg.fit(x, y, tau, weights=weights, fit_intercept=False)
    quantreg.predict(x)
    np.testing.assert_allclose(quantreg.coefficients, [[0.63670, 1.27028, 4.81500, 3.08055, 8.69929]], rtol=TOL)


def test_random_upper_weights(random_data_weights):
    quantreg = QuantileRegressionSolver()
    tau = 0.9
    x = random_data_weights[["x0", "x1", "x2", "x3", "x4"]].values
    y = random_data_weights["y"].values
    weights = random_data_weights["weights"].values
    quantreg.fit(x, y, tau, weights=weights, fit_intercept=False)
    quantreg.predict(x)
    np.testing.assert_allclose(quantreg.coefficients, [[3.47742, 2.07360, 4.51754, 4.15237, 9.58856]], rtol=TOL)


########################
# Test regularization #
########################


def test_regularization_with_intercept(random_data_no_weights):
    tau = 0.5
    x = random_data_no_weights[["x0", "x1", "x2", "x3", "x4"]].values
    x[:, 0] = 1
    y = random_data_no_weights["y"].values

    quantreg = QuantileRegressionSolver()
    lambda_ = 1e6
    quantreg.fit(x, y, tau, lambda_=lambda_, fit_intercept=True)
    coefficients_w_reg = quantreg.coefficients

    assert all(np.abs(coefficients_w_reg[1:] - [0, 0, 0, 0]) <= TOL)
    assert np.abs(coefficients_w_reg[0]) > TOL

    objective_w_reg = quantreg.problem.value
    quantreg.fit(x, y, tau, save_problem=True)
    assert quantreg.problem.value < objective_w_reg


def test_regularization_with_intercept_warning(random_data_no_weights, caplog):
    caplog.clear()
    tau = 0.5
    x = random_data_no_weights[["x0", "x1", "x2", "x3", "x4"]].values
    y = random_data_no_weights["y"].values

    quantreg = QuantileRegressionSolver()
    lambda_ = 1e6
    with pytest.warns(UserWarning):
        quantreg.fit(x, y, tau, lambda_=lambda_, fit_intercept=True)


def test_regularization_without_intercept(random_data_no_weights):
    tau = 0.5
    x = random_data_no_weights[["x0", "x1", "x2", "x3", "x4"]].values
    y = random_data_no_weights["y"].values

    quantreg = QuantileRegressionSolver()
    lambda_ = 1e6
    quantreg.fit(x, y, tau, lambda_=lambda_, fit_intercept=False)
    coefficients_w_reg = quantreg.coefficients
    assert all(np.abs(coefficients_w_reg - [0, 0, 0, 0, 0]) <= TOL)


########################
# Test checking matrix #
########################


def test_ill_conditioned_error():
    quantreg = QuantileRegressionSolver()

    x = np.asarray([[1, 0, 1], [4, 3, 4], [5, 2, 5]])
    with pytest.raises(IllConditionedMatrixException):
        quantreg._check_matrix_condition(x)


def test_ill_conditioned_warning():
    quantreg = QuantileRegressionSolver()

    random_number_generator = np.random.RandomState(42)
    mu = np.asarray([1, 3, 5])
    sigma = np.asarray([[1, 0.9, 0], [0.9, 1, 0], [0, 0, 1]])
    x = random_number_generator.multivariate_normal(mu, sigma, size=3)
    with pytest.warns(UserWarning):
        quantreg._check_matrix_condition(x)


########################
# Test checking NaN/Inf #
########################


def test_no_nan_inf_error(random_data_weights):
    quantreg = QuantileRegressionSolver()
    tau = 0.9
    x = random_data_weights[["x0", "x1", "x2", "x3", "x4"]].values
    y = random_data_weights["y"].values

    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        quantreg.fit(x, y, tau, fit_intercept=False)

    x[0, 0] = np.inf
    with pytest.raises(ValueError):
        quantreg.fit(x, y, tau, fit_intercept=False)

    x = random_data_weights[["x0", "x1", "x2", "x3", "x4"]].values
    y[5] = np.nan
    with pytest.raises(ValueError):
        quantreg.fit(x, y, tau, fit_intercept=False)

    y[5] = np.inf
    with pytest.raises(ValueError):
        quantreg.fit(x, y, tau, fit_intercept=False)

    quantreg.coefficients = [4, 32, 4, 24, 7]
    x = np.vstack([x, [4, 2, 6, np.nan, 3]])
    with pytest.raises(ValueError):
        quantreg.predict(x)

    x = np.vstack([x, [4, 2, 6, np.inf, 3]])
    with pytest.raises(ValueError):
        quantreg.predict(x)
