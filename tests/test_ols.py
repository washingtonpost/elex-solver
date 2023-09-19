import numpy as np
import pytest

from elexsolver.OLSRegressionSolver import OLSRegressionSolver

# relatively high tolerance, since different implementation.
TOL = 1e-3

# the outputs are compared against lm

###############
# Basic tests #
###############


def test_basic1():
    lm = OLSRegressionSolver()
    x = np.asarray([[1], [1], [1], [2]])
    y = np.asarray([3, 8, 9, 15])
    lm.fit(x, y, fit_intercept=False)
    preds = lm.predict(x)
    assert all(np.abs(preds - [7.142857, 7.142857, 7.142857, 14.285714]) <= TOL)


def test_basic2():
    lm = OLSRegressionSolver()
    x = np.asarray([[1, 1], [1, 1], [1, 1], [1, 2]])
    y = np.asarray([3, 8, 9, 15])
    lm.fit(x, y, fit_intercept=True)
    preds = lm.predict(x)
    assert all(np.abs(preds - [6.666667, 6.666667, 6.666667, 15]) <= TOL)


######################
# Intermediate tests #
######################


def test_random_no_intercept(random_data_no_weights):
    lm = OLSRegressionSolver()
    x = random_data_no_weights[["x0", "x1", "x2", "x3", "x4"]].values
    y = random_data_no_weights["y"].values.reshape(-1, 1)
    lm.fit(x, y, fit_intercept=False)
    predictions = lm.predict(x)
    # check coefficients
    assert all(np.abs(lm.coefficients - [[1.037], [7.022], [4.794], [4.776], [4.266]]) <= TOL)

    # check hat values
    assert len(lm.hat_vals) == 100
    assert lm.hat_vals[0] == pytest.approx(0.037102687)
    assert lm.hat_vals[-1] == pytest.approx(0.038703403)

    # check predictions
    assert predictions[0] == pytest.approx(10.743149)
    assert predictions[-1] == pytest.approx(9.878154)


def test_random_intercept(random_data_no_weights):
    lm = OLSRegressionSolver()
    random_data_no_weights["intercept"] = 1
    x = random_data_no_weights[["intercept", "x0", "x1", "x2", "x3", "x4"]].values
    y = random_data_no_weights["y"].values.reshape(-1, 1)
    lm.fit(x, y, fit_intercept=True)
    predictions = lm.predict(x)
    # check coefficients
    assert all(np.abs(lm.coefficients - [[0.08111], [1.01166], [6.98917], [4.77003], [4.73864], [4.23325]]) <= TOL)

    # check hat values
    assert len(lm.hat_vals) == 100
    assert lm.hat_vals[0] == pytest.approx(0.03751295)
    assert lm.hat_vals[-1] == pytest.approx(0.03880323)

    # check predictions
    assert predictions[0] == pytest.approx(10.739329)
    assert predictions[-1] == pytest.approx(9.880039)


######################
# Tests with weights #
######################


def test_random_weights_no_intercept(random_data_weights):
    lm = OLSRegressionSolver()
    x = random_data_weights[["x0", "x1", "x2", "x3", "x4"]].values
    y = random_data_weights["y"].values.reshape(-1, 1)
    weights = random_data_weights["weights"].values
    lm.fit(x, y, weights=weights, fit_intercept=False)
    predictions = lm.predict(x)

    # coefficients
    assert all(np.abs(lm.coefficients - [[1.455], [2.018], [4.699], [3.342], [9.669]]) <= TOL)

    # check hat values
    assert len(lm.hat_vals) == 100
    assert lm.hat_vals[0] == pytest.approx(0.013961893)
    assert lm.hat_vals[-1] == pytest.approx(0.044913885)

    # check predictions
    assert predictions[0] == pytest.approx(16.090619)
    assert predictions[-1] == pytest.approx(12.538442)


def test_random_weights_intercept(random_data_weights):
    lm = OLSRegressionSolver()
    random_data_weights["intercept"] = 1
    x = random_data_weights[["intercept", "x0", "x1", "x2", "x3", "x4"]].values
    y = random_data_weights["y"].values.reshape(-1, 1)
    weights = random_data_weights["weights"].values
    lm.fit(x, y, weights=weights, fit_intercept=True)
    predictions = lm.predict(x)

    # coefficients
    assert all(np.abs(lm.coefficients - [[0.1151], [1.4141], [1.9754], [4.6761], [3.2803], [9.6208]]) <= TOL)

    # check hat values
    assert len(lm.hat_vals) == 100
    assert lm.hat_vals[0] == pytest.approx(0.014940744)
    assert lm.hat_vals[-1] == pytest.approx(0.051229900)

    # check predictions
    assert predictions[0] == pytest.approx(16.069887)
    assert predictions[-1] == pytest.approx(12.565045)


########################
# Test regularization #
########################


def test_regularizer():
    lm = OLSRegressionSolver()

    lambda_I = lm._get_regularizer(7, 10, fit_intercept=True, regularize_intercept=True, n_feat_ignore_reg=2)

    assert lambda_I.shape == (10, 10)
    assert lambda_I[0, 0] == pytest.approx(7)
    assert lambda_I[1, 1] == pytest.approx(0)
    assert lambda_I[2, 2] == pytest.approx(0)
    assert lambda_I[3, 3] == pytest.approx(7)
    assert lambda_I[4, 4] == pytest.approx(7)
    assert lambda_I[5, 5] == pytest.approx(7)
    assert lambda_I[6, 6] == pytest.approx(7)
    assert lambda_I[7, 7] == pytest.approx(7)

    lambda_I = lm._get_regularizer(7, 10, fit_intercept=True, regularize_intercept=False, n_feat_ignore_reg=2)

    assert lambda_I.shape == (10, 10)
    assert lambda_I[0, 0] == pytest.approx(0)
    assert lambda_I[1, 1] == pytest.approx(0)
    assert lambda_I[2, 2] == pytest.approx(0)
    assert lambda_I[3, 3] == pytest.approx(7)
    assert lambda_I[4, 4] == pytest.approx(7)
    assert lambda_I[5, 5] == pytest.approx(7)
    assert lambda_I[6, 6] == pytest.approx(7)
    assert lambda_I[7, 7] == pytest.approx(7)


def test_regularization_with_intercept(random_data_no_weights):
    x = random_data_no_weights[["x0", "x1", "x2", "x3", "x4"]].values
    x[:, 0] = 1
    y = random_data_no_weights["y"].values

    lm = OLSRegressionSolver()
    lambda_ = 1e6
    lm.fit(x, y, lambda_=lambda_, fit_intercept=True, regularize_intercept=False)
    coefficients_w_reg = lm.coefficients
    assert all(np.abs(coefficients_w_reg[1:] - [0, 0, 0, 0]) <= TOL)
    assert np.abs(coefficients_w_reg[0]) > TOL


def test_regularization_with_intercept_and_unreg_feature(random_data_no_weights):
    x = random_data_no_weights[["x0", "x1", "x2", "x3", "x4"]].values
    x[:, 0] = 1
    y = random_data_no_weights["y"].values

    lm = OLSRegressionSolver()
    lambda_ = 1e6
    lm.fit(x, y, lambda_=lambda_, fit_intercept=True, regularize_intercept=False, n_feat_ignore_reg=2)
    coefficients_w_reg = lm.coefficients
    assert all(np.abs(coefficients_w_reg[3:] - [0, 0]) <= TOL)
    assert np.abs(coefficients_w_reg[0]) > TOL
    assert np.abs(coefficients_w_reg[1]) > TOL
    assert np.abs(coefficients_w_reg[2]) > TOL


def test_regularization_with_intercept_but_no_intercept_reg_and_unreg_feature(random_data_no_weights):
    x = random_data_no_weights[["x0", "x1", "x2", "x3", "x4"]].values
    x[:, 0] = 1
    y = random_data_no_weights["y"].values

    lm = OLSRegressionSolver()
    lambda_ = 1e6
    lm.fit(x, y, lambda_=lambda_, fit_intercept=True, regularize_intercept=True, n_feat_ignore_reg=2)
    coefficients_w_reg = lm.coefficients
    assert all(np.abs(coefficients_w_reg[3:] - [0, 0]) <= TOL)
    assert np.abs(coefficients_w_reg[0]) < TOL
    assert np.abs(coefficients_w_reg[1]) > TOL
    assert np.abs(coefficients_w_reg[2]) > TOL


##################
# Test residuals #
##################


def test_residuals_no_weights(random_data_no_weights):
    lm = OLSRegressionSolver()
    x = random_data_no_weights[["x0", "x1", "x2", "x3", "x4"]].values
    y = random_data_no_weights["y"].values.reshape(-1, 1)
    lm.fit(x, y, fit_intercept=False)
    predictions = lm.predict(x)

    residuals = lm.residuals(y, predictions, loo=False, center=False)

    assert residuals[0] == pytest.approx(0.885973530)
    assert residuals[-1] == pytest.approx(0.841996302)

    residuals = lm.residuals(y, predictions, loo=True, center=False)

    assert residuals[0] == pytest.approx(0.920112164)
    assert residuals[-1] == pytest.approx(0.875896477)

    residuals = lm.residuals(y, predictions, loo=True, center=True)
    assert np.sum(residuals) == pytest.approx(0)


def test_residuals_weights(random_data_weights):
    lm = OLSRegressionSolver()
    x = random_data_weights[["x0", "x1", "x2", "x3", "x4"]].values
    y = random_data_weights["y"].values.reshape(-1, 1)
    weights = random_data_weights["weights"].values

    lm.fit(x, y, weights=weights, fit_intercept=False)
    predictions = lm.predict(x)

    residuals = lm.residuals(y, predictions, loo=False, center=False)

    assert residuals[0] == pytest.approx(-1.971798590)
    assert residuals[-1] == pytest.approx(-1.373951578)

    residuals = lm.residuals(y, predictions, loo=True, center=False)

    assert residuals[0] == pytest.approx(-1.999718445)
    assert residuals[-1] == pytest.approx(-1.438563033)

    residuals = lm.residuals(y, predictions, loo=True, center=True)
    assert np.sum(residuals) == pytest.approx(0)


################################
# Test saving normal equations #
################################


def test_saving_normal_equations(random_data_no_weights):
    lm = OLSRegressionSolver()
    x = random_data_no_weights[["x0", "x1", "x2", "x3", "x4"]].values
    y = random_data_no_weights["y"].values.reshape(-1, 1)
    lm.fit(x, y, fit_intercept=False)

    normal_equations = lm.normal_eqs

    # passing normal equations so should stay the same
    x_new = np.ones_like(x)
    y_new = np.zeros_like(y)
    lm.fit(x_new, y_new, normal_eqs=normal_equations)
    np.testing.assert_array_equal(lm.normal_eqs, normal_equations)

    # not passing normal equations so should now change
    x_new = random_data_no_weights[["x0", "x1"]].values
    y_new = np.zeros_like(y)
    lm.fit(x_new, y_new, fit_intercept=False)
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(lm.normal_eqs, normal_equations)
