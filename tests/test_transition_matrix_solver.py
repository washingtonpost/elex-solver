import numpy as np
import pytest

from elexsolver.TransitionMatrixSolver import BootstrapTransitionMatrixSolver, TransitionMatrixSolver

RTOL = 1e-04
ATOL = 1e-04


def test_matrix_fit_predict():
    X = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
        ]
    )

    Y = np.array(
        [
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
            [10, 11],
            [12, 13],
        ]
    )

    expected = np.array([[0.760428, 0.239572], [0.216642, 0.783358]])

    tms = TransitionMatrixSolver()
    current = tms.fit_predict(X, Y)
    np.testing.assert_allclose(expected, current, rtol=RTOL, atol=ATOL)


def test_matrix_fit_predict_with_weights():
    X = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
        ]
    )

    Y = np.array(
        [
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
            [10, 11],
            [12, 13],
        ]
    )

    weights = np.array([500, 250, 125, 62.5, 31.25, 15.625])

    expected = np.array([[0.737329, 0.262671], [0.230589, 0.769411]])

    tms = TransitionMatrixSolver()
    current = tms.fit_predict(X, Y, weights=weights)
    np.testing.assert_allclose(expected, current, rtol=RTOL, atol=ATOL)


def test_matrix_fit_predict_not_strict():
    X = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
        ]
    )

    Y = np.array(
        [
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
            [10, 11],
            [12, 13],
        ]
    )

    expected = np.array([[0.760451, 0.239558], [0.216624, 0.783369]])

    tms = TransitionMatrixSolver(strict=False)
    current = tms.fit_predict(X, Y)
    np.testing.assert_allclose(expected, current, rtol=RTOL, atol=ATOL)


def test_ridge_matrix_fit_predict():
    X = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
        ]
    )

    Y = np.array(
        [
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
            [10, 11],
            [12, 13],
        ]
    )

    expected = np.array([[0.479416, 0.520584], [0.455918, 0.544082]])

    tms = TransitionMatrixSolver(lam=1)
    current = tms.fit_predict(X, Y)
    np.testing.assert_allclose(expected, current, rtol=RTOL, atol=ATOL)


def test_matrix_fit_predict_pivoted():
    X = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
        ]
    ).T

    Y = np.array(
        [
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
            [10, 11],
            [12, 13],
        ]
    ).T

    expected = np.array([[0.760428, 0.239572], [0.216642, 0.783358]])

    tms = TransitionMatrixSolver()
    current = tms.fit_predict(X, Y)
    np.testing.assert_allclose(expected, current, rtol=RTOL, atol=ATOL)


def test_matrix_get_prediction_interval():
    with pytest.raises(NotImplementedError):
        tms = TransitionMatrixSolver()
        tms.get_prediction_interval(0)


def test_bootstrap_fit_predict():
    X = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
        ]
    )

    Y = np.array(
        [
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
            [10, 11],
            [12, 13],
        ]
    )

    expected = np.array([[0.809393, 0.190607], [0.173843, 0.826157]])

    btms = BootstrapTransitionMatrixSolver(B=10, verbose=False)
    current = btms.fit_predict(X, Y)
    np.testing.assert_allclose(expected, current, rtol=RTOL, atol=ATOL)


def test_bootstrap_fit_predict_with_weights():
    X = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
        ]
    )

    Y = np.array(
        [
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
            [10, 11],
            [12, 13],
        ]
    )

    weights = np.array([500, 250, 125, 62.5, 31.25, 15.625])

    expected = np.array([[0.739798, 0.260202], [0.229358, 0.770642]])

    btms = BootstrapTransitionMatrixSolver(B=10, verbose=False)
    current = btms.fit_predict(X, Y, weights=weights)
    np.testing.assert_allclose(expected, current, rtol=RTOL, atol=ATOL)


def test_bootstrap_confidence_interval():
    X = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
        ]
    )

    Y = np.array(
        [
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
            [10, 11],
            [12, 13],
        ]
    )

    expected_lower = np.array([[0.757573, 0.095978], [0.09128, 0.779471]])
    expected_upper = np.array([[0.904022, 0.242427], [0.220529, 0.90872]])

    btms = BootstrapTransitionMatrixSolver(B=10, verbose=False)
    _ = btms.fit_predict(X, Y)
    (current_lower, current_upper) = btms.get_confidence_interval(0.95)
    np.testing.assert_allclose(expected_lower, current_lower, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(expected_upper, current_upper, rtol=RTOL, atol=ATOL)
