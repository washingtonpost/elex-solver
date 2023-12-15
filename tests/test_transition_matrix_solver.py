import numpy as np
import pytest

from elexsolver.TransitionMatrixSolver import BootstrapTransitionMatrixSolver, TransitionMatrixSolver


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

    expected = np.array([[0.35096678, 0.11057168], [0.11665334, 0.4218082]])

    tms = TransitionMatrixSolver()
    current = tms.fit_predict(X, Y)
    np.testing.assert_allclose(expected, current, rtol=1e-08, atol=1e-02)


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

    expected = np.array([[0.340306, 0.121233], [0.124163, 0.414298]])

    tms = TransitionMatrixSolver()
    current = tms.fit_predict(X, Y, weights=weights)
    np.testing.assert_allclose(expected, current, rtol=1e-08, atol=1e-02)


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

    expected = np.array([[0.374623, 0.087791], [0.093755, 0.44383]])

    btms = BootstrapTransitionMatrixSolver(B=10, verbose=False)
    current = btms.fit_predict(X, Y)
    np.testing.assert_allclose(expected, current, rtol=1e-08, atol=1e-02)


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

    expected = np.array([[0.319791, 0.112347], [0.130296, 0.437565]])

    btms = BootstrapTransitionMatrixSolver(B=10, verbose=False)
    current = btms.fit_predict(X, Y, weights=weights)
    np.testing.assert_allclose(expected, current, rtol=1e-08, atol=1e-02)


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

    expected_lower = np.array([[0.34326, 0.045649], [0.047865, 0.418057]])
    expected_upper = np.array([[0.429978, 0.112171], [0.119081, 0.477393]])

    btms = BootstrapTransitionMatrixSolver(B=10, verbose=False)
    _ = btms.fit_predict(X, Y)
    (current_lower, current_upper) = btms.get_confidence_interval(0.95)
    np.testing.assert_allclose(expected_lower, current_lower, rtol=1e-08, atol=1e-02)
    np.testing.assert_allclose(expected_upper, current_upper, rtol=1e-08, atol=1e-02)
