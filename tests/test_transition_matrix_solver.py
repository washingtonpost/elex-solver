import numpy as np
import pytest

from elexsolver.TransitionMatrixSolver import TransitionMatrixSolver


def test_fit_predict():
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


def test_fit_predict_with_weights():
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


def test_get_prediction_interval():
    with pytest.raises(NotImplementedError):
        tms = TransitionMatrixSolver()
        tms.get_prediction_interval(0)
