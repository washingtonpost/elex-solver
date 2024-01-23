import numpy as np
import pytest

from elexsolver.EITransitionSolver import EITransitionSolver

RTOL = 1e-04
ATOL = 1e-04


def test_ei_fit_predict():
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

    expected = np.array([[0.530026, 0.469974], [0.401865, 0.598135]])

    ei = EITransitionSolver(random_seed=1024)
    current = ei.fit_predict(X, Y)
    np.testing.assert_allclose(expected, current, rtol=RTOL, atol=ATOL)


def test_ei_fit_predict_with_weights():
    # NOTE: currently, supplying weights to the EI solver does nothing.
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

    expected = np.array([[0.530026, 0.469974], [0.401865, 0.598135]])

    ei = EITransitionSolver(random_seed=1024)
    current = ei.fit_predict(X, Y, weights=weights)
    np.testing.assert_allclose(expected, current, rtol=RTOL, atol=ATOL)


def test_ei_fit_predict_pivoted():
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

    expected = np.array([[0.530026, 0.469974], [0.401865, 0.598135]])

    ei = EITransitionSolver(random_seed=1024)
    current = ei.fit_predict(X, Y)
    np.testing.assert_allclose(expected, current, rtol=RTOL, atol=ATOL)


def test_ei_get_prediction_interval():
    with pytest.raises(NotImplementedError):
        ei = EITransitionSolver()
        ei.get_prediction_interval(0)
