import numpy as np
import pytest

from elexsolver.EITransitionSolver import EITransitionSolver

# high tolerance due to random sampling
# (which can produce different outcomes on different architectures, despite setting seeds)
RTOL = 1e-02
ATOL = 1e-02

np.random.seed(1024)


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

    expected = np.array([[0.279297, 0.720703], [0.623953, 0.376047]])

    ei = EITransitionSolver(random_seed=1024, n_samples=10, sampling_chains=1)
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

    expected = np.array([[0.279297, 0.720703], [0.623953, 0.376047]])

    ei = EITransitionSolver(random_seed=1024, n_samples=10, sampling_chains=1)
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

    expected = np.array([[0.279297, 0.720703], [0.623953, 0.376047]])

    ei = EITransitionSolver(random_seed=1024, n_samples=10, sampling_chains=1)
    current = ei.fit_predict(X, Y)
    np.testing.assert_allclose(expected, current, rtol=RTOL, atol=ATOL)


def test_ei_get_prediction_interval():
    with pytest.raises(NotImplementedError):
        ei = EITransitionSolver()
        ei.get_prediction_interval(0)


def test_ei_credible_interval_percentages():
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

    expected_lower = np.array([[0.037212, 0.356174], [0.424652, 0.117605]])
    expected_upper = np.array([[0.643826, 0.962788], [0.882395, 0.575348]])

    ei = EITransitionSolver(random_seed=1024, n_samples=10, sampling_chains=1)
    _ = ei.fit_predict(X, Y)
    (current_lower, current_upper) = ei.get_credible_interval(99, transitions=False)
    np.testing.assert_allclose(expected_lower, current_lower, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(expected_upper, current_upper, rtol=RTOL, atol=ATOL)


def test_ei_credible_interval_percentages_float_interval():
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

    expected_lower = np.array([[0.037212, 0.356174], [0.424652, 0.117605]])
    expected_upper = np.array([[0.643826, 0.962788], [0.882395, 0.575348]])

    ei = EITransitionSolver(random_seed=1024, n_samples=10, sampling_chains=1)
    _ = ei.fit_predict(X, Y)
    (current_lower, current_upper) = ei.get_credible_interval(0.99, transitions=False)
    np.testing.assert_allclose(expected_lower, current_lower, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(expected_upper, current_upper, rtol=RTOL, atol=ATOL)


def test_ei_credible_interval_invalid():
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

    ei = EITransitionSolver(random_seed=1024, n_samples=10, sampling_chains=1)
    _ = ei.fit_predict(X, Y)

    with pytest.raises(ValueError):
        ei.get_credible_interval(3467838976, transitions=False)


def test_ei_credible_interval_transitions():
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

    expected_lower = np.array([[0.017175, 0.164388], [0.228659, 0.063326]])
    expected_upper = np.array([[0.29715, 0.444364], [0.475136, 0.309803]])

    ei = EITransitionSolver(random_seed=1024, n_samples=10, sampling_chains=1)
    _ = ei.fit_predict(X, Y)
    (current_lower, current_upper) = ei.get_credible_interval(99, transitions=True)
    np.testing.assert_allclose(expected_lower, current_lower, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(expected_upper, current_upper, rtol=RTOL, atol=ATOL)
