import numpy as np
import pytest

from elexsolver.TransitionMatrixSolver import TransitionMatrixSolver

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


def test_matrix_fit_predict_bad_dimensions():
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
        ]
    )

    tms = TransitionMatrixSolver()
    with pytest.raises(ValueError):
        tms.fit_predict(X, Y)


def test_matrix_fit_predict_pandas():
    try:
        import pandas  # pylint: disable=import-outside-toplevel

        X = pandas.DataFrame(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
                [9, 10],
                [11, 12],
            ],
            columns=["x1", "x2"],
        )

        Y = pandas.DataFrame(
            [
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9],
                [10, 11],
                [12, 13],
            ],
            columns=["y1", "y2"],
        )

        expected = np.array([[0.760428, 0.239572], [0.216642, 0.783358]])

        tms = TransitionMatrixSolver()
        current = tms.fit_predict(X, Y)
        np.testing.assert_allclose(expected, current, rtol=RTOL, atol=ATOL)

    except ImportError:
        # pass this test through since pandas isn't a requirement for elex-solver
        assert True
