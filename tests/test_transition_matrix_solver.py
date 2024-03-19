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

    expected_betas = np.array([[0.760428, 0.239572], [0.216642, 0.783358]])
    expected_yhat = np.array(
        [
            [1.19371187, 1.80628813],
            [3.14785177, 3.85214823],
            [5.10199167, 5.89800833],
            [7.05613156, 7.94386844],
            [9.01027146, 9.98972854],
            [10.96441136, 12.03558864],
        ]
    )

    tms = TransitionMatrixSolver().fit(X, Y)
    current_yhat = tms.predict(X)
    np.testing.assert_allclose(expected_betas, tms.betas, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(expected_yhat, current_yhat, rtol=RTOL, atol=ATOL)


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

    expected_betas = np.array([[0.737329, 0.262671], [0.230589, 0.769411]])

    tms = TransitionMatrixSolver().fit(X, Y, sample_weight=weights)
    np.testing.assert_allclose(expected_betas, tms.betas, rtol=RTOL, atol=ATOL)


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

    expected_betas = np.array([[0.760451, 0.239558], [0.216624, 0.783369]])

    tms = TransitionMatrixSolver(strict=False).fit(X, Y)
    np.testing.assert_allclose(expected_betas, tms.betas, rtol=RTOL, atol=ATOL)


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

    expected_betas = np.array([[0.479416, 0.520584], [0.455918, 0.544082]])

    tms = TransitionMatrixSolver(lam=1).fit(X, Y)
    np.testing.assert_allclose(expected_betas, tms.betas, rtol=RTOL, atol=ATOL)


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

    expected_betas = np.array(
        [
            [0.68274443, 0.18437159, 0.06760119, 0.03363495, 0.0197597, 0.01188814],
            [0.13541428, 0.48122828, 0.22128163, 0.0960816, 0.04540571, 0.02058852],
            [0.04545795, 0.16052607, 0.38881747, 0.27665629, 0.12758087, 0.00096135],
            [0.02289342, 0.06401812, 0.17970185, 0.28708764, 0.28820718, 0.15809178],
            [0.01424566, 0.03468587, 0.08136858, 0.21299756, 0.26935036, 0.38735196],
            [0.00995853, 0.02159863, 0.04337214, 0.1113991, 0.30326763, 0.51040397],
        ]
    )

    tms = TransitionMatrixSolver().fit(X, Y)
    np.testing.assert_allclose(expected_betas, tms.betas, rtol=RTOL, atol=ATOL)


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
        tms.fit(X, Y)


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

        expected_betas = np.array([[0.760428, 0.239572], [0.216642, 0.783358]])

        tms = TransitionMatrixSolver().fit(X, Y)
        np.testing.assert_allclose(expected_betas, tms.betas, rtol=RTOL, atol=ATOL)

    except ImportError:
        # pass this test through since pandas isn't a requirement for elex-solver
        assert True
