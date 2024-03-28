import numpy as np
import pytest

from elexsolver.TransitionMatrixSolver import TransitionMatrixSolver

RTOL = 1e-04
ATOL = 1e-04


def test_matrix_fit_predict_with_integers():
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

    expected_betas = np.array([[9.99831808e-01, 1.68191521e-04], [1.49085896e-04, 9.99850914e-01]])
    expected_yhat = np.array(
        [
            [1.00012998, 1.99987002],
            [3.00009177, 3.99990823],
            [5.00005356, 5.99994644],
            [7.00001535, 7.99998465],
            [8.99997714, 10.00002286],
            [10.99993892, 12.00006108],
        ]
    )

    tms = TransitionMatrixSolver().fit(X, Y)
    current_yhat = tms.predict(X)
    np.testing.assert_allclose(expected_betas, tms.coefficients, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(expected_yhat, current_yhat, rtol=RTOL, atol=ATOL)


def test_matrix_fit_predict():
    X = np.array(
        [
            [0.33333333, 0.66666667],
            [0.42857143, 0.57142857],
            [0.45454545, 0.54545455],
            [0.46666667, 0.53333333],
            [0.47368421, 0.52631579],
            [0.47826087, 0.52173913],
        ]
    )

    Y = np.array(
        [
            [0.4, 0.6],
            [0.44444444, 0.55555556],
            [0.46153846, 0.53846154],
            [0.47058824, 0.52941176],
            [0.47619048, 0.52380952],
            [0.48, 0.52],
        ]
    )

    expected_betas = np.array([[0.760428, 0.239572], [0.216642, 0.783358]])
    expected_yhat = np.array(
        [
            [0.39790396, 0.60209604],
            [0.44969311, 0.55030689],
            [0.46381742, 0.53618258],
            [0.47040877, 0.52959123],
            [0.47422481, 0.52577519],
            [0.47671354, 0.52328646],
        ]
    )

    tms = TransitionMatrixSolver().fit(X, Y)
    current_yhat = tms.predict(X)
    np.testing.assert_allclose(expected_betas, tms.coefficients, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(expected_yhat, current_yhat, rtol=RTOL, atol=ATOL)


def test_matrix_fit_predict_with_weights():
    X = np.array(
        [
            [0.33333333, 0.66666667],
            [0.42857143, 0.57142857],
            [0.45454545, 0.54545455],
            [0.46666667, 0.53333333],
            [0.47368421, 0.52631579],
            [0.47826087, 0.52173913],
        ]
    )

    Y = np.array(
        [
            [0.4, 0.6],
            [0.44444444, 0.55555556],
            [0.46153846, 0.53846154],
            [0.47058824, 0.52941176],
            [0.47619048, 0.52380952],
            [0.48, 0.52],
        ]
    )

    weights = np.array([500, 250, 125, 62.5, 31.25, 15.625])

    expected_betas = np.array([[0.737329, 0.262671], [0.230589, 0.769411]])

    tms = TransitionMatrixSolver().fit(X, Y, sample_weight=weights)
    np.testing.assert_allclose(expected_betas, tms.coefficients, rtol=RTOL, atol=ATOL)


def test_matrix_fit_predict_not_strict():
    X = np.array(
        [
            [0.33333333, 0.66666667],
            [0.42857143, 0.57142857],
            [0.45454545, 0.54545455],
            [0.46666667, 0.53333333],
            [0.47368421, 0.52631579],
            [0.47826087, 0.52173913],
        ]
    )

    Y = np.array(
        [
            [0.4, 0.6],
            [0.44444444, 0.55555556],
            [0.46153846, 0.53846154],
            [0.47058824, 0.52941176],
            [0.47619048, 0.52380952],
            [0.48, 0.52],
        ]
    )

    expected_betas = np.array([[0.760451, 0.239558], [0.216624, 0.783369]])

    tms = TransitionMatrixSolver(strict=False).fit(X, Y)
    np.testing.assert_allclose(expected_betas, tms.coefficients, rtol=RTOL, atol=ATOL)


def test_ridge_matrix_fit_predict():
    X = np.array(
        [
            [0.33333333, 0.66666667],
            [0.42857143, 0.57142857],
            [0.45454545, 0.54545455],
            [0.46666667, 0.53333333],
            [0.47368421, 0.52631579],
            [0.47826087, 0.52173913],
        ]
    )

    Y = np.array(
        [
            [0.4, 0.6],
            [0.44444444, 0.55555556],
            [0.46153846, 0.53846154],
            [0.47058824, 0.52941176],
            [0.47619048, 0.52380952],
            [0.48, 0.52],
        ]
    )

    expected_betas = np.array([[0.479416, 0.520584], [0.455918, 0.544082]])

    tms = TransitionMatrixSolver(lam=1).fit(X, Y)
    np.testing.assert_allclose(expected_betas, tms.coefficients, rtol=RTOL, atol=ATOL)


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
            [9.99706e-01, 1.85000e-04, 5.00000e-05, 2.80000e-05, 1.90000e-05, 1.30000e-05],
            [4.80000e-05, 9.99464e-01, 3.43000e-04, 8.10000e-05, 4.00000e-05, 2.40000e-05],
            [1.70000e-05, 1.56000e-04, 9.99188e-01, 4.86000e-04, 1.06000e-04, 4.70000e-05],
            [1.00000e-05, 4.60000e-05, 2.76000e-04, 9.98960e-01, 5.93000e-04, 1.14000e-04],
            [7.00000e-06, 2.40000e-05, 7.40000e-05, 3.88000e-04, 9.98887e-01, 6.20000e-04],
            [5.00000e-06, 1.50000e-05, 3.60000e-05, 9.70000e-05, 4.66000e-04, 9.99382e-01],
        ]
    )

    tms = TransitionMatrixSolver().fit(X, Y)
    np.testing.assert_allclose(expected_betas, tms.coefficients, rtol=RTOL, atol=ATOL)


def test_matrix_fit_predict_bad_dimensions():
    X = np.array(
        [
            [0.33333333, 0.66666667],
            [0.42857143, 0.57142857],
            [0.45454545, 0.54545455],
            [0.46666667, 0.53333333],
            [0.47368421, 0.52631579],
            [0.47826087, 0.52173913],
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
                [0.33333333, 0.66666667],
                [0.42857143, 0.57142857],
                [0.45454545, 0.54545455],
                [0.46666667, 0.53333333],
                [0.47368421, 0.52631579],
                [0.47826087, 0.52173913],
            ],
            columns=["x1", "x2"],
        )

        Y = pandas.DataFrame(
            [
                [0.4, 0.6],
                [0.44444444, 0.55555556],
                [0.46153846, 0.53846154],
                [0.47058824, 0.52941176],
                [0.47619048, 0.52380952],
                [0.48, 0.52],
            ],
            columns=["y1", "y2"],
        )

        expected_betas = np.array([[0.760428, 0.239572], [0.216642, 0.783358]])

        tms = TransitionMatrixSolver().fit(X, Y)
        np.testing.assert_allclose(expected_betas, tms.coefficients, rtol=RTOL, atol=ATOL)

    except ImportError:
        # pass this test through since pandas isn't a requirement for elex-solver
        assert True
