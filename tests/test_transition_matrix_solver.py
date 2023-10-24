import numpy as np
import pytest

from elexsolver.TransitionMatrixSolver import TransitionMatrixSolver


def test_fit_predict():
    X = np.array(
        [
            [0.13991186, 0.19010302],
            [0.13774767, 0.19199878],
            [0.17240947, 0.16163616],
            [0.19983843, 0.13760927],
            [0.15796058, 0.17429292],
            [0.192132, 0.14435986],
        ]
    )

    Y = np.array(
        [
            [0.15551131, 0.16977255],
            [0.1573689, 0.16925536],
            [0.16995309, 0.16575166],
            [0.15144583, 0.17090446],
            [0.16700258, 0.16657314],
            [0.19871829, 0.15774283],
        ]
    )

    expected = np.array([[0.29134295, 0.20806254], [0.2076699, 0.29292461]])

    tms = TransitionMatrixSolver()
    current = tms.fit_predict(X, Y)
    np.testing.assert_allclose(expected, current)


def test_mean_absolute_error():
    X = np.ones((10, 3))
    Y = np.ones((10, 4))
    expected = 0.0
    tms = TransitionMatrixSolver()
    tms.fit_predict(X, Y)
    current = np.around(tms.mean_absolute_error(X, Y), 6)
    np.testing.assert_allclose(expected, current)


def test_get_prediction_interval():
    with pytest.raises(NotImplementedError):
        tms = TransitionMatrixSolver()
        tms.get_prediction_interval(0)
