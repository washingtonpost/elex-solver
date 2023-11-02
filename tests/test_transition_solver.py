from unittest.mock import patch

import numpy as np
import pytest

from elexsolver.TransitionSolver import TransitionSolver


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_superclass_fit_predict():
    with pytest.raises(NotImplementedError):
        ts = TransitionSolver()
        ts.fit_predict(None, None)


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_superclass_get_prediction_interval():
    with pytest.raises(NotImplementedError):
        ts = TransitionSolver()
        ts.get_prediction_interval(0)


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_mean_absolute_error():
    Y = np.ones((5, 4))
    Y_pred = Y - 0.02
    expected = 0.08
    ts = TransitionSolver()
    current = np.around(ts.mean_absolute_error(Y, Y_pred), 6)
    np.testing.assert_allclose(expected, current)


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_check_any_element_nan_or_inf_with_nan():
    with pytest.raises(ValueError):
        A = np.array([[0.1, 0.2, 0.3], [0.4, np.nan, 0.6]])
        ts = TransitionSolver()
        ts._check_any_element_nan_or_inf(A)  # pylint: disable=protected-access


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_check_any_element_nan_or_inf_without_nan():
    A = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    ts = TransitionSolver()
    ts._check_any_element_nan_or_inf(A)  # pylint: disable=protected-access


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_check_dimensions_bad():
    with pytest.raises(ValueError):
        A = np.array([[0.1, 0.2, 0.3]])
        ts = TransitionSolver()
        ts._check_dimensions(A)  # pylint: disable=protected-access


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_check_dimensions_good():
    A = np.array(
        [
            [0.1, 0.4, 0.7, 0.1, 0.4, 0.7, 0.1, 0.4, 0.7],
            [0.2, 0.5, 0.8, 0.2, 0.5, 0.8, 0.2, 0.5, 0.8],
            [0.3, 0.6, 0.9, 0.3, 0.6, 0.9, 0.3, 0.6, 0.9],
        ]
    )
    ts = TransitionSolver()
    ts._check_dimensions(A)  # pylint: disable=protected-access


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_rescale_rescaled_numpy():
    A = np.ones((2, 2))
    expected = np.array([[0.5, 0.5], [0.5, 0.5]])
    ts = TransitionSolver()
    np.testing.assert_array_equal(ts._rescale(A), expected)  # pylint: disable=protected-access


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_rescale_rescaled_pandas():
    import pandas

    a_df = pandas.DataFrame(np.ones((2, 2)), columns=["A", "B"])
    expected_df = pandas.DataFrame([[0.5, 0.5], [0.5, 0.5]], columns=["A", "B"])
    ts = TransitionSolver()
    np.testing.assert_array_equal(ts._rescale(a_df), expected_df)  # pylint: disable=protected-access


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_check_data_type_good():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    ts = TransitionSolver()
    ts._check_data_type(A)  # pylint: disable=protected-access


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_check_data_type_bad():
    with pytest.raises(ValueError):
        A = np.array([[0.1, 0.2, 0.3]])
        ts = TransitionSolver()
        ts._check_data_type(A)  # pylint: disable=protected-access