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
def test_superclass_mean_absolute_error():
    with pytest.raises(NotImplementedError):
        ts = TransitionSolver()
        ts.mean_absolute_error(None, None)


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_superclass_get_prediction_interval():
    with pytest.raises(NotImplementedError):
        ts = TransitionSolver()
        ts.get_prediction_interval(0)


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_get_expected_totals():
    A = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    expected = np.array([0.23809524, 0.33333333, 0.42857143])
    ts = TransitionSolver()
    np.testing.assert_allclose(ts._get_expected_totals(A), expected)  # pylint: disable=protected-access


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
def test_check_percentages_bad():
    with pytest.raises(ValueError):
        A = np.array([[0.1, 0.2, 3], [0.4, 0.5, 0.6]])
        ts = TransitionSolver()
        ts._check_percentages(A)  # pylint: disable=protected-access


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_check_percentages_good():
    A = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    ts = TransitionSolver()
    ts._check_percentages(A)  # pylint: disable=protected-access
