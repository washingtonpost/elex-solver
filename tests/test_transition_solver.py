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
def test_superclass_get_transitions():
    ts = TransitionSolver()
    assert ts.transitions is None


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
def test_check_for_zero_units_good():
    A = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
    )
    ts = TransitionSolver()
    ts._check_for_zero_units(A)  # pylint: disable=protected-access


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_check_for_zero_units_bad():
    with pytest.raises(ValueError):
        A = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.0, 0.0, 0.0],
                [0.7, 0.8, 0.9],
            ]
        )
        ts = TransitionSolver()
        ts._check_for_zero_units(A)  # pylint: disable=protected-access


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_rescale_rescaled_numpy():
    A = np.ones((2, 2)).astype(int)
    expected = np.array([[0.5, 0.5], [0.5, 0.5]])
    ts = TransitionSolver()
    np.testing.assert_array_equal(ts._rescale(A), expected)  # pylint: disable=protected-access


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_rescale_rescaled_pandas():
    try:
        import pandas  # pylint: disable=import-outside-toplevel

        a_df = pandas.DataFrame(np.ones((2, 2)), columns=["A", "B"]).astype(int)
        expected_df = pandas.DataFrame([[0.5, 0.5], [0.5, 0.5]], columns=["A", "B"])
        ts = TransitionSolver()
        np.testing.assert_array_equal(expected_df, ts._rescale(a_df))  # pylint: disable=protected-access
    except ImportError:
        # pass this test through since pandas isn't a requirement for elex-solver
        assert True


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


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_check_and_prepare_weights_bad():
    with pytest.raises(ValueError):
        weights = [1, 2]
        A = np.array([[1, 2], [3, 4], [5, 6]])
        B = A.copy()
        ts = TransitionSolver()
        ts._check_and_prepare_weights(A, B, weights)  # pylint: disable=protected-access


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_check_and_prepare_weights_none():
    A = np.array([[1, 2], [3, 4], [5, 6]])
    B = A.copy()
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    ts = TransitionSolver()
    current = ts._check_and_prepare_weights(A, B, None)  # pylint: disable=protected-access
    np.testing.assert_array_equal(expected, current)


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_check_and_prepare_weights_with_weights():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = A.copy()
    weights = np.array([0.6, 0.4])
    expected = np.array([[0.77459667, 0], [0, 0.63245553]])

    ts = TransitionSolver()
    current = ts._check_and_prepare_weights(A, B, weights)  # pylint: disable=protected-access
    np.testing.assert_allclose(expected, current)


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_check_and_prepare_weights_with_weights_list():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = A.copy()
    weights = [0.6, 0.4]
    expected = np.array([[0.77459667, 0], [0, 0.63245553]])

    ts = TransitionSolver()
    current = ts._check_and_prepare_weights(A, B, weights)  # pylint: disable=protected-access
    np.testing.assert_allclose(expected, current)


@patch.object(TransitionSolver, "__abstractmethods__", set())
def test_check_and_prepare_weights_with_weights_pandas():
    try:
        import pandas  # pylint: disable=import-outside-toplevel

        A = np.array([[1, 2, 3], [4, 5, 6]])
        B = A.copy()
        weights = pandas.Series([0.6, 0.4])
        expected = np.array([[0.77459667, 0], [0, 0.63245553]])

        ts = TransitionSolver()
        current = ts._check_and_prepare_weights(A, B, weights)  # pylint: disable=protected-access
        np.testing.assert_allclose(expected, current)
    except ImportError:
        # pass this test through since pandas isn't a requirement for elex-solver
        assert True
