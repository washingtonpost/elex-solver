
import numpy as np
import pytest
from elexsolver.QuantileRegressionSolver import QuantileRegressionSolver, IllConditionedMatrixException

# relatively high tolerance, since different implementation.
TOL = 1e-3

# the outputs are compared against quantreg package in R

###############
# Basic tests #
###############

def test_basic_median_1():
    quantreg = QuantileRegressionSolver()
    tau = 0.5
    x = np.asarray([[1], [1], [1], [2]])
    y = np.asarray([3, 8, 9, 15])
    quantreg.fit(x, y, tau)
    preds = quantreg.predict(x)
    # you'd think it would be 8 instead of 7.5, but run quantreg in R to confirm
    # has to do with missing intercept
    assert all(np.abs(preds - [7.5, 7.5, 7.5, 15]) <= TOL)

def test_basic_median_2():
    quantreg = QuantileRegressionSolver()
    tau = 0.5
    x = np.asarray([[1, 1], [1, 1], [1, 1], [1, 2]])
    y = np.asarray([3, 8, 9, 15])
    quantreg.fit(x, y, tau)
    preds = quantreg.predict(x)
    assert all(np.abs(preds - [8, 8, 8, 15]) <= TOL)

def test_basic_lower():
    quantreg = QuantileRegressionSolver()
    tau = 0.1
    x = np.asarray([[1, 1], [1, 1], [1, 1], [1, 2]])
    y = np.asarray([3, 8, 9, 15])
    quantreg.fit(x, y, tau)
    preds = quantreg.predict(x)
    assert all(np.abs(preds - [3, 3, 3, 15]) <= TOL)

def test_basic_upper():
    quantreg = QuantileRegressionSolver()
    tau = 0.9
    x = np.asarray([[1, 1], [1, 1], [1, 1], [1, 2]])
    y = np.asarray([3, 8, 9, 15])
    quantreg.fit(x, y, tau)
    preds = quantreg.predict(x)
    assert all(np.abs(preds - [9, 9, 9, 15]) <= TOL)

######################
# Intermediate tests #
######################

def test_random_median(random_data_no_weights):
    quantreg = QuantileRegressionSolver()
    tau = 0.5
    x = random_data_no_weights[['x0', 'x1', 'x2', 'x3', 'x4']].values
    y = random_data_no_weights['y'].values
    quantreg.fit(x, y, tau)
    preds = quantreg.predict(x)
    assert all(np.abs(quantreg.coefficients - [1.57699, 6.74906, 4.40175, 4.85346, 4.51814]) <= TOL)

def test_random_lower(random_data_no_weights):
    quantreg = QuantileRegressionSolver()
    tau = 0.1
    x = random_data_no_weights[['x0', 'x1', 'x2', 'x3', 'x4']].values
    y = random_data_no_weights['y'].values
    quantreg.fit(x, y, tau)
    preds = quantreg.predict(x)
    assert all(np.abs(quantreg.coefficients - [0.17759, 6.99588, 4.18896, 4.83906, 3.22546]) <= TOL)

def test_random_upper(random_data_no_weights):
    quantreg = QuantileRegressionSolver()
    tau = 0.9
    x = random_data_no_weights[['x0', 'x1', 'x2', 'x3', 'x4']].values
    y = random_data_no_weights['y'].values
    quantreg.fit(x, y, tau)
    preds = quantreg.predict(x)
    assert all(np.abs(quantreg.coefficients - [1.85617, 6.81286, 6.05586, 5.51965, 4.19864]) <= TOL)

######################
# Tests with weights #
######################

def test_basic_median_weights():
    quantreg = QuantileRegressionSolver()
    tau = 0.5
    x = np.asarray([[1, 1], [1, 1], [1, 1], [1, 2]])
    y = np.asarray([3, 8, 9, 15])
    weights = np.asarray([1, 1, 100, 3])
    quantreg.fit(x, y, tau, weights)
    preds = quantreg.predict(x)
    assert all(np.abs(preds - [9, 9, 9, 15]) <= TOL)

def test_random_median_weights(random_data_weights):
    quantreg = QuantileRegressionSolver()
    tau = 0.5
    x = random_data_weights[['x0', 'x1', 'x2', 'x3', 'x4']].values
    y = random_data_weights['y'].values
    weights = random_data_weights['weights'].values
    quantreg.fit(x, y, tau, weights=weights)
    preds = quantreg.predict(x)
    assert all(np.abs(quantreg.coefficients - [1.59521, 2.17864, 4.68050, 3.10920, 9.63739]) <= TOL)

def test_random_lower_weights(random_data_weights):
    quantreg = QuantileRegressionSolver()
    tau = 0.1
    x = random_data_weights[['x0', 'x1', 'x2', 'x3', 'x4']].values
    y = random_data_weights['y'].values
    weights = random_data_weights['weights'].values
    quantreg.fit(x, y, tau, weights=weights)
    preds = quantreg.predict(x)
    assert all(np.abs(quantreg.coefficients - [0.63670, 1.27028, 4.81500, 3.08055, 8.69929]) <= TOL)

def test_random_upper_weights(random_data_weights):
    quantreg = QuantileRegressionSolver()
    tau = 0.9
    x = random_data_weights[['x0', 'x1', 'x2', 'x3', 'x4']].values
    y = random_data_weights['y'].values
    weights = random_data_weights['weights'].values
    quantreg.fit(x, y, tau, weights=weights)
    preds = quantreg.predict(x)
    assert all(np.abs(quantreg.coefficients - [3.47742, 2.07360, 4.51754, 4.15237, 9.58856]) <= TOL)

########################
# Test changing solver #
########################

def test_changing_solver(random_data_no_weights):
    tau = 0.5
    x = random_data_no_weights[['x0', 'x1', 'x2', 'x3', 'x4']].values
    y = random_data_no_weights['y'].values
    
    quantreg_scs = QuantileRegressionSolver(solver='SCS')
    quantreg_ecos = QuantileRegressionSolver(solver='ECOS')
    quantreg_scs.fit(x, y, tau, save_problem=True)
    quantreg_ecos.fit(x, y, tau, save_problem=True)
    
    assert quantreg_scs.problem.value == pytest.approx(quantreg_ecos.problem.value, TOL)

def test_changing_solver_weights(random_data_weights):
    tau = 0.5
    x = random_data_weights[['x0', 'x1', 'x2', 'x3', 'x4']].values
    y = random_data_weights['y'].values
    weights = random_data_weights['weights'].values
    
    quantreg_scs = QuantileRegressionSolver(solver='SCS')
    quantreg_ecos = QuantileRegressionSolver(solver='ECOS')
    quantreg_scs.fit(x, y, tau, weights=weights, save_problem=True)
    quantreg_ecos.fit(x, y, tau, weights=weights, save_problem=True)

    assert quantreg_scs.problem.value == pytest.approx(quantreg_ecos.problem.value, TOL)

#######################
# Test saving problem #
#######################

def test_saving_problem(random_data_no_weights):
    tau = 0.5
    x = random_data_no_weights[['x0', 'x1', 'x2', 'x3', 'x4']].values
    y = random_data_no_weights['y'].values
    
    quantreg = QuantileRegressionSolver(solver='ECOS')
    
    quantreg.fit(x, y, tau, save_problem=False)
    assert quantreg.problem is None

    quantreg.fit(x, y, tau, save_problem=True)
    assert quantreg.problem is not None

    # testing whether overwrite works
    quantreg.fit(x, y, tau, save_problem=False)
    assert quantreg.problem is None

#############################
# Test weight normalization #
#############################

def test_weight_normalization_divide_by_zero(random_data_no_weights):
    tau = 0.5
    x = random_data_no_weights[['x0', 'x1', 'x2', 'x3', 'x4']].values
    y = random_data_no_weights['y'].values
    weights = [0] * x.shape[0]  # all zero weights
    
    quantreg = QuantileRegressionSolver(solver='ECOS')

    # Will succeed without weight normalization
    quantreg.fit(x, y, tau, normalize_weights=False, weights=weights)

    # Will fail with weight normalization
    with pytest.raises(ZeroDivisionError):
        quantreg.fit(x, y, tau, normalize_weights=True, weights=weights)

def test_weight_normalization_same_fit(random_data_weights):
    quantreg = QuantileRegressionSolver()
    tau = 0.5
    x = np.asarray([[1, 1], [1, 1], [1, 1], [1, 2]])
    y = np.asarray([3, 8, 9, 15])
    weights = np.asarray([1, 1, 100, 3])

    # Test predictions are right when normalize_weights is True and False
    quantreg.fit(x, y, tau, weights, normalize_weights=True)
    preds = quantreg.predict(x)
    assert all(np.abs(preds - [9, 9, 9, 15]) <= TOL)

    quantreg.fit(x, y, tau, weights, normalize_weights=False)
    preds = quantreg.predict(x)
    assert all(np.abs(preds - [9, 9, 9, 15]) <= TOL)

########################
# Test checking matrix #
########################

def test_ill_conditioned_error():
    quantreg = QuantileRegressionSolver()

    x = np.asarray([[1, 0, 1], [4, 3, 4], [5, 2, 5]])
    with pytest.raises(IllConditionedMatrixException) as e_info:
        quantreg._check_matrix_condition(x)

def test_ill_conditioned_warning():
    quantreg = QuantileRegressionSolver()

    random_number_generator = np.random.RandomState(42)
    mu = np.asarray([1, 3, 5])
    sigma = np.asarray([[1, 0.9, 0], [0.9, 1, 0], [0, 0, 1]])
    x = random_number_generator.multivariate_normal(mu, sigma, size=3)
    matrix_check = quantreg._check_matrix_condition(x)
    assert not matrix_check

def test_ill_conditioned_warning():
    quantreg = QuantileRegressionSolver()

    random_number_generator = np.random.RandomState(42)
    mu = np.asarray([1, 3, 5])
    sigma = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    x = random_number_generator.multivariate_normal(mu, sigma, size=3)
    matrix_check = quantreg._check_matrix_condition(x)
    assert matrix_check