import logging

import numpy as np

from elexsolver.LinearSolver import LinearSolver
from elexsolver.logging import initialize_logging

initialize_logging()

LOG = logging.getLogger(__name__)


class OLSRegressionSolver(LinearSolver):
    """
    A class for Ordinary Least Squares Regression optimized for the bootstrap
    """

    # OLS setup:
    #       X \beta = y
    # since X might not be square, we multiply the above equation on both sides by X^T to generate X^T X, which is guaranteed
    # to be square
    #       X^T X \beta = X^T y
    # Since X^T X is square we can invert it
    #       \beta = (X^T X)^{-1} X^T y
    # Since our version of the model bootstraps y, but keeps X constant we can
    # pre-compute (X^T X)^{-1} X^T and then re-use it to compute \beta_b for every bootstrap sample

    def __init__(self):
        super().__init__()
        self.normal_eqs = None
        self.hat_vals = None

    def _get_regularizer(
        self, lambda_: float, dim: int, fit_intercept: bool, regularize_intercept: bool, n_feat_ignore_reg: int
    ) -> np.ndarray:
        """
        Returns the regularization matrix
        """
        # lambda_I is the matrix for regularization, which need to be the same shape as R and
        # have the regularization constant lambda_ along the diagonal
        lambda_I = lambda_ * np.eye(dim)

        # we don't want to regularize the coefficient for intercept
        # but we also might not want to fit the intercept
        # for some number of features
        # so set regularization constant to zero for intercept
        # and the first n_feat_ignore_reg features
        for i in range(fit_intercept + n_feat_ignore_reg):
            # if we are fitting an intercept and want to regularize intercept then we don't want
            # to set the regularization matrix at lambda_I[0, 0] to zero
            if fit_intercept and i == 0 and regularize_intercept:
                continue
            lambda_I[i, i] = 0

        return lambda_I

    def _compute_normal_equations(
        self,
        x: np.ndarray,
        L: np.ndarray,
        lambda_: float,
        fit_intercept: bool,
        regularize_intercept: bool,
        n_feat_ignore_reg: int,
    ) -> np.ndarray:
        """
        Computes the normal equations for OLS: (X^T X)^{-1} X^T
        """
        # Inverting X^T X directly is computationally expensive and mathematically unstable, so we use QR factorization
        # which factors x into the sum of an orthogonal matrix Q and a upper tringular matrix R
        # L is a diagonal matrix of weights
        Q, R = np.linalg.qr(L @ x)

        # get regularization matrix
        lambda_I = self._get_regularizer(lambda_, R.shape[0], fit_intercept, regularize_intercept, n_feat_ignore_reg)

        # substitute X = QR into the normal equations to get
        #       R^T Q^T Q R \beta = R^T Q^T y
        #       R^T R \beta = R^T Q^T y
        #       \beta = (R^T R)^{-1} R^T Q^T y
        # since R is upper triangular it is eqsier to invert
        # lambda_I is the regularization matrix
        return np.linalg.inv(R.T @ R + lambda_I) @ R.T @ Q.T

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
        lambda_: float = 0.0,
        normal_eqs: np.ndarray | None = None,
        fit_intercept: bool = True,
        regularize_intercept: bool = False,
        n_feat_ignore_reg: int = 0,
    ):
        self._check_any_element_nan_or_inf(x)
        self._check_any_element_nan_or_inf(y)

        if fit_intercept:
            self._check_intercept(x)

        # if weights are none, all rows should be weighted equally
        if weights is None:
            weights = np.ones((y.shape[0],))

        # normalize weights and turn into diagional matrix
        # square root because will be squared when R^T R happens later
        L = np.diag(np.sqrt(weights.flatten() / weights.sum()))

        # if normal equations are provided then use those, otherwise compute them
        # in the bootstrap setting we can now pass in the normal equations and can
        # save time re-computing them
        if normal_eqs is None:
            self.normal_eqs = self._compute_normal_equations(
                x, L, lambda_, fit_intercept, regularize_intercept, n_feat_ignore_reg
            )
        else:
            self.normal_eqs = normal_eqs

        # compute hat matrix: X (X^T X)^{-1} X^T
        self.hat_vals = np.diag(x @ self.normal_eqs @ L)

        # compute coefficients: (X^T X)^{-1} X^T y
        self.coefficients = self.normal_eqs @ L @ y

    def residuals(self, y: np.ndarray, y_hat: np.ndarray, loo: bool = True, center: bool = True) -> np.ndarray:
        """
        Computes residuals for the model
        """
        # compute standard residuals
        residuals = y - y_hat

        # if leave one out is True, inflate by (1 - P)
        # in OLS setting inflating by (1 - P) is the same as computing the leave one out residuals
        # the un-inflated training residuals are too small, since training covariates were observed during fitting
        if loo:
            residuals /= (1 - self.hat_vals).reshape(-1, 1)

        # centering removes the column mean
        if center:
            residuals -= np.mean(residuals, axis=0)

        return residuals
