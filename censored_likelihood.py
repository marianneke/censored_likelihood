"""
Maximum Likelihood Estimator for censored observations.

This is an estimator based on the GenericLikelihoodModel class in statsmodels
which takes into account that some observations may be right-censored, as is
often the case in survival analysis.

:Authors: Marianne Hoogeveen <marianne.hoogeveen@gmail.com>
"""

import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel


def _initial_values_params(target, x):
    """Use OLS regression to initialize parameters."""
    return sm.OLS(target, x).fit().params


def _ll_ols_obs(target, x, beta, sigma):
    """
    Compute the log-likelihood for non-censored observations.

    :param list-like target: observed target variable
    :param matrix x: features
    :param list-like beta: parameters
    :param float sigma: standard deviation
    :return: log-likelihood for non-censored variables
    """
    mu = x.dot(beta)
    return norm(mu, sigma).logpdf(target).sum()


def _ll_ols_cens(target, x, beta, sigma):
    """
    Compute the log-likelihood for censored observations.

    :param list-like target: censored target variable
    :param matrix x: features
    :param list-like beta: parameters
    :param float sigma: standard deviation
    :return: log-likelihood for censored variables
    """
    mu = x.dot(beta)
    return norm(mu, sigma).logsf(target).sum()


def _ll_ols(y, x, beta, sigma):
    """
    Compute log-likelihood with possibly censored data.

    :param matrix-like y: matrix where the first column is the target variable,
        and the second column is a boolean column indicating whether the target
        variable is observed (1) or censored (0)
    :param matrix x: features
    :param list-like beta: parameters
    :param float sigma: standard deviation
    :return: log-likelihood
    """
    if y.shape[1] == 1:
        return _ll_ols_obs(y, x, beta, sigma)

    target, obs = y[:, 0], y[:, 1]
    target_obs, x_obs = target[obs], x[obs]
    target_cens, x_cens = y[~obs], x[~obs]

    ll_obs = _ll_ols_obs(target_obs, x_obs, beta, sigma)
    ll_cens = _ll_ols_cens(target_cens, x_cens, beta, sigma)

    return ll_obs + ll_cens


class CensoredLikelihoodOLS(GenericLikelihoodModel):
    """Maximum likelihood estimation for censored data."""

    def __init__(self, endog, exog, **kwds):
        """
        Maximum likelihood estimation for censored data.

        This class defined a model for estimating OLS parameters using maximum
        likelihood estimation for target data that may be right-censored, for
        instance in survival data where some subjects are still alive.

        :param endog: contains the target variable in the first column, and a
            boolean column indicating whether the subject has been observed (1)
            or is censored (0)
        :param exog: any independent variables that are to be included in the
            model. If an intercept is to be fitted, this should be added
            explicitly as a column containing ones for all observations
        """
        super(CensoredLikelihoodOLS, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        """Compute the negative log likelihood to pass into the optimizer."""
        sigma = params[-1]
        beta = params[:-1]
        ll = _ll_ols(self.endog, self.exog, beta, sigma)
        return -ll

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        """Fit OLS using maximum likelihood estimation."""
        # we have one additional parameter and we need to add it for summary
        self.exog_names.append('sigma')
        if start_params is None:
            # Initialize starting values using the results of OLS regression
            start_params = np.append(
                _initial_values_params(self.endog[:, 0], self.exog), .5)
        return super(CensoredLikelihoodOLS, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds)
