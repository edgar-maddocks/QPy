import finstats
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt


def var_historic(returns, level=5):
    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(var_historic, level=level)
    elif isinstance(returns, pd.Series):
        return -np.percentile(returns, level)
    else:
        raise TypeError("Expected returns to be of type pd.Series or pd.DataFrame")


def var_gaussian(returns, level=5, cornish_fischer_z=False):
    z = scipy.stats.norm.ppf(level / 100)
    if cornish_fischer_z:
        s = finstats.skewness(returns)
        k = finstats.kurtosis(returns)
        z = (
            z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * (k - 3) / 24
            - (2 * z**3 - 5 * z) * (s**2) / 36
        )
    return -(returns.mean() + z * returns.std(ddof=0))


def compare_vars(returns):
    comparison = pd.concat(
        [
            var_gaussian(returns),
            var_gaussian(returns, cornish_fischer_z=True),
            var_historic(returns),
        ],
        axis=1,
    )
    comparison.columns = ["Gaussian", "Cornish Fischer", "Historic"]
    ax = comparison.plot.bar(title="Comparison of VaR approximations")
    return comparison, ax


def cvar_historic(returns, level=5):
    if isinstance(returns, pd.Series):
        is_beyond = returns <= -var_historic(returns, level)
        return -returns[is_beyond].mean()
    elif isinstance(returns, pd.Series):
        return returns.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected returns to be of type pd.Series or pd.DataFrame")
