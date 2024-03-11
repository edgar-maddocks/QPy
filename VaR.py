import stats
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
        s = stats.skewness(returns)
        k = stats.kurtosis(returns)
        z = (
            z
            + s * ((z**2 - 1) / 6)
            + (k - 3) * ((z**3 - 3 * z) / 24)
            - (s**2) * (((2 * (z**3)) - 5 * z) / 36)
        )
    return -(returns.mean() + z * returns.std(ddof=0))


def compare_vars(returns):
    comparison = pd.concat(
        [
            var_gaussian(hfi),
            var_gaussian(hfi, cornish_fischer_z=True),
            var_historic(hfi),
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


hfi = pd.read_csv("edhec-hedgefundindices.csv", header=0, index_col=0, parse_dates=True)
hfi = hfi / 100
hfi.index = hfi.index.to_period("M")

hfi = hfi["2000":]

print(var_historic(hfi, level=1))
