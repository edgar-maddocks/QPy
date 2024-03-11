import numpy as np
import pandas as pd
import scipy


def rets(data):
    data["Returns"] = data["Close"].pct_change()
    return data


def vol(data):
    if "Returns" in data.columns:
        return data["Returns"].std()
    else:
        data = rets(data)
        vol(data)


def annualize_rets(returns, ppy, start_date=None, end_date=None):
    if start_date is None and end_date is None:
        annualized = ((1 + returns).prod()) ** (ppy / returns.shape[0]) - 1
    elif start_date is None and end_date is not None:
        if isinstance(end_date, str) is False:
            raise Exception("end date must be a string")
        annualized = ((1 + returns[:end_date]).prod()) ** (
            ppy / returns[:end_date].shape[0]
        ) - 1
    elif start_date is not None and end_date is None:
        if isinstance(start_date, str) is False:
            raise Exception("start date must be a string")
        annualized = ((1 + returns[start_date:]).prod()) ** (
            ppy / returns[start_date:].shape[0]
        ) - 1
    elif isinstance(start_date, str) and isinstance(end_date, str):
        annualized = ((1 + returns[start_date:end_date]).prod()) ** (
            12 / returns[start_date:end_date].shape[0]
        ) - 1
    return annualized


def annualize_vol(returns, ppy, start_date=None, end_date=None):
    if start_date is None and end_date is None:
        annualized = returns.std() * (ppy**0.5)
    elif start_date is None and end_date is not None:
        if isinstance(end_date, str) is False:
            raise Exception("end date must be a string")
        annualized = returns[:end_date].std() * (ppy**0.5)
    elif start_date is not None and end_date is None:
        if isinstance(start_date, str) is False:
            raise Exception("start date must be a string")
        annualized = returns[start_date:].std() * (ppy**0.5) * 100
    elif isinstance(start_date, str) and isinstance(end_date, str):
        annualized = returns[start_date:end_date].std() * (12**0.5)
    return annualized


def sharpe_ratio(returns, ppy, rfr):
    rf_per_period = (1 + rfr) ** (1 / ppy) - 1
    ret_minus_rf = returns - rf_per_period
    annualized_ret_minus_rf = annualize_rets(ret_minus_rf, ppy)
    annualized_vol = annualize_vol(ret_minus_rf, ppy)
    return annualized_ret_minus_rf / annualized_vol


def max_dd(returns, start_date=None, end_date=None):
    if start_date is None and end_date is None:
        wealth_index = 1000 * (1 + returns[start_date:end_date]).cumprod()
    elif start_date is None and end_date is not None:
        if isinstance(end_date, str) is False:
            raise Exception("end date must be a string")
        wealth_index = 1000 * (1 + returns[:end_date]).cumprod()
    elif start_date is not None and end_date is None:
        if isinstance(start_date, str) is False:
            raise Exception("start date must be a string")
        wealth_index = 1000 * (1 + returns[start_date:]).cumprod()
    elif isinstance(start_date, str) and isinstance(end_date, str):
        wealth_index = 1000 * (1 + returns[start_date:end_date]).cumprod()

    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    max_drawdown = drawdown.min()

    return max_drawdown, pd.DataFrame(
        {"Wealth": wealth_index, "Previous Peak": previous_peaks, "Drawdown": drawdown}
    )


def skewness(returns):
    demeaned_r = returns - returns.mean()
    sigma_r = returns.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp / sigma_r**3


def kurtosis(returns):
    demeaned_r = returns - returns.mean()
    sigma_r = returns.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp / sigma_r**4


def jarque_bera(returns, sig_level=0.01):
    jb_value, p_value = scipy.stats.jarque_bera(returns)
    return p_value > sig_level


def semi_deviation(returns):
    return returns[returns < 0].std(ddof=0)
