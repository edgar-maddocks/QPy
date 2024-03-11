import yfinance as yf
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import datetime as dt
import stats
import EF


def get_data(tickers, interval="1D", n_years=5):
    all_data = {}
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=(365 * n_years))
    yf.pdr_override()
    for ticker in tickers:
        data = pdr.get_data_yahoo(
            ticker, start=start_date, end=end_date, interval=interval
        )
        all_data[ticker] = data
    return all_data


def get_portfolio_data(tickers, interval="1D", n_years=5):
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=(365 * n_years))
    data = yf.download(tickers, start=start_date, end=end_date)
    data = data["Close"]

    returns = data.pct_change()
    cov_matrix = returns.cov()

    match interval:
        case "1D":
            returns = ((1 + returns).prod()) ** (252 / returns.shape[0]) - 1
        case "1W":
            returns = ((1 + returns).prod()) ** (52 / returns.shape[0]) - 1
        case "1M":
            returns = ((1 + returns).prod()) ** (12 / returns.shape[0]) - 1
        case "1Y":
            returns = ((1 + returns).prod()) ** (1 / returns.shape[0]) - 1

    return returns, cov_matrix



