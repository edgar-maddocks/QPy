import yfinance as yf
import pandas as pd
import pandas_datareader as pdr
import datetime as dt

def get_data(tickers, interval="1D", n_years = 5):
    all_data = {}
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days = (365 * n_years))
    yf.pdr_override()
    for ticker in tickers:
        data = pdr.get_data_yahoo(ticker, start = start_date, end = end_date, interval = interval)
        all_data[ticker] = data
    return all_data