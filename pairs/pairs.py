import data_getter.data as data
from statsmodels.tsa.stattools import adfuller

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt


class Stock:
    def __init__(self, ticker: str):
        self.data = data.get_data(ticker)
        self.name = ticker

        self.returns = self._calc_returns()

    def _calc_returns(self):
        return self.data.Close.pct_change()


class Pair:
    def __init__(self, tickers: tuple):
        self.stock_names = tickers
        self.stocks = (Stock(tickers[0]), Stock(tickers[1]))

        self.corr_mat = self._calc_correlation()
        self.correlation = self.corr_mat.to_numpy()[0, 1]
        self.spread = self._estimate_spread()

    def _calc_correlation(self):
        df = pd.concat([self.stocks[0].returns, self.stocks[1].returns], axis=1)
        df.columns = [self.stock_names[0], self.stock_names[1]]
        corr_mat = df.corr()
        return corr_mat

    def _estimate_spread(self):
        x = np.log(self.stocks[0].data.Close)
        y = np.log(self.stocks[1].data.Close)
        model = LinearRegression()
        x_const = pd.concat([x, pd.Series([1] * x.shape[0], index=x.index)], axis=1)
        x_const.columns = x_const.columns.astype(str)
        model.fit(x_const, y)
        n = model.coef_[0]
        i = model.intercept_
        spread = y - n * x - i
        return spread

    def plot_prices(self):
        prices = pd.concat(
            [self.stocks[0].data.Close, self.stocks[1].data.Close], axis=1
        )
        prices.columns = [self.stock_names[0], self.stocks_names[1]]
        prices.plot(figsize=(15, 10))
        plt.show()

    def plot_spread(self):
        self.spread.plot()
        plt.show()

    def is_stationary(self, sig_level=0.05, verbose=1):
        result = adfuller(self.spread, maxlag=1)
        if verbose:
            print(f"ADF test stat: {result[0]}")
            for k, v in result[4].items():
                print(f"\t{k}: {v}")
            print(f"p-value: {result[1]}")

        if (result[0] < float(list(result[4].keys())[0][0])) and (
            result[1] < sig_level
        ):
            print("Data has no unit root. And therefore is stationary")
        else:
            print("Data has a unit root. And therefore is not stationary")
