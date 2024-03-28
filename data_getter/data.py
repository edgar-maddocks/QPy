import yfinance as yf
import datetime as dt


def get_data(tickers, interval="1D", n_years=5):
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=(365 * n_years))
    data = yf.download(tickers, start=start_date, end=end_date)

    return data


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


def get_pairs_data(tickers: tuple, interval="1D", n_years=5):
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=(365 * n_years))
    all_data = {}
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        all_data[ticker] = data

    return all_data
