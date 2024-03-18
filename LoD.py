import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Stats

returns = pd.read_csv("ind30_m_vw_rets.csv", index_col=0, header=0)
returns.index = pd.to_datetime(returns.index, format="%Y%m")
returns = returns / 100

nfirms = pd.read_csv("ind30_m_nfirms.csv", index_col=0, header=0)
nfirms.index = pd.to_datetime(nfirms.index, format="%Y%m")

size = pd.read_csv("ind30_m_size.csv", index_col=0, header=0)
size.index = pd.to_datetime(size.index, format="%Y%m")

market_cap = nfirms * size
total_market_cap = market_cap.sum(axis=1)

capital_weights = market_cap.divide(total_market_cap, axis=0)

market_return = (capital_weights * returns).sum(axis=1)

total_market_index = Stats.max_dd(market_return)[1].Wealth

trailing_36_returns = market_return.rolling(36).aggregate(Stats.annualize_rets, ppy=12)
