from data_getter import get_data
from backtesting import Btester, Strategy
import pandas_ta as ta

data = get_data("AAPL")
data["EMA"] = ta.ema(data["Close"], length=14)
data["SMA"] = ta.sma(data["Close"], length=28)


class MyStrat(Strategy):
    def step(self):
        if self.data["SMA"][-1] < self.data["EMA"][-1] and (
            self.total_position_size <= 0
        ):
            self.buy(1)
        if (
            self.data["SMA"][-1] > self.data["EMA"][-1]
            and self.total_position_size >= 0
        ):
            self.sell(1)


btester = Btester(MyStrat(data))
btester.run()
