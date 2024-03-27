import pandas as pd
import termtables as tt
from tqdm import tqdm
from typing import List
import numpy as np


class Order:
    def __init__(self, side: int, size: float, idx) -> None:
        self.side = side
        self.size = size
        self.idx = idx


class Trade:
    def __init__(self, side: int, size: float, price: float, idx) -> None:
        self.side = side
        self.size = size
        self.idx = idx
        self.price = price


class Strategy:
    def __init__(self, data: pd.DataFrame) -> None:
        self.curr_idx = None
        self.data = data

        self.orders: List[Order] = []
        self.trades: List[Trade] = []

    def step(self) -> None:
        """
        Must be implemented by user
        """
        raise NotImplementedError

    def buy(self, size: float) -> None:
        self.orders.append(Order(side=0, size=size, idx=self.curr_idx))

    def sell(self, size: float) -> None:
        self.orders.append(Order(side=1, size=-size, idx=self.curr_idx))

    @property
    def total_position_size(self) -> int:
        return np.sum([t.size for t in self.trades])


class Btester:
    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        init_balance: int = 1000,
        hedging: bool = False,
        clear_orders: bool = True,
    ):
        self.init_balance = init_balance
        self.equity = init_balance
        self.balance = init_balance

        self.data = data

        self.strat = strategy

        self.curr_idx = None

        self.hedging = hedging
        self.clear_orders = clear_orders

    def run(self):
        for idx in tqdm(self.data.index):
            self.curr_idx = idx

            self.strat.curr_idx = self.curr_idx
            self.strat.step()

            self._fill_orders()

            self._calculate_equity()

        string = tt.to_string(
            [
                ["Final Equity", self.equity],
                ["Total Return [$]", round(self.equity - self.init_balance, 5)],
                [
                    "Total Return [%]",
                    round(
                        ((self.equity - self.init_balance) / self.init_balance) * 100, 5
                    ),
                ],
            ],
            header=["Statistic", "Value"],
            style=tt.styles.ascii_thin_double,
        )
        print(string)

    def _fill_orders(self):
        for order in self.strat.orders:
            fillable = False
            if self.hedging is False:
                if (
                    order.side == 0
                    and self.balance
                    >= self.data.loc[self.curr_idx]["Open"] * order.size
                ):
                    fillable = True
                if order.side == 1 and self.strat.total_position_size >= order.size:
                    fillable = True

            if fillable:
                trade = Trade(
                    side=order.side,
                    size=order.size,
                    price=self.data.loc[self.curr_idx]["Open"],
                    idx=self.curr_idx,
                )

                self.strat.trades.append(trade)

                self.balance -= trade.size * trade.price
                self.equity = self._calculate_equity()

        if self.clear_orders:
            self.strat.orders = []

    def _calculate_equity(self):
        self.equity = round(
            self.balance
            + (self.strat.total_position_size * self.data.loc[self.curr_idx]["Close"]),
            2,
        )

    def plot(self):
        raise NotImplementedError
