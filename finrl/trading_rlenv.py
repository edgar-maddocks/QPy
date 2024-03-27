import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

class StocksEnv(gym.Env):
    def __init__(self, data, starting_balance=1000, order_size=1):
        """Init Function Of Stocks Env

        Args:
            data (pd.DataFrame): Dataframe containing candlestick-like data and other indicators - first 5 columns must be OHLCV
            starting_balance (int, optional): Starting balance for the environment. Defaults to 1000.
            order_size (int, optional): Number of shares to buy for each action. Defaults to 1.
        """
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.order_size = order_size
        self.data = data
        self.OHLCV = data.iloc[:,:5]
        self.position = 0
        self.current_time_step = 0

        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape = (self.data.shape[0], self.data.shape[1]))
        self.action_space = gym.spaces.Discrete(3) # 0 hold 1 buy 2 sell

        self.past_balances = []
        self.past_prices = []
        self.past_actions = []

    def step(self, action):
        """Execute action in env

        Args:
            action (int): Action to be executed in the env

        Returns:
            (np.array, int, bool, int): (next observation in the env, reward, if the env is finished, previous action taken)
        """
        action_taken = action
        current_price = self.data.iloc[self.current_time_step, 3]

        if action_taken == 1 and self.balance >= current_price:
            self.position += self.order_size
            self.balance -= current_price * self.order_size
        elif action_taken == 2 and self.position > 0:
            self.position -= self.order_size
            self.balance += current_price * self.order_size

        if self.position == 0:
            action_taken = 0

        self.current_time_step += 1
        done = self.current_time_step == len(self.data) - 1

        self.past_balances.append(self.balance)
        self.past_prices.append(current_price)
        self.past_actions.append(action_taken)

        reward = self.balance - self.starting_balance # reward = profit

        return self.data.iloc[self.current_time_step], reward, done, action_taken
    
    def render(self):
        """
            Renders the most recent environment run through
        """
        returns = np.diff(self.past_balances, prepend=self.starting_balance) / np.array(self.past_prices)
        cumulative_returns = np.cumsum(returns)

        cumulative_returns_plot = mpf.make_addplot(cumulative_returns, 
                                                   title = "Cumulative Returns", 
                                                   color = "blue",
                                                   panel = 0)
        
        prices = pd.DataFrame(self.past_prices, columns = ["Close"])
        actions = self.past_actions[-len(self.data):]
        prices["Action"] = actions
        
        signals = prices.copy()
        signals.loc[signals["Action"] == 0, "Close"] = None

        buy = signals.copy()
        sell = signals.copy()

        buy.loc[buy["Action"] == 2, "Close"] = None
        sell.loc[sell["Action"] == 1, "Close"] = None

        buy_plot = mpf.make_addplot(buy["Close"], 
                                    type = "scatter",
                                    marker = "^",
                                    markersize = 30,
                                    color = "g",
                                    panel = 1)
        
        sell_plot = mpf.make_addplot(sell["Close"], 
                                    type = "scatter",
                                    marker = "v",
                                    markersize = 30,
                                    color = "r",
                                    panel = 1)

        # Check if sell and buy actualy have signals
        sell_has_no_signals = len(sell[sell["Close"] > 0]) == 0
        buy_has_no_signals = len(buy[buy["Close"] > 0]) == 0

        if buy_has_no_signals:
            add_plots = [sell_plot, cumulative_returns_plot]
        elif sell_has_no_signals:
            add_plots = [buy_plot, cumulative_returns_plot]
        elif sell_has_no_signals and buy_has_no_signals:
            add_plots = [cumulative_returns_plot]
        else:
            add_plots = [buy_plot, sell_plot, cumulative_returns_plot]
        

        mpf.plot(self.OHLCV[-len(self.past_prices):],
                    type = "candle",
                    main_panel = 1,
                    volume = True,
                    volume_panel = 2,
                    style = "yahoo",
                    panel_ratios = (3, 6, 2),
                    addplot = add_plots)
        

    def reset(self):
        """Resets the environment

        Returns:
            np.array: first observation of the environment
        """
        self.balance = self.starting_balance
        self.current_time_step = 0
        self.position = 0
        self.past_balances = []
        self.past_prices = []

        return self.data.iloc[0]
    

        
        

