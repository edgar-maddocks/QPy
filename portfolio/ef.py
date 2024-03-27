from finstats import *
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def portfolio_return(weights, returns):
    """Returns the return of any given portfolio

    Returns:
        float: 1 + R format of the return of the portfolio
    """
    return np.sum(weights * returns)


def portfolio_vol(weights, cov_matrix):
    """Returns the volatility of any given portfolio

    Returns:
        float: volatility of the portfolio
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def portfolio_stats(weights, returns, cov_matrix):
    """Returns both the return and volatility of the portfolio

    Returns:
        tuple<float, float>: (return, volatility)
    """
    returns = portfolio_return(weights, returns)
    vol = portfolio_vol(weights, cov_matrix)
    return returns, vol


def neg_portfolio_sr(weights, returns, cov_matrix, risk_free_rate):
    rets = portfolio_return(weights, returns)
    vol = portfolio_vol(weights, cov_matrix)

    return -((rets - risk_free_rate) / vol)


class Portfolio:
    """Class which represents a portfolio"""

    def __init__(
        self, returns, cov_matrix, weights, risk_free_rate=0.0, weight_bounds=(0.0, 1.0)
    ):
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.weights = weights
        self.risk_free_rate = risk_free_rate
        self.weight_bounds = weight_bounds

        self.n_assets = len(self.returns)

    def portfolio_return(self):
        """Returns the return of the portfolio

        Returns:
            float: 1 + R format of the return of the portfolio
        """
        return np.sum(self.weights * self.returns)

    def portfolio_vol(self):
        """Returns the volatility of the portfolio

        Returns:
            float: volatility of the portfolio
        """
        return np.sqrt(np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights)))

    def portfolio_stats(self):
        """Returns both the return and volatility of the portfolio

        Returns:
            tuple<float, float>: (return, volatility)
        """
        returns = self.portfolio_return()
        vol = self.portfolio_vol()
        return returns, vol

    def minimize_vol(self, target_return, labels=False):
        """Minimizes volatility of a portfolio to meet a certain return

        Args:
            target_return (float): value between 0 and 1

        Returns:
            array<float>: array of weights
        """
        init_guess = np.repeat(1 / self.n_assets, self.n_assets)
        bounds = (self.weight_bounds,) * self.n_assets

        weights_sum_one = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        return_meets_target = {
            "type": "eq",
            "args": (self.returns,),
            "fun": lambda weights, returns: target_return
            - portfolio_return(weights, returns),
        }

        results = minimize(
            portfolio_vol,
            init_guess,
            args=(self.cov_matrix,),
            method="SLSQP",
            options={"disp": False},
            constraints=(weights_sum_one, return_meets_target),
            bounds=bounds,
        )

        weights = results.x
        if labels:
            weights = pd.Series(
                results.x, index=self.returns.index, name="min_var_weights"
            )
        return weights

    def maximize_sr(self, labels=False):
        """Finds weights for maximum SR portfolio

        Returns:
            array<float>: array of weights
        """
        init_guess = np.repeat(1 / self.n_assets, self.n_assets)
        bounds = (self.weight_bounds,) * self.n_assets

        weights_sum_one = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        results = minimize(
            neg_portfolio_sr,
            init_guess,
            args=(self.returns, self.cov_matrix, self.risk_free_rate),
            method="SLSQP",
            options={"disp": False},
            constraints=(weights_sum_one),
            bounds=bounds,
        )

        weights = results.x
        if labels:
            weights = pd.Series(
                results.x, index=self.returns.index, name="max_sr_weights"
            )
        return weights

    def gmv(self, labels=False):
        """Finds weights for the portfolio with the lowest volatility

        Returns:
            array<float>: array of weights
        """
        init_guess = np.repeat(1 / self.n_assets, self.n_assets)
        bounds = (self.weight_bounds,) * self.n_assets

        weights_sum_one = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        returns = np.repeat(1, self.cov_matrix.shape[0])

        results = minimize(
            neg_portfolio_sr,
            init_guess,
            args=(
                returns,
                self.cov_matrix,
                self.risk_free_rate,
            ),
            method="SLSQP",
            options={"disp": False},
            constraints=(weights_sum_one,),
            bounds=bounds,
        )

        weights = results.x
        if labels:
            weights = pd.Series(results.x, index=self.returns.index, name="gmv_weights")
        return weights

    def plot_ef(
        self,
        target_return=None,
        plot_max_SR=False,
        plot_ew=False,
        plot_gmv=False,
        draw_cml=False,
        n_points=2500,
        verbose=0,
    ):
        if verbose:
            return_args = {}

        print("Getting weights for EF - this may take a moment")
        ef_weights = []
        target_returns = np.linspace(self.returns.min(), self.returns.max(), n_points)
        for target in target_returns:
            ef_weights.append(self.minimize_vol(target))

        pt_rets = [portfolio_return(w, self.returns) for w in ef_weights]
        pt_vol = [portfolio_vol(w, self.cov_matrix) for w in ef_weights]

        ef = pd.DataFrame(
            {
                "V": pt_vol,
                "R": pt_rets,
                "SR": ((np.array(pt_rets) - self.risk_free_rate) / np.array(pt_vol)),
            }
        )
        if verbose:
            return_args["Efficient Frontier Data"] = ef

        fig, axs = plt.subplots(1, 1, figsize=(8, 6))

        # xmin = ef["V"].min() - 0.1 * (ef["V"].max() - ef["V"].min())
        # xmax = ef["V"].max() - 0.1 * (ef["V"].max() - ef["V"].min())
        # axs.set_xlim(xmin, xmax)

        # ymin = ef["R"].min() - 0.1 * (ef["R"].max() - ef["R"].min())
        # ymax = ef["R"].max() - 0.1 * (ef["R"].max() - ef["R"].min())
        # axs.set_ylim(ymin, ymax)

        scat = axs.scatter(ef["V"], ef["R"], s=5, c=ef["SR"])

        if isinstance(target_return, float):
            min_vol_weights = self.minimize_vol(target_return)
            min_vol_ret, min_vol_vol = portfolio_stats(
                min_vol_weights, self.returns, self.cov_matrix
            )
            if verbose:
                return_args[f"Min Vol"] = {
                    "weights": min_vol_weights,
                    "ret": min_vol_ret,
                    "vol": min_vol_ret,
                }
            axs.scatter(min_vol_vol, min_vol_ret, marker="*", s=75, c="black")
            axs.text(
                min_vol_vol,
                min_vol_ret,
                f"Minimal volatility for return of {target_return}",
            )
        if plot_max_SR:
            max_sr_weights = self.maximize_sr()
            max_sr_ret, max_sr_vol = portfolio_stats(
                max_sr_weights, self.returns, self.cov_matrix
            )
            if verbose:
                return_args["Max SR"] = {
                    "weights": max_sr_weights,
                    "ret": max_sr_ret,
                    "vol": max_sr_vol,
                }
            axs.scatter(max_sr_vol, max_sr_ret, marker="*", s=75, c="black")
            axs.text(max_sr_vol, max_sr_ret, "Max SR portfolio")
            if draw_cml:
                cml_x = [0, max_sr_vol]
                cml_y = [self.risk_free_rate, max_sr_ret]
                axs.plot(
                    cml_x,
                    cml_y,
                    c="red",
                    label="Capital Market Line",
                    linestyle="dashed",
                )
                axs.legend()
        if plot_ew:
            ew_weights = np.repeat(1 / self.n_assets, self.n_assets)
            ew_ret = portfolio_return(ew_weights, self.returns)
            ew_vol = portfolio_vol(ew_weights, self.cov_matrix)
            if verbose:
                return_args["EW"] = {
                    "weights": ew_weights,
                    "ret": ew_ret,
                    "vol": ew_vol,
                }
            axs.scatter(ew_vol, ew_ret, marker="*", s=75, c="black")
            axs.text(ew_vol, ew_ret, "EW portfolio")
        if plot_gmv:
            gmv_weights = self.gmv()
            gmv_ret = portfolio_return(gmv_weights, self.returns)
            gmv_vol = portfolio_vol(gmv_weights, self.cov_matrix)
            if verbose:
                return_args["GMV"] = {
                    "weights": gmv_weights,
                    "ret": gmv_ret,
                    "vol": gmv_vol,
                }
            axs.scatter(gmv_vol, gmv_ret, marker="*", s=75, c="black")
            axs.text(gmv_vol, gmv_ret, "GMV portfolio")

        axs.set_xlabel("Volatility")
        axs.set_ylabel("Returns")
        axs.set_title("Efficient Frontier")
        fig.colorbar(scat, label="Portfolio SR")

        plt.show()

        if verbose:
            return return_args
