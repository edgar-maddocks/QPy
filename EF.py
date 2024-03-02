import stats
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def portfolio_return(weights, annualized_returns):
    return np.sum(weights * annualized_returns)

def portfolio_vol(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def portfolio_stats(weights, annualized_returns, covmatrix):
    returns = portfolio_return(weights, annualized_returns)
    vol = portfolio_vol(weights, covmatrix)
    return returns, vol

def get_rets_and_cov_matrix(data, period_per_year):
    ann_rets = stats.annualize_rets(data, period_per_year)
    cov_matrix = data.cov()

    return ann_rets, cov_matrix

def minimize_vol(target_return, returns, cov_matrix, weight_bounds = (0., 1.)):
    n_assets = returns.shape[0]
    init_guess = np.repeat(1/n_assets, n_assets)
    bounds = (weight_bounds,) * n_assets

    weights_sum_one = {
        "type" : "eq",
        "fun" : lambda w: np.sum(w) - 1
    }

    return_meets_target = {
        "type" : "eq",
        "args" : (returns, ),
        "fun" : lambda w, returns: target_return - portfolio_return(w, returns)
    }

    results = minimize(portfolio_vol,
                       init_guess,
                       args=(cov_matrix, ),
                       method="SLSQP",
                       options={"disp" : False},
                       constraints=(weights_sum_one, return_meets_target),
                       bounds=bounds)
    
    weights = results.x
    return weights
    

def plot_ef(data, ppy, target_return = None, n_points=2500, risk_free_rate=0, verbose=0):
    ann_rets, cov_mat = get_rets_and_cov_matrix(data, ppy)

    n_assets = ann_rets.shape[0]
    weights = np.random.rand(n_points, n_assets)
    weights = [x/sum(x) for x in weights]

    pt_rets = [portfolio_return(w, ann_rets) for w in weights]
    pt_vol = [portfolio_vol(w, cov_mat) for w in weights]

    ef = pd.DataFrame({"V" : pt_vol, "R" : pt_rets, "SR" : ((np.array(pt_rets) - risk_free_rate)/ np.array(pt_vol))})
    
    if isinstance(target_return, float):
        opt_weights = minimize_vol(target_return, ann_rets, cov_mat)
        opt_ret, opt_vol = portfolio_stats(opt_weights, ann_rets, cov_mat)

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    scat = axs.scatter(ef["V"], ef["R"], s = 5, c = ef["SR"])
    if isinstance(target_return, float):
        axs.scatter(opt_vol, opt_ret, marker = "*", s = 75, c = "black")
        axs.text(opt_vol, opt_ret, f"Minimal volatility for return of {target_return}")
    axs.set_xlabel("Volatility")
    axs.set_ylabel("Returns")
    axs.set_title("Efficient Frontier")
    fig.colorbar(scat, label="Portfolio's Sharpe Ratio")

    plt.show()

    if verbose and isinstance(target_return, float):
        return ef, opt_weights
    elif verbose:
        return ef

data = pd.read_csv("ind30_m_ew_rets.csv")
data = data.set_index("Unnamed: 0")
data.index = pd.to_datetime(data.index, format = "%Y%m")
data.columns = data.columns.str.strip()
data = data / 100
data = data[["Food", "Beer"]]

ef_data, opt_weights = plot_ef(data, 12, target_return=0.129, risk_free_rate=0.05, verbose=1)




