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

def maximize_sr(returns, cov_matrix, risk_free_rate=0, weight_bounds = (0., 1.)):
    n_assets = returns.shape[0]
    init_guess = np.repeat(1/n_assets, n_assets)
    bounds = (weight_bounds,) * n_assets

    weights_sum_one = {
        "type" : "eq",
        "fun" : lambda w: np.sum(w) - 1
    }

    def neg_portfolio_sr(weights, returns, cov_matrix, risk_free_rate):
        rets = portfolio_return(weights, returns)
        vol = portfolio_vol(weights, cov_matrix)

        return -((rets - risk_free_rate) / vol)

    results = minimize(neg_portfolio_sr,
                       init_guess,
                       args=(returns, cov_matrix, risk_free_rate, ),
                       method="SLSQP",
                       options={"disp" : False},
                       constraints=(weights_sum_one),
                       bounds=bounds)
    
    weights = results.x
    return weights

def gmv(cov_matrix):
    n_assets = cov_matrix.shape[0]

    return maximize_sr(np.repeat(1, n_assets), cov_matrix)
    

def plot_ef(data, ppy, 
            target_return = None, plot_max_SR = False,
            plot_ew = False, plot_gmv = False,
            draw_cml = False, 
            n_points=2500, risk_free_rate=0, 
            weight_bounds=(0., 1.), verbose=0):
    if verbose:
        return_args = {}
    ann_rets, cov_mat = get_rets_and_cov_matrix(data, ppy)

    n_assets = ann_rets.shape[0]
    weights = np.random.rand(n_points, n_assets)
    weights = [x/sum(x) for x in weights]

    pt_rets = [portfolio_return(w, ann_rets) for w in weights]
    pt_vol = [portfolio_vol(w, cov_mat) for w in weights]

    ef = pd.DataFrame({"V" : pt_vol, "R" : pt_rets, "SR" : ((np.array(pt_rets) - risk_free_rate)/ np.array(pt_vol))})
    if verbose: 
        return_args["Efficient Frontier Data"] = ef

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    xmin = ef["V"].min()-0.1*(ef["V"].max()-ef["V"].min())
    xmax = ef["V"].max()-0.1*(ef["V"].max()-ef["V"].min())
    axs.set_xlim(xmin, xmax)

    ymin = ef["R"].min()-0.1*(ef["R"].max()-ef["R"].min())
    ymax = ef["R"].max()-0.1*(ef["R"].max()-ef["R"].min()) 
    axs.set_ylim(ymin, ymax)
    
    scat = axs.scatter(ef["V"], ef["R"], s = 5, c = ef["SR"])

    if isinstance(target_return, float):
        min_vol_weights = minimize_vol(target_return, ann_rets, cov_mat, weight_bounds)
        min_vol_ret, min_vol_vol = portfolio_stats(min_vol_weights, ann_rets, cov_mat)
        if verbose:
            return_args[f"Min Vol"] = {"weights" : min_vol_weights, "ret" : min_vol_ret, "vol" : min_vol_ret}
        axs.scatter(min_vol_vol, min_vol_ret, marker = "*", s = 75, c = "black")
        axs.text(min_vol_vol, min_vol_ret, f"Minimal volatility for return of {target_return}")
    if plot_max_SR:
        max_sr_weights = maximize_sr(ann_rets, cov_mat, risk_free_rate, weight_bounds)
        max_sr_ret, max_sr_vol = portfolio_stats(max_sr_weights, ann_rets, cov_mat)
        if verbose:
            return_args["Max SR"] = {"weights" : max_sr_weights, "ret" : max_sr_ret, "vol" : max_sr_vol}
        axs.scatter(max_sr_vol, max_sr_ret, marker = "*", s = 75, c = "black")
        axs.text(max_sr_vol, max_sr_ret, "Max SR portfolio")
        if draw_cml:
            cml_x = [0, max_sr_vol]
            cml_y = [risk_free_rate, max_sr_ret]
            axs.plot(cml_x, cml_y, c = "red", label = "Capital Market Line", linestyle="dashed")
            axs.legend()
    if plot_ew:
        ew_weights = np.repeat(1/n_assets, n_assets)
        ew_ret = portfolio_return(ew_weights, ann_rets)
        ew_vol = portfolio_vol(ew_weights, cov_mat)
        if verbose:
            return_args["EW"] = {"weights" : ew_weights, "ret" : ew_ret, "vol" : ew_vol}
        axs.scatter(ew_vol, ew_ret, marker= "*", s = 75, c ="black")
        axs.text(ew_vol, ew_ret, "EW portfolio")
    if plot_gmv:
        gmv_weights = gmv(cov_mat)
        gmv_ret = portfolio_return(gmv_weights, ann_rets)
        gmv_vol = portfolio_vol(gmv_weights, cov_mat)
        if verbose:
            return_args["GMV"] = {"weights" : gmv_weights, "ret" : gmv_ret, "vol" : gmv_vol}
        axs.scatter(gmv_vol, gmv_ret, marker= "*", s = 75, c ="black")
        axs.text(gmv_vol, gmv_ret, "GMV portfolio")

    axs.set_xlabel("Volatility")
    axs.set_ylabel("Returns")
    axs.set_title("Efficient Frontier")
    fig.colorbar(scat, label="Portfolio's Sharpe Ratio")

    plt.show()

    if verbose:
        return return_args
    
def ef_data(data, ppy,
            target_return = None,
            n_points=2500, 
            risk_free_rate=0, 
            weight_bounds=(0., 1.)):
    return_args = {}
    ann_rets, cov_mat = get_rets_and_cov_matrix(data, ppy)

    n_assets = ann_rets.shape[0]
    weights = np.random.rand(n_points, n_assets)
    weights = [x/sum(x) for x in weights]

    pt_rets = [portfolio_return(w, ann_rets) for w in weights]
    pt_vol = [portfolio_vol(w, cov_mat) for w in weights]

    ef = pd.DataFrame({"V" : pt_vol, "R" : pt_rets, "SR" : ((np.array(pt_rets) - risk_free_rate)/ np.array(pt_vol))}) 
    return_args["Efficient Frontier Data"] = ef

    if isinstance(target_return, float):
        min_vol_weights = minimize_vol(target_return, ann_rets, cov_mat, weight_bounds)
        min_vol_ret, min_vol_vol = portfolio_stats(min_vol_weights, ann_rets, cov_mat)
        return_args[f"Min Vol"] = {"weights" : min_vol_weights, "ret" : min_vol_ret, "vol" : min_vol_vol}

    max_sr_weights = maximize_sr(ann_rets, cov_mat, risk_free_rate, weight_bounds)
    max_sr_ret, max_sr_vol = portfolio_stats(max_sr_weights, ann_rets, cov_mat)
    return_args["Max SR"] = {"weights" : max_sr_weights, "ret" : max_sr_ret, "vol" : max_sr_vol}

    ew_weights = np.repeat(1/n_assets, n_assets)
    ew_ret = portfolio_return(ew_weights, ann_rets)
    ew_vol = portfolio_vol(ew_weights, cov_mat)
    return_args["EW"] = {"weights" : ew_weights, "ret" : ew_ret, "vol" : ew_vol}

    gmv_weights = gmv(cov_mat)
    gmv_ret = portfolio_return(gmv_weights, ann_rets)
    gmv_vol = portfolio_vol(gmv_weights, cov_mat)
    return_args["GMV"] = {"weights" : gmv_weights, "ret" : gmv_ret, "vol" : gmv_vol}

    return return_args

data = pd.read_csv("ind30_m_ew_rets.csv")
data = data.set_index("Unnamed: 0")
data.index = pd.to_datetime(data.index, format = "%Y%m")
data.columns = data.columns.str.strip()
data = data / 100
data = data[["Food", "Beer"]]

response = plot_ef(data, 12, risk_free_rate=0.05, plot_ew=True, plot_gmv=True, plot_max_SR=True, weight_bounds=(.1, .7))




