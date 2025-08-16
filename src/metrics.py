import numpy as np

def annualized_sharpe(returns):
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - (RISK_FREE_RATE / 12)
    return np.sqrt(12) * excess_returns.mean() / (excess_returns.std() + 1e-10)

def annualized_sortino(returns):
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return np.inf
    excess_returns = returns - (RISK_FREE_RATE / 12)
    return np.sqrt(12) * excess_returns.mean() / (downside.std() + 1e-10)

def var_cvar(returns, alpha=0.95):
    if len(returns) == 0:
        return 0.0, 0.0
    var = np.percentile(returns, 100*(1-alpha))
    cvar = returns[returns <= var].mean()
    return abs(var), abs(cvar)

def herfindahl(weights):
    w = np.array(list(weights.values()))
    return float(np.sum(w**2))

def turnover(prev_weights, new_weights):
    if prev_weights is None:
        return 0.0
    all_assets = set(prev_weights.keys()) | set(new_weights.keys())
    return sum(abs(new_weights.get(a, 0) - prev_weights.get(a, 0)) for a in all_assets)

def calculate_metrics(test_returns, weights_dict, prev_weights_dict, train_returns):
    metrics = {}
    for strategy, weights in weights_dict.items():
        common_assets = [a for a in weights.keys() if a in test_returns.columns]
        if not common_assets:
            ret = 0.0
            strat_train_returns = pd.Series(0, index=train_returns.index)
        else:
            w_series = pd.Series(weights)[common_assets]
            ret = test_returns[common_assets].dot(w_series).iloc[0]
            strat_train_returns = train_returns[common_assets].dot(w_series)
        prev_weights = prev_weights_dict.get(strategy.lower(), {})
        to = turnover(prev_weights, weights)
        net_ret = ret - TRANSACTION_COST * to
        sharpe = annualized_sharpe(strat_train_returns)
        sortino = annualized_sortino(strat_train_returns)
        var, cvar = var_cvar(strat_train_returns)
        hhi = herfindahl(weights)
        metrics.update({
            f'{strategy}_Return': ret,
            f'{strategy}_NetReturn': net_ret,
            f'{strategy}_Sharpe': sharpe,
            f'{strategy}_Sortino': sortino,
            f'{strategy}_VaR95': var,
            f'{strategy}_CVaR95': cvar,
            f'{strategy}_HHI': hhi,
            f'{strategy}_Turnover': to
        })
    return metrics
