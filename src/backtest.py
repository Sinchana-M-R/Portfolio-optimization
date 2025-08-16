def run_backtest(prices):
    if prices.empty:
        print("No price data available. Exiting backtest.")
        return pd.DataFrame(), {}
    returns = monthly_returns(prices)
    dates = returns.index
    if len(dates) < ESTIMATION_WINDOW + 1:
        print("Insufficient data for backtest. Need at least", ESTIMATION_WINDOW + 1, "months.")
        return pd.DataFrame(), {}
    results = []
    prev_weights = {'pso': None, 'mvo': None, 'ew': None}
    weights_history = {'pso': [], 'mvo': [], 'ew': []}
    for p in range(len(dates) - ESTIMATION_WINDOW):
        est_period = dates[p:p+ESTIMATION_WINDOW]
        test_period = dates[p+ESTIMATION_WINDOW:p+ESTIMATION_WINDOW+TEST_WINDOW]
        if len(test_period) == 0:
            break
        selected = select_by_momentum(prices.loc[est_period])
        forecast_df = parallel_lstm_forecast(returns, est_period)
        mu = forecast_df['forecast']
        mu_lower = forecast_df['lower_bound']
        mu_upper = forecast_df['upper_bound']
        returns_train = returns.loc[est_period]
        cov_sel = returns_train[selected].cov().values
        mu_sel = mu[selected].values
        w_pso = pso_optimize(mu_sel, cov_sel, prev_weights.get('pso', {}))
        weights_pso = dict(zip(selected, w_pso))
        weights_history['pso'].append(weights_pso)
        cov_full = returns_train.cov().values
        w_mvo = mvo_optimize(mu.values, cov_full)
        weights_mvo = dict(zip(returns.columns, w_mvo))
        weights_history['mvo'].append(weights_mvo)
        weights_ew = {a: 1.0/len(returns.columns) for a in returns.columns}
        weights_history['ew'].append(weights_ew)
        test_returns = returns.loc[test_period]
        metrics = calculate_metrics(
            test_returns,
            {'PSO': weights_pso, 'MVO': weights_mvo, 'EW': weights_ew},
            prev_weights,
            returns_train
        )
        results.append({
            'Phase': p+1,
            'Date': test_period[0].strftime('%Y-%m'),
            'Forecast_Uncertainty': (mu_upper - mu_lower).mean(),
            **metrics
        })
        prev_weights = {'pso': weights_pso, 'mvo': weights_mvo, 'ew': weights_ew}
    return pd.DataFrame(results), weights_history
