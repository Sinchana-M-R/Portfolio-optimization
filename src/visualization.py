import matplotlib.pyplot as plt

def plot_weights(weights_history):
    """Plot bar charts for weight distribution in the last phase"""
    if not weights_history:
        print("No weights history to plot.")
        return
    strategies = ['pso', 'mvo', 'ew']
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, strategy in enumerate(strategies):
        last_weights = weights_history[strategy][-1]
        assets = list(last_weights.keys())
        weights = list(last_weights.values())
        axs[i].bar(assets, weights, color='skyblue')
        axs[i].set_title(f'{strategy.upper()} Weights (Last Phase)')
        axs[i].set_xlabel('Assets')
        axs[i].set_ylabel('Weight')
        axs[i].tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.show()

def plot_results(results, weights_history):
    """Plot cumulative returns, sharpe, HHI pie, and call weight plots"""
    if results.empty:
        print("No results to plot.")
        return
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    for strategy in ['PSO', 'MVO', 'EW']:
        cum_ret = (1 + results[f'{strategy}_NetReturn']).cumprod()
        ax1.plot(results['Date'], cum_ret, label=strategy, marker='o')
    uncertainty = results['Forecast_Uncertainty']
    ax1.scatter(results['Date'], [1.05] * len(results),
               c=uncertainty, cmap='Reds', alpha=0.5,
               s=uncertainty*500, label='Forecast Uncertainty')
    ax1.set_title('Cumulative Returns with Forecast Uncertainty')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)
    sharpe_ratios = {
        'PSO': results['PSO_Sharpe'],
        'MVO': results['MVO_Sharpe'],
        'EW': results['EW_Sharpe']
    }
    for strategy, sharpe in sharpe_ratios.items():
        ax2.plot(results['Date'], sharpe, label=strategy, marker='o')
    ax2.set_title('Sharpe Ratio Comparison Across Strategies')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Annualized Sharpe Ratio')
    ax2.legend()
    ax2.grid(True)
    ax2.tick_params(axis='x', rotation=45)
    avg_hhi = {
        'PSO': results['PSO_HHI'].mean(),
        'MVO': results['MVO_HHI'].mean(),
        'EW': results['EW_HHI'].mean()
    }
    ax3.pie(avg_hhi.values(), labels=avg_hhi.keys(), autopct='%1.1f%%', startangle=90)
    ax3.set_title('Average Portfolio Concentration (HHI)')
    plt.tight_layout()
    plt.show()
    plot_weights(weights_history)
