import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import warnings
import tensorflow as tf
import multiprocessing
import time
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR') # Suppress TensorFlow warnings
# ----------------- Configuration -----------------
TICKERS = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'ITC.NS',
           'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS']
START_DATE = '2021-01-01'
END_DATE = '2025-07-01'
ESTIMATION_WINDOW = 24 # months
TEST_WINDOW = 1 # months
CARDINALITY = 5 # max assets in portfolio (adjusted for 10 stocks)
MAX_WEIGHT = 0.3 # increased slightly to allow flexibility
TRANSACTION_COST = 0.001
RISK_FREE_RATE = 0.03
CONFIDENCE_LEVEL = 0.95
NUM_PARTICLES = 50
MAX_ITERATIONS = 10

from src.data import download_monthly_prices, monthly_returns
from src.lstm_model import parallel_lstm_forecast, forecast_with_lstm, build_lstm_model, prepare_lstm_data
from src.optimization import pso_optimize, mvo_optimize, select_by_momentum
from src.metrics import annualized_sharpe, annualized_sortino, var_cvar, herfindahl, turnover, calculate_metrics
from src.backtest import run_backtest
from src.visualization import plot_weights, plot_results

if __name__ == "__main__":
    print("Running portfolio optimization with LSTM and PSO on 10 Nifty 50 stocks...")
    start_time = time.time()
    try:
        prices = download_monthly_prices(TICKERS, START_DATE, END_DATE)
        if prices.empty:
            print("No price data available. Exiting.")
        else:
            results, weights_history = run_backtest(prices)
            if results.empty:
                print("No backtest results generated.")
            else:
                print("\nPerformance Summary:")
                print(results[['Phase', 'Date', 'PSO_NetReturn', 'MVO_NetReturn', 'EW_NetReturn',
                               'Forecast_Uncertainty']].to_string(index=False))
                print("\nAverage Metrics:")
                avg_metrics = results[[
                    'PSO_Sharpe', 'MVO_Sharpe', 'EW_Sharpe',
                    'PSO_Sortino', 'MVO_Sortino', 'EW_Sortino',
                    'PSO_VaR95', 'MVO_VaR95', 'EW_VaR95',
                    'PSO_HHI', 'MVO_HHI', 'EW_HHI'
                ]].mean()
                print(avg_metrics.to_string())
                plot_results(results, weights_history)
    except Exception as e:
        print(f"Error in main execution: {e}")
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")
