import pandas as pd
import yfinance as yf
import time

def download_monthly_prices(tickers, start, end, retries=3):
    """Download and clean price data from Yahoo Finance with retries"""
    for attempt in range(retries):
        try:
            raw = yf.download(tickers, start=start, end=end, progress=False, ignore_tz=True)
            if raw.empty:
                raise ValueError("No data retrieved from Yahoo Finance. Check tickers or date range.")
            df = raw['Adj Close'] if 'Adj Close' in raw.columns.levels[0] else raw['Close']
            monthly = df.resample('ME').last()
            monthly.columns = [c.upper() for c in monthly.columns]
            valid_tickers = monthly.columns[~monthly.isna().all()].tolist()
            if not valid_tickers:
                raise ValueError("No valid data for any tickers after cleaning.")
            print("Downloaded tickers:", valid_tickers)
            print("Number of tickers:", len(valid_tickers))
            print("Date Range:", monthly.index.min(), "to", monthly.index.max())
            return monthly[valid_tickers]
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2) # Wait before retrying
            else:
                print("All retries failed.")
                return pd.DataFrame()

def monthly_returns(prices):
    """Calculate monthly returns"""
    returns = prices.pct_change().dropna(how='all').fillna(0)
    return returns
