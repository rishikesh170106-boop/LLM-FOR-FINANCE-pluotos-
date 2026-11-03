"""
src/features/technical_features.py
Extracts strictly lagged, leakage-free technical indicators from historical prices:
- Lagged returns (t-1 to t-5)
- Lagged open, high, low price ratios
- Lagged volume (log-transformed)
- 5-day rolling momentum, return mean, and volatility
- 14-day Relative Strength Index (RSI) using strictly lagged prices
All features for prediction on date t use data strictly from dates < t.
"""
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

STOCKNET_PRICE_DIR = Path("data/raw/stocknet-dataset/price/preprocessed")

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes Relative Strength Index (RSI) on a lagged price/return series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def extract_lagged_features_for_ticker(ticker: str, lookback: int = 5) -> pd.DataFrame:
    """Extracts strictly lagged features for a single stock."""
    price_file = STOCKNET_PRICE_DIR / f"{ticker}.txt"
    if not price_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(price_file, sep=r'\s+', header=None)
    df.columns = ["date", "movement", "open_return", "high_return", "low_return", "price_change", "volume"]
    df["date"] = df["date"].astype(str)
    df = df.sort_values("date").reset_index(drop=True)

    # All features for date t are computed strictly from t-1, t-2, ...
    features = pd.DataFrame()
    features["date"] = df["date"]
    features["ticker"] = ticker

    # 1. Individual Lagged Returns and Prices (1 to lookback days prior)
    lag_ret_cols = []
    for k in range(1, lookback + 1):
        ret_col = f"lag_{k}_return"
        features[ret_col] = df["movement"].shift(k)
        features[f"lag_{k}_open_ret"] = df["open_return"].shift(k)
        features[f"lag_{k}_high_ret"] = df["high_return"].shift(k)
        features[f"lag_{k}_low_ret"] = df["low_return"].shift(k)
        features[f"lag_{k}_log_volume"] = np.log1p(df["volume"].shift(k))
        lag_ret_cols.append(ret_col)

    # 2. Rolling Summary Features over the lookback window (t-lookback to t-1)
    lag_rets_df = features[lag_ret_cols]
    features["roll_return_mean_5d"] = lag_rets_df.mean(axis=1)
    features["roll_return_std_5d"] = lag_rets_df.std(axis=1)
    features["roll_momentum_5d"] = lag_rets_df.sum(axis=1)

    # 3. Lagged RSI (computed on shifted return series so today's price is excluded)
    lagged_series = df["movement"].shift(1)
    features["lagged_rsi_14"] = compute_rsi(lagged_series, period=14)

    return features

def extract_all_technical_features(tickers: List[str]) -> pd.DataFrame:
    """Extracts and concatenates technical features across all active stocks."""
    print(f"[Features] Extracting lagged technical features across {len(tickers)} stocks...")
    all_dfs = []
    for ticker in tickers:
        tdf = extract_lagged_features_for_ticker(ticker)
        if not tdf.empty:
            all_dfs.append(tdf)

    if not all_dfs:
        return pd.DataFrame()

    full_features = pd.concat(all_dfs, ignore_index=True)
    full_features.sort_values(["date", "ticker"], inplace=True)
    full_features.reset_index(drop=True, inplace=True)
    return full_features

if __name__ == "__main__":
    t_dir = Path("data/raw/stocknet-dataset/tweet/preprocessed")
    sample_tickers = sorted([d.name for d in t_dir.iterdir() if d.is_dir()])
    feats = extract_all_technical_features(sample_tickers)
    print(f"Extracted {len(feats):,} feature rows with {len(feats.columns)} columns.")
    print(f"Sample columns: {list(feats.columns[:10])}")
