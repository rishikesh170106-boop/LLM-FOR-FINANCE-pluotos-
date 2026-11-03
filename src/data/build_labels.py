"""
src/data/build_labels.py
Constructs binary movement targets from StockNet preprocessed price files.
Applies research threshold rules:
- DOWN (0) if return <= -0.005
- UP (1) if return > 0.0055
- Excludes neutral returns in (-0.005, 0.0055]
"""
import os
import sys
from pathlib import Path
from typing import Tuple
import pandas as pd
import yaml

CONFIG_PATH = Path("configs/data.yaml")
STOCKNET_PRICE_DIR = Path("data/raw/stocknet-dataset/price/preprocessed")
STOCKNET_TWEET_DIR = Path("data/raw/stocknet-dataset/tweet/preprocessed")
OUTPUT_DIR = Path("data/processed")

def load_data_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "labeling": {"down_threshold": -0.005, "up_threshold": 0.0055},
        "splits": {
            "train_start": "2014-01-02", "train_end": "2015-07-31",
            "val_start": "2015-08-03", "val_end": "2015-09-30",
            "test_start": "2015-10-01", "test_end": "2015-12-31"
        }
    }

def construct_labels() -> pd.DataFrame:
    config = load_data_config()
    down_thresh = config["labeling"]["down_threshold"]
    up_thresh = config["labeling"]["up_threshold"]

    # Active 87 stocks that exist in tweet directories
    active_stocks = sorted([d.name for d in STOCKNET_TWEET_DIR.iterdir() if d.is_dir()])
    print(f"[Labels] Processing {len(active_stocks)} active stocks...")

    records = []
    total_raw_rows = 0
    excluded_neutral_rows = 0

    for ticker in active_stocks:
        pf = STOCKNET_PRICE_DIR / f"{ticker}.txt"
        if not pf.exists():
            continue

        # Column format in StockNet price/preprocessed:
        # col 0: Date, col 1: return (movement), col 2: open_ret, col 3: high_ret, col 4: low_ret, col 5: delta_price, col 6: volume
        df = pd.read_csv(pf, sep=r'\s+', header=None)
        df.columns = ["date", "movement", "open_return", "high_return", "low_return", "price_change", "volume"]
        df["date"] = df["date"].astype(str)

        # Filter strictly within 2014-01-02 to 2015-12-31
        mask_period = (df["date"] >= "2014-01-02") & (df["date"] <= "2015-12-31")
        df_period = df[mask_period].copy()
        total_raw_rows += len(df_period)

        for _, row in df_period.iterrows():
            ret = float(row["movement"])
            if ret <= down_thresh:
                label = 0
            elif ret > up_thresh:
                label = 1
            else:
                excluded_neutral_rows += 1
                continue

            records.append({
                "ticker": ticker,
                "date": row["date"],
                "movement": ret,
                "label": label,
                "open_return": float(row["open_return"]),
                "high_return": float(row["high_return"]),
                "low_return": float(row["low_return"]),
                "price_change": float(row["price_change"]),
                "volume": float(row["volume"])
            })

    labeled_df = pd.DataFrame(records)
    labeled_df.sort_values(by=["date", "ticker"], inplace=True)
    labeled_df.reset_index(drop=True, inplace=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "all_labeled_samples.csv"
    labeled_df.to_csv(output_path, index=False)

    print(f"[Labels] Construction complete.")
    print(f"  Raw 2014-2015 observations: {total_raw_rows:,}")
    print(f"  Excluded neutral observations: {excluded_neutral_rows:,}")
    print(f"  Final labeled observations: {len(labeled_df):,}")
    print(f"  DOWN (0): {(labeled_df['label'] == 0).sum():,} ({(labeled_df['label'] == 0).mean() * 100:.2f}%)")
    print(f"  UP   (1): {(labeled_df['label'] == 1).sum():,} ({(labeled_df['label'] == 1).mean() * 100:.2f}%)")
    print(f"  Saved to: {output_path}")

    return labeled_df

if __name__ == "__main__":
    construct_labels()
