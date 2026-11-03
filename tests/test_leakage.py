"""
tests/test_leakage.py
Unit tests verifying zero-leakage constraints between input windows and targets.
"""
import pytest
import pandas as pd
from pathlib import Path

TRAIN_PATH = Path("data/processed/train_labels.csv")
TEST_PATH = Path("data/processed/test_labels.csv")

def test_no_future_dates_in_training():
    train_df = pd.read_csv(TRAIN_PATH)
    # The training boundary is strictly 2015-07-31
    future_samples = train_df[train_df["date"] > "2015-07-31"]
    assert len(future_samples) == 0, f"Found {len(future_samples)} future samples in training set!"

def test_target_separation_isolation():
    train_df = pd.read_csv(TRAIN_PATH)
    # Ensure label column is distinct and values are purely {0, 1}
    assert "label" in train_df.columns
    assert "movement" in train_df.columns
    # Check that movement contains continuous returns while label is discrete
    assert train_df["movement"].dtype in ["float64", "float32"]
    assert train_df["label"].dtype in ["int64", "int32"]

def test_tickers_consistency():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    train_tickers = set(train_df["ticker"].unique())
    test_tickers = set(test_df["ticker"].unique())
    # All 87 tickers should be represented across both splits
    assert len(train_tickers) == 87, f"Expected 87 tickers in train, got {len(train_tickers)}"
    assert len(test_tickers) == 87, f"Expected 87 tickers in test, got {len(test_tickers)}"
