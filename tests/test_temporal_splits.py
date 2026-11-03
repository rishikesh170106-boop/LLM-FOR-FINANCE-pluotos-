"""
tests/test_temporal_splits.py
Unit tests verifying strict chronological temporal ordering and split boundaries.
"""
import pytest
import pandas as pd
from pathlib import Path

TRAIN_PATH = Path("data/processed/train_labels.csv")
VAL_PATH = Path("data/processed/val_labels.csv")
TEST_PATH = Path("data/processed/test_labels.csv")

@pytest.fixture(scope="module")
def splits():
    if not (TRAIN_PATH.exists() and VAL_PATH.exists() and TEST_PATH.exists()):
        from src.data.split_dataset import split_dataset
        return split_dataset()
    return pd.read_csv(TRAIN_PATH), pd.read_csv(VAL_PATH), pd.read_csv(TEST_PATH)

def test_split_files_exist():
    assert TRAIN_PATH.exists()
    assert VAL_PATH.exists()
    assert TEST_PATH.exists()

def test_split_sample_counts_match_benchmark(splits):
    train_df, val_df, test_df = splits
    assert len(train_df) == 20348, f"Expected 20,348 train samples, got {len(train_df)}"
    assert len(val_df) == 2555, f"Expected 2,555 val samples, got {len(val_df)}"
    assert len(test_df) == 3720, f"Expected 3,720 test samples, got {len(test_df)}"

def test_temporal_ordering_no_overlap(splits):
    train_df, val_df, test_df = splits

    max_train_date = train_df["date"].max()
    min_val_date = val_df["date"].min()
    max_val_date = val_df["date"].max()
    min_test_date = test_df["date"].min()

    # Chronological guarantee: max(train) < min(val) and max(val) < min(test)
    assert max_train_date < min_val_date, f"Train date {max_train_date} leaks into Val date {min_val_date}"
    assert max_val_date < min_test_date, f"Val date {max_val_date} leaks into Test date {min_test_date}"

def test_no_duplicate_stock_day_pairs(splits):
    train_df, val_df, test_df = splits
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        duplicates = df.duplicated(subset=["ticker", "date"]).sum()
        assert duplicates == 0, f"Found {duplicates} duplicate stock-day records in {name} split."
