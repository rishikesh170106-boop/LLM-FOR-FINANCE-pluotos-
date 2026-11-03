"""
tests/test_text_processing.py
Unit tests verifying text processing, daily tweet cap, and missing text handling.
"""
import pytest
import pandas as pd
from pathlib import Path

TRAIN_TEXT_PATH = Path("data/processed/train_with_text.csv")
VAL_TEXT_PATH = Path("data/processed/val_with_text.csv")
TEST_TEXT_PATH = Path("data/processed/test_with_text.csv")
ALL_TEXT_PATH = Path("data/processed/all_labeled_with_text.csv")

@pytest.fixture(scope="module")
def all_text_df():
    if not ALL_TEXT_PATH.exists():
        from src.data.process_tweets import process_and_align_tweets
        process_and_align_tweets()
    return pd.read_csv(ALL_TEXT_PATH)

def test_text_files_exist():
    assert TRAIN_TEXT_PATH.exists()
    assert VAL_TEXT_PATH.exists()
    assert TEST_TEXT_PATH.exists()
    assert ALL_TEXT_PATH.exists()

def test_daily_cap_enforced(all_text_df):
    # Tweet count per stock-day must not exceed 10
    assert (all_text_df["tweet_count"] <= 10).all(), "Found stock-day with more than 10 tweets!"
    assert (all_text_df["tweet_count"] >= 0).all()

def test_tweet_coverage_matches_benchmark(all_text_df):
    total = len(all_text_df)
    with_tweets = all_text_df["has_tweets"].sum()
    coverage_pct = (with_tweets / total) * 100
    # Expected: 12,018 / 26,623 = 45.14%
    assert with_tweets == 12018, f"Expected exactly 12,018 stock-days with tweets, got {with_tweets}"
    assert abs(coverage_pct - 45.14) < 0.1

def test_missing_text_properly_handled(all_text_df):
    no_tweet_rows = all_text_df[~all_text_df["has_tweets"]]
    placeholder = "[No textual evidence available for this trading day]"
    assert (no_tweet_rows["text_evidence"] == placeholder).all()
    # Ensure no NaN / null text values
    assert not all_text_df["text_evidence"].isna().any()

def test_sample_counts_preserved():
    train_df = pd.read_csv(TRAIN_TEXT_PATH)
    val_df = pd.read_csv(VAL_TEXT_PATH)
    test_df = pd.read_csv(TEST_TEXT_PATH)

    assert len(train_df) == 20348
    assert len(val_df) == 2555
    assert len(test_df) == 3720
