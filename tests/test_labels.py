"""
tests/test_labels.py
Unit tests verifying target label construction and threshold handling.
"""
import pytest
import pandas as pd
from pathlib import Path

LABELS_PATH = Path("data/processed/all_labeled_samples.csv")

@pytest.fixture(scope="module")
def labels_df():
    if not LABELS_PATH.exists():
        from src.data.build_labels import construct_labels
        return construct_labels()
    return pd.read_csv(LABELS_PATH)

def test_labels_file_exists():
    assert LABELS_PATH.exists(), "Processed labeled samples file does not exist."

def test_labels_binary_values(labels_df):
    unique_labels = set(labels_df["label"].unique())
    assert unique_labels == {0, 1}, f"Labels must be strictly binary {0, 1}, found {unique_labels}"

def test_threshold_enforcement(labels_df):
    # DOWN = 0 must have movement <= -0.005
    down_samples = labels_df[labels_df["label"] == 0]
    assert (down_samples["movement"] <= -0.005).all(), "Found DOWN sample with return > -0.005"

    # UP = 1 must have movement > 0.0055
    up_samples = labels_df[labels_df["label"] == 1]
    assert (up_samples["movement"] > 0.0055).all(), "Found UP sample with return <= 0.0055"

def test_no_neutral_samples_leaked(labels_df):
    # Intermediate range (-0.005, 0.0055] must be strictly empty
    neutral_mask = (labels_df["movement"] > -0.005) & (labels_df["movement"] <= 0.0055)
    assert not neutral_mask.any(), f"Found {neutral_mask.sum()} samples in the excluded neutral zone!"

def test_total_count_matches_benchmark(labels_df):
    # Target from paper / previous benchmark is exactly 26,623
    assert len(labels_df) == 26623, f"Expected 26,623 labeled samples, got {len(labels_df)}"
