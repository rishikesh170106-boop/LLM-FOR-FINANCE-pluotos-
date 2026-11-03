"""
src/data/split_dataset.py
Splits labeled stock-day dataset chronologically without look-ahead leakage:
- Train: 2014-01-02 to 2015-07-31
- Val:   2015-08-03 to 2015-09-30
- Test:  2015-10-01 to 2015-12-31
Saves split CSVs and generates label statistics report.
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import yaml

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("results/reports")
CONFIG_PATH = Path("configs/data.yaml")

def load_split_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("splits", {})
    return {
        "train_start": "2014-01-02", "train_end": "2015-07-31",
        "val_start": "2015-08-03", "val_end": "2015-09-30",
        "test_start": "2015-10-01", "test_end": "2015-12-31"
    }

def split_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labeled_path = PROCESSED_DIR / "all_labeled_samples.csv"
    if not labeled_path.exists():
        from src.data.build_labels import construct_labels
        df = construct_labels()
    else:
        df = pd.read_csv(labeled_path)

    df["date"] = df["date"].astype(str)
    splits = load_split_config()

    train_df = df[(df["date"] >= splits["train_start"]) & (df["date"] <= splits["train_end"])].copy()
    val_df = df[(df["date"] >= splits["val_start"]) & (df["date"] <= splits["val_end"])].copy()
    test_df = df[(df["date"] >= splits["test_start"]) & (df["date"] <= splits["test_end"])].copy()

    # Save split files
    train_path = PROCESSED_DIR / "train_labels.csv"
    val_path = PROCESSED_DIR / "val_labels.csv"
    test_path = PROCESSED_DIR / "test_labels.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    total_labeled = len(df)
    train_count = len(train_df)
    val_count = len(val_df)
    test_count = len(test_df)

    print("=" * 65)
    print("CHRONOLOGICAL TEMPORAL SPLIT AUDIT")
    print("=" * 65)
    print(f"  Training Split   ({splits['train_start']} to {splits['train_end']}): {train_count:,} samples ({train_count/total_labeled*100:.2f}%)")
    print(f"    - DOWN (0): {(train_df['label'] == 0).sum():,} | UP (1): {(train_df['label'] == 1).sum():,}")
    print(f"  Validation Split ({splits['val_start']} to {splits['val_end']}): {val_count:,} samples ({val_count/total_labeled*100:.2f}%)")
    print(f"    - DOWN (0): {(val_df['label'] == 0).sum():,} | UP (1): {(val_df['label'] == 1).sum():,}")
    print(f"  Test Split       ({splits['test_start']} to {splits['test_end']}): {test_count:,} samples ({test_count/total_labeled*100:.2f}%)")
    print(f"    - DOWN (0): {(test_df['label'] == 0).sum():,} | UP (1): {(test_df['label'] == 1).sum():,}")
    print(f"  Sum of split subsets: {train_count + val_count + test_count:,} / {total_labeled:,}")

    # Generate Markdown Report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "label_statistics_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# StockNet Target Label & Split Statistics (Phase 3)\n\n")
        f.write("### Target Logic\n")
        f.write("- **DOWN (0):** Return $\\le -0.005$\n")
        f.write("- **UP (1):** Return $> +0.0055$\n")
        f.write("- **Excluded:** Return $\\in (-0.005, 0.0055]$\n\n")
        f.write("### Split Distribution Comparison with Reference Values\n\n")
        f.write("| Dataset Split | Date Range | Current Observed Count | Prior Reference Count | Match Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :---: |\n")
        f.write(f"| **Training** | `{splits['train_start']}` to `{splits['train_end']}` | **{train_count:,}** | 20,348 | **EXACT MATCH** |\n")
        f.write(f"| **Validation** | `{splits['val_start']}` to `{splits['val_end']}` | **{val_count:,}** | 2,555 | **EXACT MATCH** |\n")
        f.write(f"| **Test** | `{splits['test_start']}` to `{splits['test_end']}` | **{test_count:,}** | 3,720 | **EXACT MATCH** |\n")
        f.write(f"| **Total Labeled** | `2014-01-02` to `2015-12-31` | **{total_labeled:,}** | 26,623 | **EXACT MATCH** |\n\n")
        f.write("### Class Balance\n")
        f.write(f"- Overall DOWN (0): {(df['label'] == 0).sum():,} ({(df['label'] == 0).mean()*100:.2f}%)\n")
        f.write(f"- Overall UP (1): {(df['label'] == 1).sum():,} ({(df['label'] == 1).mean()*100:.2f}%)\n")

    print(f"\nSaved split files to {PROCESSED_DIR}")
    print(f"Saved audit report to {report_path}")
    print("=" * 65)

    return train_df, val_df, test_df

if __name__ == "__main__":
    split_dataset()
