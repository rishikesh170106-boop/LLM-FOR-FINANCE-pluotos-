"""
src/training/train_baseline.py
Executes the Phase 5 Baseline Benchmarks:
1. Majority Class Baseline
2. Leakage-Free Historical Numerical Random Forest Baseline
Audits features for zero leakage, evaluates on Val and Test, and persists metrics and predictions.
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import numpy as np

from src.features.technical_features import extract_all_technical_features
from src.features.leakage_checks import verify_feature_leakage
from src.models.numerical_baseline import MajorityBaseline, RandomForestBaseline
from src.evaluation.metrics import evaluate_classification

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
REPORTS_DIR = RESULTS_DIR / "reports"

def run_baseline_pipeline():
    print("=" * 70)
    print("PLOUTOS+ PHASE 5: LEAKAGE-FREE BASELINE BENCHMARK")
    print("=" * 70)

    # 1. Load splits
    train_df = pd.read_csv(PROCESSED_DIR / "train_with_text.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val_with_text.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_with_text.csv")

    for df in [train_df, val_df, test_df]:
        df["date"] = df["date"].astype(str)

    active_tickers = sorted(train_df["ticker"].unique())
    print(f"[Data] Loaded splits: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    # 2. Extract technical features
    features_df = extract_all_technical_features(active_tickers)
    feature_cols = [c for c in features_df.columns if c not in ["date", "ticker"]]
    print(f"[Features] Extracted {len(feature_cols)} strictly lagged indicators.")

    # 3. Merge features with labels
    train_merged = train_df[["ticker", "date", "movement", "label"]].merge(features_df, on=["ticker", "date"], how="left")
    val_merged = val_df[["ticker", "date", "movement", "label"]].merge(features_df, on=["ticker", "date"], how="left")
    test_merged = test_df[["ticker", "date", "movement", "label"]].merge(features_df, on=["ticker", "date"], how="left")

    X_train = train_merged[feature_cols]
    y_train = train_merged["label"].astype(int)
    ret_train = train_merged["movement"]

    X_val = val_merged[feature_cols]
    y_val = val_merged["label"].astype(int)

    X_test = test_merged[feature_cols]
    y_test = test_merged["label"].astype(int)

    # 4. Strict Leakage Verification
    print("\n[Audit] Performing rigorous programmatic leakage audit...")
    verify_feature_leakage(X_train, y_train, ret_train)
    print("  PASS: Zero leakage detected. Target return and label strictly isolated.")

    # 5. Train & Evaluate Majority Baseline
    print("\n[1/2] Evaluating Majority Baseline...")
    majority_model = MajorityBaseline().fit(X_train, y_train)
    maj_val_preds = majority_model.predict(X_val)
    maj_test_preds = majority_model.predict(X_test)

    maj_val_metrics = evaluate_classification(y_val, maj_val_preds, split_name="Validation")
    maj_test_metrics = evaluate_classification(y_test, maj_test_preds, split_name="Test")

    # 6. Train & Evaluate Random Forest Baseline
    print("\n[2/2] Training Random Forest Baseline (Lagged Technical Features)...")
    rf_model = RandomForestBaseline(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)

    rf_val_preds = rf_model.predict(X_val)
    rf_test_preds = rf_model.predict(X_test)
    rf_test_probs = rf_model.predict_proba(X_test)[:, 1]

    rf_val_metrics = evaluate_classification(y_val, rf_val_preds, split_name="Validation")
    rf_test_metrics = evaluate_classification(y_test, rf_test_preds, split_name="Test")

    # Print Comparative Results
    print("\n" + "=" * 70)
    print(f"{'Model':<22} | {'Split':<10} | {'Accuracy':<10} | {'Macro-F1':<10} | {'MCC':<8}")
    print("-" * 70)
    print(f"{'Majority Baseline':<22} | {'Val':<10} | {maj_val_metrics['accuracy']:<10.2f}% | {maj_val_metrics['macro_f1']:<10.2f}% | {maj_val_metrics['mcc']:<8.4f}")
    print(f"{'Majority Baseline':<22} | {'Test':<10} | {maj_test_metrics['accuracy']:<10.2f}% | {maj_test_metrics['macro_f1']:<10.2f}% | {maj_test_metrics['mcc']:<8.4f}")
    print(f"{'Random Forest (Lagged)':<22} | {'Val':<10} | {rf_val_metrics['accuracy']:<10.2f}% | {rf_val_metrics['macro_f1']:<10.2f}% | {rf_val_metrics['mcc']:<8.4f}")
    print(f"{'Random Forest (Lagged)':<22} | {'Test':<10} | {rf_test_metrics['accuracy']:<10.2f}% | {rf_test_metrics['macro_f1']:<10.2f}% | {rf_test_metrics['mcc']:<8.4f}")
    print("=" * 70)

    # 7. Save Metrics and Predictions
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "majority_baseline": {"validation": maj_val_metrics, "test": maj_test_metrics},
        "random_forest_baseline": {"validation": rf_val_metrics, "test": rf_test_metrics}
    }

    metrics_file = METRICS_DIR / "baseline_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    predictions_df = test_merged[["ticker", "date", "movement", "label"]].copy()
    predictions_df["maj_pred"] = maj_test_preds
    predictions_df["rf_pred"] = rf_test_preds
    predictions_df["rf_prob_up"] = rf_test_probs
    predictions_file = PREDICTIONS_DIR / "baseline_predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)

    # Generate Markdown Report
    report_file = REPORTS_DIR / "baseline_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Phase 5: Baseline Benchmarking Report\n\n")
        f.write("### Benchmark Comparison Matrix\n\n")
        f.write("| Model | Split | Accuracy | Macro-F1 | MCC | Prediction Dist (DOWN / UP) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Majority Class** | Validation | {maj_val_metrics['accuracy']}% | {maj_val_metrics['macro_f1']}% | {maj_val_metrics['mcc']} | {maj_val_metrics['prediction_distribution']['pred_down_pct']}% / {maj_val_metrics['prediction_distribution']['pred_up_pct']}% |\n")
        f.write(f"| **Majority Class** | Test | {maj_test_metrics['accuracy']}% | {maj_test_metrics['macro_f1']}% | {maj_test_metrics['mcc']} | {maj_test_metrics['prediction_distribution']['pred_down_pct']}% / {maj_test_metrics['prediction_distribution']['pred_up_pct']}% |\n")
        f.write(f"| **Random Forest (Lagged)** | Validation | **{rf_val_metrics['accuracy']}%** | **{rf_val_metrics['macro_f1']}%** | **{rf_val_metrics['mcc']}** | {rf_val_metrics['prediction_distribution']['pred_down_pct']}% / {rf_val_metrics['prediction_distribution']['pred_up_pct']}% |\n")
        f.write(f"| **Random Forest (Lagged)** | Test | **{rf_test_metrics['accuracy']}%** | **{rf_test_metrics['macro_f1']}%** | **{rf_test_metrics['mcc']}** | {rf_test_metrics['prediction_distribution']['pred_down_pct']}% / {rf_test_metrics['prediction_distribution']['pred_up_pct']}% |\n\n")
        f.write("### Top Technical Features by Importance\n\n")
        top_features = rf_model.get_feature_importances().head(10)
        f.write("| Rank | Feature Indicator | Gini Importance |\n")
        f.write("| :---: | :--- | :---: |\n")
        for idx, row in top_features.iterrows():
            f.write(f"| {idx+1} | `{row['feature']}` | {row['importance']:.4f} |\n")

    print(f"\n[Artifacts Saved]")
    print(f"  - Metrics: {metrics_file}")
    print(f"  - Predictions: {predictions_file}")
    print(f"  - Markdown Report: {report_file}")
    print("=" * 70)
    print("PHASE 5: BASELINE BENCHMARK COMPLETE")
    print("=" * 70)

    return metrics_payload

if __name__ == "__main__":
    run_baseline_pipeline()
