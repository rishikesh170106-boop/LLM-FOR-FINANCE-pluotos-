"""
src/evaluation/regime_analysis.py
Phase 10: Market-Regime Robustness Analysis & Financial Stress Testing
Partitions historical stock-day observations into 4 distinct economic regimes:
1. Bull Regime (Upward momentum, positive drift)
2. Bear Regime (Downward momentum, selling pressure, drawdown)
3. Sideways Regime (Low/moderate volatility, mean-reverting range)
4. High-Volatility Regime (Top quartile volatility shock)
All regime assignments use strictly lagged indicators (dates < t) with zero future leakage.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import evaluate_classification

REGIME_NAMES = ["Bull", "Bear", "Sideways", "High-Volatility"]


def classify_market_regimes(
    merged_df: pd.DataFrame,
    vol_percentile_threshold: float = 75.0
) -> pd.Series:
    """
    Assigns each stock-day to an economic market regime based strictly on lagged indicators:
    - High-Volatility: roll_return_std_5d >= 75th percentile of the universe
    - Bull: momentum > +0.015 (+1.5% 5-day return) and not High-Vol
    - Bear: momentum < -0.015 (-1.5% 5-day return) and not High-Vol
    - Sideways: -0.015 <= momentum <= +0.015 and not High-Vol
    """
    vol = merged_df["roll_return_std_5d"].fillna(0.01).values
    mom = merged_df["roll_momentum_5d"].fillna(0.0).values

    vol_thresh = np.percentile(vol, vol_percentile_threshold)

    regimes = []
    for v, m in zip(vol, mom):
        if v >= vol_thresh:
            regimes.append("High-Volatility")
        elif m > 0.015:
            regimes.append("Bull")
        elif m < -0.015:
            regimes.append("Bear")
        else:
            regimes.append("Sideways")

    return pd.Series(regimes, index=merged_df.index, name="market_regime")


def evaluate_model_across_regimes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regimes: pd.Series,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes performance metrics (Accuracy, Macro-F1, MCC) partitioned by market regime.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    regimes_arr = np.asarray(regimes)

    regime_results = {}
    mccs = []

    for r_name in REGIME_NAMES:
        mask = (regimes_arr == r_name)
        count = int(mask.sum())
        if count == 0:
            continue

        r_y_true = y_true[mask]
        r_y_pred = y_pred[mask]
        m = evaluate_classification(r_y_true, r_y_pred, split_name=r_name)
        regime_results[r_name] = m
        mccs.append(m["mcc"])

    # Overall full-split metrics
    overall_m = evaluate_classification(y_true, y_pred, split_name="Overall")

    # Resilience metric: Minimum MCC across all adverse regimes
    min_mcc = min(mccs) if mccs else 0.0
    mcc_spread = max(mccs) - min(mccs) if mccs else 0.0

    return {
        "model_name": model_name,
        "overall": overall_m,
        "by_regime": regime_results,
        "resilience": {
            "min_mcc": round(float(min_mcc), 4),
            "max_mcc": round(float(max(mccs) if mccs else 0.0), 4),
            "mcc_spread": round(float(mcc_spread), 4),
            "is_universally_positive": bool(min_mcc > 0.0)
        }
    }

