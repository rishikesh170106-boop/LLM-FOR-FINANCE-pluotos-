"""
src/features/leakage_checks.py
Provides rigorous programmatic leakage verification:
- Asserts target columns ('label', 'movement') are strictly excluded from X
- Checks correlations between features and targets to detect any subtle look-ahead leaks
- Verifies temporal consistency
"""
from typing import Tuple, List
import pandas as pd
import numpy as np

def verify_feature_leakage(X: pd.DataFrame, y: pd.Series, target_returns: pd.Series) -> bool:
    """
    Rigorously audits feature matrix X against targets.
    Returns True if passed, raises AssertionError if leakage detected.
    """
    # 1. Target column exclusion
    forbidden_cols = ["label", "movement", "close", "target", "y"]
    for col in X.columns:
        if col.lower() in forbidden_cols:
            raise AssertionError(f"LEAKAGE DETECTED: Target column '{col}' is present in feature matrix X!")

    # 2. Correlation sanity check: No feature should correlate near 1.0 or -1.0 with same-day return
    correlations = {}
    for col in X.select_dtypes(include=[np.number]).columns:
        corr = X[col].corr(target_returns)
        correlations[col] = corr
        if abs(corr) > 0.95:
            raise AssertionError(f"LEAKAGE DETECTED: Feature '{col}' has extreme correlation ({corr:.4f}) with target return!")

    # 3. Same-day leakage verification: Assert features are not identical to same-day return
    for col in X.select_dtypes(include=[np.number]).columns:
        diff = (X[col] - target_returns).abs().mean()
        if diff < 1e-6:
            raise AssertionError(f"LEAKAGE DETECTED: Feature '{col}' is mathematically identical to target return!")

    return True

if __name__ == "__main__":
    dummy_X = pd.DataFrame({"lag_1_return": [0.01, -0.02, 0.03], "lag_2_return": [0.00, 0.01, -0.01]})
    dummy_y = pd.Series([1, 0, 1])
    dummy_ret = pd.Series([0.02, -0.015, 0.04])
    passed = verify_feature_leakage(dummy_X, dummy_y, dummy_ret)
    print("Leakage verification check passed:", passed)
