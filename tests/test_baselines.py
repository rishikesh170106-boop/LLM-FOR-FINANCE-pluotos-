"""
tests/test_baselines.py
Unit tests verifying baseline models, zero leakage, and metrics generation.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.features.leakage_checks import verify_feature_leakage
from src.models.numerical_baseline import MajorityBaseline, RandomForestBaseline
from src.evaluation.metrics import evaluate_classification

METRICS_PATH = Path("results/metrics/baseline_metrics.json")
PREDS_PATH = Path("results/predictions/baseline_predictions.csv")

def test_leakage_detector_catches_target():
    # If target is added to features, must raise AssertionError
    leak_X = pd.DataFrame({"movement": [0.01, -0.02, 0.03], "lag_1": [0.00, 0.01, -0.01]})
    y = pd.Series([1, 0, 1])
    ret = pd.Series([0.01, -0.02, 0.03])
    with pytest.raises(AssertionError):
        verify_feature_leakage(leak_X, y, ret)

def test_majority_baseline_predicts_constant():
    X_train = pd.DataFrame({"feat": [1, 2, 3, 4]})
    y_train = pd.Series([1, 1, 0, 1])  # Mode is 1
    model = MajorityBaseline().fit(X_train, y_train)
    preds = model.predict(X_train)
    assert (preds == 1).all()

def test_random_forest_predictions_valid():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 4), columns=[f"f_{i}" for i in range(4)])
    y = pd.Series(np.random.choice([0, 1], size=50))
    model = RandomForestBaseline(n_estimators=10, max_depth=3).fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 50
    assert set(np.unique(preds)).issubset({0, 1})

def test_evaluation_metrics_ranges():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = evaluate_classification(y_true, y_pred, "Test")
    assert 0 <= metrics["accuracy"] <= 100
    assert 0 <= metrics["macro_f1"] <= 100
    assert -1 <= metrics["mcc"] <= 1
