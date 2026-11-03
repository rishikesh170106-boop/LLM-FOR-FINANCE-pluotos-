"""
src/evaluation/metrics.py
Standardized evaluation metrics for stock movement prediction:
- Accuracy
- Macro-F1
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix
- Per-class Precision, Recall, F1
"""
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

def evaluate_classification(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    split_name: str = "Test"
) -> Dict[str, Any]:
    """Computes comprehensive research metrics for binary classification."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    pred_down_pct = float((y_pred == 0).mean() * 100)
    pred_up_pct = float((y_pred == 1).mean() * 100)

    metrics = {
        "split": split_name,
        "sample_count": len(y_true),
        "accuracy": round(acc * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "mcc": round(mcc, 4),
        "confusion_matrix": cm,
        "down_class": {
            "precision": round(report["0"]["precision"] * 100, 2),
            "recall": round(report["0"]["recall"] * 100, 2),
            "f1": round(report["0"]["f1-score"] * 100, 2),
            "support": int(report["0"]["support"])
        },
        "up_class": {
            "precision": round(report["1"]["precision"] * 100, 2),
            "recall": round(report["1"]["recall"] * 100, 2),
            "f1": round(report["1"]["f1-score"] * 100, 2),
            "support": int(report["1"]["support"])
        },
        "prediction_distribution": {
            "pred_down_pct": round(pred_down_pct, 2),
            "pred_up_pct": round(pred_up_pct, 2)
        }
    }

    return metrics
