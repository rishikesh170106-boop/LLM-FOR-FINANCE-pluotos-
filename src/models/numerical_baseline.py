"""
src/models/numerical_baseline.py
Baseline implementations for stock movement prediction:
1. MajorityBaseline: Trivial baseline predicting the most frequent class in training data.
2. RandomForestBaseline: Non-linear ensemble model trained on strictly lagged numerical features.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

class MajorityBaseline:
    """Majority Class Baseline."""
    def __init__(self):
        self.majority_class: int = 1

    def fit(self, X: pd.DataFrame, y: Any):
        if hasattr(y, "mode"):
            self.majority_class = int(y.mode().iloc[0])
        else:
            vals, counts = np.unique(y, return_counts=True)
            self.majority_class = int(vals[np.argmax(counts)])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.majority_class, dtype=int)

class RandomForestBaseline:
    """Random Forest classifier with strictly lagged numerical features."""
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, random_state: int = 42):
        self.imputer = SimpleImputer(strategy="median")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = list(X.columns)
        X_imputed = self.imputer.fit_transform(X)
        self.model.fit(X_imputed, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_imputed = self.imputer.transform(X)
        return self.model.predict(X_imputed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_imputed = self.imputer.transform(X)
        return self.model.predict_proba(X_imputed)

    def get_feature_importances(self) -> pd.DataFrame:
        imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_
        })
        return imp.sort_values("importance", ascending=False).reset_index(drop=True)
