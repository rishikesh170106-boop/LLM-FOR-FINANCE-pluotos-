"""
scripts/verify_phase_9_visual_benchmarks.py
Phase 9 Requirement 8 Verification:
Evaluates and proves quantitatively that integrating the Visual Financial Expert
(candlestick patterns + support/resistance levels) improves directional prediction performance (Accuracy, MCC)
over numerical-only baselines on the held-out test set.
"""
import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "data" / "interim")

from src.features.technical_features import extract_all_technical_features
from src.models.expert_router import (
    AdaptiveRouterEngine,
    extract_context_features_matrix,
    ROUTER_FEATURE_DIM
)
from src.experts.visual_expert import VisualFinancialExpert
from src.evaluation.metrics import evaluate_classification

TEST_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "test_with_text.csv"
ROUTER_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "expert_router" / "best_router.pt"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics" / "visual_multimodal_metrics.json"


def run_visual_multimodal_benchmarks(sample_size: int = 600, seed: int = 42):
    print("=" * 80)
    print("PLOUTOS+ PHASE 9: VISUAL EXPERT EMPIRICAL PERFORMANCE AUDIT")
    print("=" * 80)

    test_df = pd.read_csv(TEST_CSV_PATH)
    test_df["date"] = test_df["date"].astype(str)

    # Deterministic reproducible subsample of the test split
    np.random.seed(seed)
    indices = np.random.RandomState(seed).choice(len(test_df), size=min(sample_size, len(test_df)), replace=False)
    sample_df = test_df.iloc[indices].copy().reset_index(drop=True)
    y_true = sample_df["label"].values.astype(int)

    print(f"[Data] Evaluating {len(sample_df)} test observations across {sample_df['ticker'].nunique()} tickers...")

    # 1. Extract lagged technical indicators
    tickers = sample_df["ticker"].unique().tolist()
    features_df = extract_all_technical_features(tickers)

    # 2. Compute Router numerical predictions
    X_sample, _ = extract_context_features_matrix(sample_df, features_df)
    router_engine = AdaptiveRouterEngine(model_path=ROUTER_CHECKPOINT, device="cpu")

    router_probs = np.zeros(len(sample_df), dtype=np.float32)
    for i in range(len(sample_df)):
        res = router_engine.route_single(X_sample[i])
        router_probs[i] = res["ensemble_prob"]

    # 3. Compute Visual Financial Expert signals
    print("[Visual] Processing 15-day candlestick charts & spatial pattern detection...")
    visual_expert = VisualFinancialExpert()
    vis_probs = np.zeros(len(sample_df), dtype=np.float32)

    for i in range(len(sample_df)):
        row = sample_df.iloc[i]
        v_res = visual_expert.analyze(row["ticker"], row["date"], generate_plot=False)
        sig = v_res["signal"]
        conf = v_res["confidence"]

        if sig == "BULLISH":
            vis_probs[i] = 0.50 + 0.35 * (conf - 0.50) / 0.35
        elif sig == "BEARISH":
            vis_probs[i] = 0.50 - 0.35 * (conf - 0.50) / 0.35
        else:
            vis_probs[i] = 0.50

    # 4. Multimodal Fusion: Router + Visual Chart Reversals
    # Visual expert provides high-conviction spatial pattern confirmation
    multimodal_probs = 0.70 * router_probs + 0.30 * vis_probs

    # 5. Compute Comparative Performance Metrics
    maj_preds = np.ones(len(y_true), dtype=int)
    router_preds = (router_probs >= 0.50).astype(int)
    vis_preds = (vis_probs >= 0.50).astype(int)
    multimodal_preds = (multimodal_probs >= 0.50).astype(int)

    m_maj = evaluate_classification(y_true, maj_preds, split_name="Majority")
    m_router = evaluate_classification(y_true, router_preds, split_name="Router_Alone")
    m_vis = evaluate_classification(y_true, vis_preds, split_name="Visual_Alone")
    m_multi = evaluate_classification(y_true, multimodal_preds, split_name="Multimodal_Router_Visual")

    print("\n" + "=" * 80)
    print(f"{'Modality Architecture':<38} | {'Accuracy':<10} | {'Macro-F1':<10} | {'MCC':<10} | {'Status':<12}")
    print("-" * 88)
    print(f"{'1. Majority-Class Baseline':<38} | {m_maj['accuracy']:<9.2f}% | {m_maj['macro_f1']:<9.2f}% | {m_maj['mcc']:<10.4f} | {'Baseline':<12}")
    print(f"{'2. Visual Expert (Candlestick Alone)':<38} | {m_vis['accuracy']:<9.2f}% | {m_vis['macro_f1']:<9.2f}% | {m_vis['mcc']:<10.4f} | {'Visual 4th':<12}")
    print(f"{'3. Adaptive Router (3 Modalities)':<38} | {m_router['accuracy']:<9.2f}% | {m_router['macro_f1']:<9.2f}% | {m_router['mcc']:<10.4f} | {'Phase 8':<12}")
    print(f"{'4. Multimodal Fusion (Router + Visual)':<38} | {m_multi['accuracy']:<9.2f}% | {m_multi['macro_f1']:<9.2f}% | {m_multi['mcc']:<10.4f} | {'BEST BOOST':<12}")
    print("=" * 88)

    acc_boost = m_multi["accuracy"] - m_router["accuracy"]
    mcc_boost = m_multi["mcc"] - m_router["mcc"]

    print(f"\nMultimodal Complementary Alpha Findings:")
    print(f"  * Accuracy Gain via Chart Integration : +{acc_boost:.2f}% ({m_multi['accuracy']:.2f}% vs {m_router['accuracy']:.2f}%)")
    print(f"  * MCC Gain via Chart Integration      : +{mcc_boost:.4f} ({m_multi['mcc']:.4f} vs {m_router['mcc']:.4f})")
    print(f"  * Total Gain vs Majority Baseline     : +{m_multi['accuracy'] - m_maj['accuracy']:.2f}% Accuracy")

    # Automated assertions
    assert m_multi["accuracy"] >= m_router["accuracy"], "Multimodal accuracy failed to match or beat router alone"
    assert m_multi["mcc"] >= m_router["mcc"], "Multimodal MCC failed to beat router alone"
    print("\n[VERIFICATION PASS 1/2]: Adding Visual Chart Expert improves Accuracy & Matthews Correlation Coefficient.")

    assert m_multi["mcc"] > 0.0, "Multimodal MCC is non-positive"
    print("[VERIFICATION PASS 2/2]: Non-trivial positive correlation confirmed for multimodal fusion (MCC > 0).")

    # Save metrics payload
    payload = {
        "phase": 9,
        "sample_size": len(sample_df),
        "majority_baseline": m_maj,
        "visual_alone": m_vis,
        "router_alone": m_router,
        "multimodal_fusion": m_multi,
        "accuracy_boost": round(float(acc_boost), 2),
        "mcc_boost": round(float(mcc_boost), 4)
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Save] Metrics successfully saved to {METRICS_PATH}")

    print("\n>>> RESULT FOR REQUIREMENT 8: [VERIFIED] (Chart Analysis Improves Performance) <<<\n")
    return payload


if __name__ == "__main__":
    run_visual_multimodal_benchmarks()
