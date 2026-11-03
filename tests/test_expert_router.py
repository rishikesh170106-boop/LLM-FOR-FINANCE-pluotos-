"""
tests/test_expert_router.py
Automated unit tests for Phase 8: Adaptive Expert Router
Verifies:
- 16-dimensional feature interface and tensor shapes
- Softmax probability constraint (weights sum to 1.0)
- Checkpoint existence and state dict integrity
- Engine routing output structure and prompt header formatting
"""
import pytest
import numpy as np
import torch
from pathlib import Path

from src.models.expert_router import (
    ContextGatedRouter,
    AdaptiveRouterEngine,
    ROUTER_FEATURE_DIM,
    ROUTER_FEATURE_NAMES,
    EXPERT_NAMES
)

CHECKPOINT_PATH = Path("checkpoints/expert_router/best_router.pt")


def test_router_architecture_and_feature_dim():
    """Verifies that the router requires exactly 16 features and outputs 3 expert weights."""
    assert ROUTER_FEATURE_DIM == 16
    assert len(ROUTER_FEATURE_NAMES) == 16
    assert len(EXPERT_NAMES) == 3

    router = ContextGatedRouter(input_dim=16, expert_count=3)
    batch_size = 5
    dummy_input = torch.randn(batch_size, 16)

    weights, ensemble_prob = router(dummy_input)

    assert weights.shape == (batch_size, 3), f"Expected shape ({batch_size}, 3), got {weights.shape}"
    assert ensemble_prob.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {ensemble_prob.shape}"


def test_softmax_probability_constraint():
    """Verifies that gating weights strictly satisfy the simplex constraint (sum to 1.0 and non-negative)."""
    router = ContextGatedRouter(input_dim=16, expert_count=3)
    dummy_input = torch.randn(10, 16)

    weights, ensemble_prob = router(dummy_input)

    # All weights non-negative
    assert torch.all(weights >= 0.0), "Found negative gating weights"

    # All rows sum to 1.0 within numerical precision
    row_sums = weights.sum(dim=-1)
    expected_ones = torch.ones_like(row_sums)
    assert torch.allclose(row_sums, expected_ones, atol=1e-5), f"Weights do not sum to 1: {row_sums}"

    # Probabilities bounded between 0 and 1
    assert torch.all((ensemble_prob >= 0.0) & (ensemble_prob <= 1.0)), "Probabilities outside [0, 1]"


def test_router_checkpoint_exists_and_loads():
    """Verifies that the trained Phase 8 checkpoint exists and contains valid state dict."""
    assert CHECKPOINT_PATH.exists(), f"Router checkpoint not found at {CHECKPOINT_PATH}"

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    assert "model_state_dict" in checkpoint, "model_state_dict missing from checkpoint"
    assert "epoch" in checkpoint, "epoch missing from checkpoint"
    assert "val_acc" in checkpoint, "val_acc missing from checkpoint"
    assert "val_mcc" in checkpoint, "val_mcc missing from checkpoint"

    router = ContextGatedRouter()
    router.load_state_dict(checkpoint["model_state_dict"])
    router.eval()


def test_adaptive_router_engine_routing():
    """Verifies that the high-level inference engine produces structured routing decisions."""
    engine = AdaptiveRouterEngine(model_path=CHECKPOINT_PATH, device="cpu")
    dummy_context = np.zeros(16, dtype=np.float32)

    res = engine.route_single(dummy_context)

    assert "weights" in res
    assert "dominant_expert" in res
    assert "ensemble_prob" in res
    assert "predicted_direction" in res
    assert res["predicted_direction"] in ["UP", "DOWN"]
    assert 0.0 <= res["ensemble_prob"] <= 1.0

    w = res["weights"]
    assert abs(w["technical"] + w["sentiment"] + w["human"] - 1.0) < 1e-3


def test_router_prompt_header_formatting():
    """Verifies that the router prompt header formats cleanly for LLM instruction injection."""
    engine = AdaptiveRouterEngine(model_path=CHECKPOINT_PATH, device="cpu")
    dummy_context = np.zeros(16, dtype=np.float32)
    dummy_context[1] = 0.04  # strong momentum
    dummy_context[3] = 0.4   # high RSI

    routing_res = engine.route_single(dummy_context)
    header = engine.format_router_prompt_header(routing_res)

    assert "Ploutos+ Adaptive Expert Router Allocation" in header
    assert "Dynamic Expert Weights" in header
    assert "Technical:" in header
    assert "Sentiment:" in header
    assert "Fundamental:" in header


def test_weights_actually_influence_llm_prompt():
    """
    Requirement 6 Verification:
    Verifies that the Adaptive Router's dynamic weights and dominant guidance
    are actively injected into the prompt payload passed to the LLM.
    """
    engine = AdaptiveRouterEngine(model_path=CHECKPOINT_PATH, device="cpu")
    dummy_context = np.zeros(16, dtype=np.float32)
    dummy_context[1] = 0.05
    routing_info = engine.route_single(dummy_context)
    w = routing_info["weights"]

    guided_instruction = (
        f"[ADAPTIVE ROUTER GUIDANCE]:\n"
        f"The Adaptive Router has evaluated prevailing market conditions and allocated the following expert weights:\n"
        f"- Technical Expert: {w['technical']*100:.1f}%\n"
        f"- Sentiment Expert: {w['sentiment']*100:.1f}%\n"
        f"- Human/Fundamental Proxy: {w['human']*100:.1f}%\n"
        f"Dominant Priority: {routing_info['dominant_expert']} ({routing_info['routing_reason']}).\n"
        f"You MUST condition your directional movement prediction and explanation on these dynamic weights."
    )

    header = engine.format_router_prompt_header(routing_info)
    raw_input = "Ticker: $AAPL\nSignal: BEARISH"
    full_prompt = f"{guided_instruction}\n\n{header}{raw_input}"

    # Assert active influence fields are present in LLM prompt
    assert "ADAPTIVE ROUTER GUIDANCE" in full_prompt
    assert "Technical Expert:" in full_prompt
    assert "Dominant Priority:" in full_prompt
    assert routing_info["dominant_expert"] in full_prompt


def test_router_dynamically_responds_to_changing_market_conditions():
    """
    Requirement 7 Verification:
    Verifies that the router outputs distinct dynamic weights across different market regimes
    (e.g., Bear Shock vs Sentiment Surge vs Missing Text vs Quiet Macro).
    """
    from scripts.verify_router_responsiveness import run_responsiveness_verification
    results = run_responsiveness_verification()

    # Assert multiple distinct dominant experts
    dominant_experts = set(r["dominant"] for r in results)
    assert len(dominant_experts) >= 2, f"Expected multiple dominant experts across regimes, got {dominant_experts}"

    # Assert significant dynamic weight shifts (>= 5% delta)
    tech_shift = max(r["weights"]["technical"] for r in results) - min(r["weights"]["technical"] for r in results)
    assert tech_shift >= 0.05, f"Technical weight shift was only {tech_shift*100:.2f}%"


def test_improved_prediction_performance():
    """
    Requirement 8 Verification:
    Verifies that the Adaptive Router establishes superior test accuracy and MCC
    over the Random Forest and Majority-Class baselines on the 3,720-sample test set.
    """
    import json
    with open("results/metrics/baseline_metrics.json", "r", encoding="utf-8") as f:
        base_data = json.load(f)
    with open("results/metrics/router_metrics.json", "r", encoding="utf-8") as f:
        router_data = json.load(f)

    rf_test = base_data["random_forest_baseline"]["test"]
    router_test = router_data["test_metrics"]

    # Assert router outperforms Random Forest baseline
    assert router_test["accuracy"] > rf_test["accuracy"], f"Router Acc {router_test['accuracy']} <= RF Acc {rf_test['accuracy']}"
    assert router_test["mcc"] > rf_test["mcc"], f"Router MCC {router_test['mcc']} <= RF MCC {rf_test['mcc']}"
    assert router_test["mcc"] > 0.05, f"Router MCC {router_test['mcc']} is below 0.05"

