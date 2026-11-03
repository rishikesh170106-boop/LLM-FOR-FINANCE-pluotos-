"""
tests/test_ablation.py
Unit tests for Phase 13: Comprehensive Ablation Studies & Benchmark Metric Matrix.
Verifies ablation metrics integrity, statistical assertions, and publication artifact generation.
"""
import json
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = PROJECT_ROOT / "results" / "metrics" / "ablation_study_metrics.json"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
REPORT_PATH = PROJECT_ROOT / "results" / "reports" / "phase_13_ablation_report.md"


def test_ablation_metrics_file_exists_and_valid():
    """Verifies that ablation study metrics payload exists and is non-empty."""
    assert METRICS_PATH.exists(), f"Ablation metrics file not found at {METRICS_PATH}"
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["phase"] == 13
    assert data["sample_universe"] == 3720
    assert "ablations" in data
    assert len(data["ablations"]) >= 8


def test_ablation_configurations_structure():
    """Verifies that each ablation configuration has all required quantitative metrics."""
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for a in data["ablations"]:
        assert "id" in a
        assert "name" in a
        assert "accuracy" in a
        assert "mcc" in a
        assert "macro_f1" in a
        assert 0.0 <= a["accuracy"] <= 100.0
        assert -1.0 <= a["mcc"] <= 1.0


def test_scientific_superiority_assertions():
    """
    Verifies the key research hypotheses:
    1. Adaptive Router beats Static Fixed Ploutos.
    2. Adaptive Router strictly beats Random Forest in MCC and Accuracy.
    3. Full Multimodal System achieves the top benchmark performance.
    """
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    ab_dict = {a["id"]: a for a in data["ablations"]}

    rf = ab_dict["random_forest"]
    static = ab_dict["static_ploutos"]
    router = ab_dict["ploutos_adaptive_router"]
    full_multi = ab_dict["full_multimodal_ploutos"]

    # 1. Gating superiority: Adaptive Router > Static Ploutos
    assert router["accuracy"] > static["accuracy"]
    assert router["mcc"] > static["mcc"]

    # 2. Performance gain over Random Forest baseline
    assert router["accuracy"] > rf["accuracy"]
    assert router["mcc"] > rf["mcc"]

    # 3. Multimodal top rank
    assert full_multi["accuracy"] >= router["accuracy"]
    assert full_multi["mcc"] >= router["mcc"]


def test_publication_figures_and_report_exist():
    """Verifies that all visual figures and formal research report are properly saved."""
    assert (FIGURES_DIR / "ablation_accuracy_comparison.png").exists()
    assert (FIGURES_DIR / "ablation_mcc_comparison.png").exists()
    assert (FIGURES_DIR / "ablation_component_gain.png").exists()
    assert REPORT_PATH.exists()
    assert REPORT_PATH.stat().st_size > 500

