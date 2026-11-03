"""
tests/test_dashboard.py
Unit tests for Phase 12: Interactive Web Dashboard (Streamlit Full-Stack Application).
Tests cached resource loaders, data formatting, and Plotly visual chart constructions.
"""
import os
import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Matplotlib config uses interim directory
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data" / "interim"))

from app import get_tracker, get_router, load_historical_test_samples, load_regime_metrics
import plotly.graph_objects as go


def test_tracker_resource_loader():
    """Verifies the tracker resource is correctly instantiated and points to DB."""
    tracker = get_tracker()
    assert tracker is not None
    assert tracker.db_path.exists()


def test_router_resource_loader():
    """Verifies the adaptive router model loads if checkpoint exists."""
    router = get_router()
    checkpoint = PROJECT_ROOT / "checkpoints" / "expert_router" / "best_router.pt"
    if checkpoint.exists():
        assert router is not None
        assert router.is_trained is True
    else:
        assert router is None


def test_historical_test_samples_loader():
    """Verifies that historical test samples can be loaded for audit browsing."""
    samples = load_historical_test_samples()
    if (PROJECT_ROOT / "data" / "processed" / "ploutos_test_prompts.jsonl").exists():
        assert isinstance(samples, list)
        assert len(samples) > 0
        s0 = samples[0]
        assert "ticker" in s0
        assert "date" in s0
        assert "label" in s0
        assert "input" in s0


def test_regime_metrics_loader():
    """Verifies regime metrics JSON payload structure."""
    reg = load_regime_metrics()
    if (PROJECT_ROOT / "results" / "metrics" / "regime_robustness_metrics.json").exists():
        assert isinstance(reg, dict)
        assert "regime_distribution" in reg
        assert "models" in reg
        assert "random_forest" in reg["models"]
        assert ("ploutos_adaptive_router" in reg["models"] or "ploutos_plus_router" in reg["models"])


def test_plotly_donut_chart_generation():
    """Verifies Plotly donut figure renders expert weights accurately."""
    w_tech, w_sent, w_human = 0.40, 0.35, 0.25
    fig = go.Figure(data=[go.Pie(
        labels=['Technical Expert', 'Sentiment Expert', 'Human/Fundamental Proxy'],
        values=[w_tech, w_sent, w_human],
        hole=.55
    )])
    assert len(fig.data) == 1
    assert fig.data[0].values == (0.40, 0.35, 0.25)
