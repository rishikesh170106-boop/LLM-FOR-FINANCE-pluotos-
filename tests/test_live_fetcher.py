"""
tests/test_live_fetcher.py
Unit and integration tests for Phase 11: Real-Time Live Ingestion & Prediction Engine.
Tests LiveMarketFetcher, PredictionTracker, and Live Predictor pipeline.
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Matplotlib config uses interim directory
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data" / "interim"))

from src.live.live_fetcher import LiveMarketFetcher
from src.live.tracker import PredictionTracker
from src.live.live_predict import run_live_prediction, generate_synthetic_live_data
from src.models.expert_router import ROUTER_FEATURE_DIM


@pytest.fixture
def sample_ohlcv_df():
    """Generates a realistic 30-day synthetic OHLCV dataframe."""
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=30, freq="B").strftime("%Y-%m-%d")
    prices = [150.0]
    for _ in range(29):
        ret = np.random.normal(0.001, 0.015)
        prices.append(prices[-1] * (1.0 + ret))

    rows = []
    for d, p in zip(dates, prices):
        rows.append({
            "date": d,
            "open": p * 0.995,
            "high": p * 1.01,
            "low": p * 0.99,
            "close": p,
            "volume": int(np.random.uniform(1e6, 3e6))
        })
    return pd.DataFrame(rows)


def test_synthetic_live_data_generator():
    """Verifies synthetic data generator returns valid OHLCV columns and length."""
    df = generate_synthetic_live_data("NVDA")
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 20
    for col in ["date", "open", "high", "low", "close", "volume"]:
        assert col in df.columns
    assert (df["high"] >= df["low"]).all()


def test_live_technical_indicators_computation(sample_ohlcv_df):
    """Verifies technical indicator extraction from live OHLCV bars."""
    fetcher = LiveMarketFetcher()
    tech = fetcher.compute_live_technical_indicators(sample_ohlcv_df)

    assert "last_close" in tech
    assert "momentum_5d" in tech
    assert "volatility_5d" in tech
    assert "rsi_14" in tech
    assert "signal" in tech
    assert "confidence" in tech
    assert tech["signal"] in ["BULLISH", "BEARISH", "NEUTRAL"]
    assert 0.0 <= tech["rsi_14"] <= 100.0
    assert tech["confidence"] >= 0.50


def test_context_vector_dimensions_and_finite(sample_ohlcv_df):
    """Verifies the 16-D context vector satisfies router model constraints."""
    fetcher = LiveMarketFetcher()
    tech = fetcher.compute_live_technical_indicators(sample_ohlcv_df)
    sent = {
        "has_tweets": 1,
        "tweet_count": 8,
        "bull_cues": 6,
        "bear_cues": 2,
        "polarity": 0.5,
        "confidence": 0.75,
        "signal": "BULLISH"
    }
    profile = {
        "sector": "Technology",
        "beta": 1.2
    }

    vec = fetcher.construct_live_context_vector(tech, sent, profile)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (ROUTER_FEATURE_DIM,)
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()
    assert vec.dtype == np.float32


def test_prediction_tracker_lifecycle():
    """Tests the full SQLite forward tracking lifecycle: log -> pending -> settle -> summary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_db = Path(tmpdir) / "test_tracker.db"
        tracker = PredictionTracker(db_path=tmp_db)

        # 1. Log a live prediction
        weights = {"technical": 0.50, "sentiment": 0.30, "human_proxy": 0.20}
        pred_id = tracker.log_prediction(
            ticker="TSLA",
            last_close=240.0,
            predicted_direction="BULLISH",
            predicted_confidence=0.78,
            weights=weights,
            dominant_expert="Technical Expert",
            chart_pattern="Hammer",
            chart_signal="BULLISH",
            rationale="Technical momentum and hammer pattern indicate upward rebound.",
            prediction_date="2026-11-01"
        )
        assert pred_id == 1

        # 2. Check pending
        pending = tracker.get_pending_predictions()
        assert len(pending) == 1
        assert pending[0]["ticker"] == "TSLA"
        assert pending[0]["status"] == "PENDING"
        assert pending[0]["predicted_direction"] == "BULLISH"

        # 3. Settle prediction with realized price (Realized: $245 -> +2.08% return -> BULLISH)
        outcome = tracker.update_realized_outcome(pred_id, realized_next_close=245.0)
        assert outcome["is_correct"] is True
        assert outcome["actual_direction"] == "BULLISH"
        assert outcome["realized_return"] > 0

        # Check pending is now empty
        assert len(tracker.get_pending_predictions()) == 0

        # 4. Compute audit summary
        summary = tracker.compute_audit_summary()
        assert summary["total_predictions"] == 1
        assert summary["total_settled"] == 1
        assert summary["settled_correct"] == 1
        assert summary["forward_accuracy"] == 1.0


def test_live_prediction_end_to_end():
    """Tests end-to-end live prediction execution in fast analytical mode."""
    res = run_live_prediction(ticker="MSFT", use_llm=False, save_chart=True, track=True)

    assert "ticker" in res
    assert res["ticker"] == "MSFT"
    assert res["predicted_direction"] in ["BULLISH", "BEARISH"]
    assert 0.50 <= res["confidence"] <= 1.0
    assert "dominant_expert" in res
    assert "weights" in res
    assert "visual" in res
    assert res["visual"]["chart_path"] is not None
    assert Path(res["visual"]["chart_path"]).exists()
    assert len(res["rationale"]) > 20

