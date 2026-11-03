"""
tests/test_investment_engine.py
Unit tests for the Actionable Investment Decision & Quantitative Capital Allocation Engine.
Verifies investment verdicts (STRONG_BUY, SCALE_IN, HOLD_CASH, AVOID_EXIT),
risk guardrails (stop-loss, take-profit), risk/reward ratios, and allocation sizing.
"""
import pytest
from src.live.investment_engine import InvestmentDecisionEngine


@pytest.fixture
def engine():
    return InvestmentDecisionEngine()


def test_strong_buy_evaluation(engine):
    """Verifies that high-conviction bullish consensus triggers a STRONG_BUY verdict."""
    res = engine.evaluate(
        current_price=200.0,
        predicted_direction="BULLISH",
        confidence=0.74,
        technical_metrics={"signal": "BULLISH", "volatility_5d": 0.015, "momentum_5d": 0.03, "rsi_14": 55.0},
        sentiment_metrics={"signal": "BULLISH", "polarity": 0.6},
        visual_metrics={"signal": "BULLISH", "confidence": 0.70, "patterns": ["Bullish Hammer"]},
        dominant_expert="Technical Expert"
    )

    assert res["verdict"] == "STRONG_BUY"
    assert res["can_invest"] is True
    assert res["allocation_pct"] == 100
    assert "🟢" in res["action_badge"]
    assert res["stop_loss_price"] < 200.0
    assert res["take_profit_price"] > 200.0
    assert res["risk_reward_ratio"] >= 1.30


def test_scale_in_evaluation(engine):
    """Verifies that moderate confidence triggers a cautious SCALE_IN verdict."""
    res = engine.evaluate(
        current_price=150.0,
        predicted_direction="BULLISH",
        confidence=0.62,
        technical_metrics={"signal": "BULLISH", "volatility_5d": 0.022, "momentum_5d": 0.01, "rsi_14": 52.0},
        sentiment_metrics={"signal": "NEUTRAL", "polarity": 0.0},
        visual_metrics={"signal": "NEUTRAL", "confidence": 0.50, "patterns": []},
        dominant_expert="Technical Expert"
    )

    assert res["verdict"] == "SCALE_IN"
    assert res["can_invest"] is True
    assert res["allocation_pct"] == 50
    assert "🟡" in res["action_badge"]
    assert res["stop_loss_price"] < 150.0
    assert res["take_profit_price"] > 150.0


def test_hold_cash_on_weak_confidence(engine):
    """Verifies that low-confidence signals trigger a capital-preserving HOLD_CASH verdict."""
    res = engine.evaluate(
        current_price=100.0,
        predicted_direction="BULLISH",
        confidence=0.52,
        technical_metrics={"signal": "NEUTRAL", "volatility_5d": 0.020, "momentum_5d": 0.002, "rsi_14": 50.0},
        sentiment_metrics={"signal": "NEUTRAL", "polarity": 0.0},
        visual_metrics={"signal": "NEUTRAL", "confidence": 0.50, "patterns": []},
        dominant_expert="Technical Expert"
    )

    assert res["verdict"] == "HOLD_CASH"
    assert res["can_invest"] is False
    assert res["allocation_pct"] == 0
    assert "⚪" in res["action_badge"]


def test_hold_cash_on_conflicting_signals(engine):
    """Verifies that sharp conflict between Technicals and Sentiment avoids capital risk."""
    res = engine.evaluate(
        current_price=180.0,
        predicted_direction="BULLISH",
        confidence=0.70,
        technical_metrics={"signal": "BEARISH", "volatility_5d": 0.035, "momentum_5d": -0.04, "rsi_14": 32.0},
        sentiment_metrics={"signal": "BULLISH", "polarity": 0.8},
        visual_metrics={"signal": "BEARISH", "confidence": 0.65, "patterns": []},
        dominant_expert="Sentiment Expert"
    )

    # In conflicting setup where technicals are actively bearish, engine refuses strong buy
    assert res["verdict"] in ["SCALE_IN", "HOLD_CASH"]
    assert res["allocation_pct"] <= 50


def test_avoid_exit_on_bearish_forecast(engine):
    """Verifies that high-confidence bearish movements produce an explicit AVOID/EXIT warning."""
    res = engine.evaluate(
        current_price=220.0,
        predicted_direction="BEARISH",
        confidence=0.68,
        technical_metrics={"signal": "BEARISH", "volatility_5d": 0.018, "momentum_5d": -0.03, "rsi_14": 42.0},
        sentiment_metrics={"signal": "BEARISH", "polarity": -0.5},
        visual_metrics={"signal": "BEARISH", "confidence": 0.65, "patterns": ["Shooting Star"]},
        dominant_expert="Technical Expert"
    )

    assert res["verdict"] == "AVOID_EXIT"
    assert res["can_invest"] is False
    assert res["allocation_pct"] == 0
    assert "🔴" in res["action_badge"]
    # Bearish risk levels: Invalidation is above entry, downside target is below entry
    assert res["stop_loss_price"] > 220.0
    assert res["take_profit_price"] < 220.0

