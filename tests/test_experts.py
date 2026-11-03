"""
tests/test_experts.py
Unit tests verifying PloutosGen expert components and prompt formatting.
"""
import pytest
import pandas as pd
from src.experts.technical_expert import TechnicalExpert
from src.experts.sentiment_expert import SentimentExpert
from src.experts.human_expert import HumanExpert
from src.experts.prompt_builder import PloutosPromptBuilder

@pytest.fixture
def sample_row():
    return pd.Series({
        "ticker": "AAPL",
        "date": "2015-10-15",
        "label": 1,
        "movement": 0.012,
        "roll_momentum_5d": 0.025,
        "roll_return_std_5d": 0.012,
        "lagged_rsi_14": 58.0,
        "has_tweets": True,
        "text_evidence": "- $AAPL surge expected ahead of quarterly product cycle\n- Breaking out of consolidation"
    })

@pytest.fixture
def sample_empty_row():
    return pd.Series({
        "ticker": "MSFT",
        "date": "2014-06-10",
        "label": 0,
        "movement": -0.015,
        "roll_momentum_5d": -0.020,
        "roll_return_std_5d": 0.015,
        "lagged_rsi_14": 32.0,
        "has_tweets": False,
        "text_evidence": "[No textual evidence available for this trading day]"
    })

def test_technical_expert_output_schema(sample_row):
    expert = TechnicalExpert()
    out = expert.analyze(sample_row)
    assert out["expert_name"] == "Technical Expert"
    assert out["ticker"] == "AAPL"
    assert out["signal"] in {"BULLISH", "BEARISH", "NEUTRAL"}
    assert 0.0 <= out["confidence"] <= 1.0
    assert "metrics" in out
    assert out["availability"] is True

def test_sentiment_expert_with_tweets(sample_row):
    expert = SentimentExpert()
    out = expert.analyze(sample_row)
    assert out["expert_name"] == "Sentiment Expert"
    assert out["availability"] is True
    assert out["signal"] in {"BULLISH", "BEARISH", "NEUTRAL"}
    assert out["metrics"]["tweet_count"] == 2

def test_sentiment_expert_without_tweets(sample_empty_row):
    expert = SentimentExpert()
    out = expert.analyze(sample_empty_row)
    assert out["availability"] is False
    assert out["signal"] == "NEUTRAL"
    assert out["confidence"] == 0.50
    assert "No social-media discourse" in out["evidence"]

def test_human_expert_sector_mapping():
    expert = HumanExpert()
    aapl_row = pd.Series({"ticker": "AAPL", "date": "2015-01-05"})
    out = expert.analyze(aapl_row)
    assert out["expert_name"] == "Human / Fundamental Expert (Proxy)"
    assert out["metrics"]["sector"] == "Consumer Goods"
    assert "Apple" in out["metrics"]["company_name"]

def test_prompt_builder_structure(sample_row):
    builder = PloutosPromptBuilder()
    example = builder.build_instruction_example(sample_row)
    assert example["ticker"] == "AAPL"
    assert "instruction" in example
    assert "input" in example
    assert "output" in example
    assert "Prediction: UP" in example["output"]
    assert "Rationale:" in example["output"]
