"""
tests/test_server.py
Unit tests for the Institutional Trading Terminal FastAPI backend (server.py).
Tests route handler functions directly without network/DNS overhead.
"""
import pytest
import json
from server import (
    get_terminal_ui,
    api_chart_data,
    api_ledger,
    api_historical_samples,
    api_regime_data,
    api_ablation_data
)


def test_terminal_root_endpoint():
    """Verifies that root URL returns the institutional HTML interface."""
    response = get_terminal_ui()
    assert response.status_code == 200
    html_text = response.body.decode("utf-8")
    assert "PLOUTOS" in html_text
    assert "LightweightCharts" in html_text or "TradingView" in html_text
    assert "Rishikesh" in html_text
    assert "Vineel" in html_text
    assert "Arjun" in html_text
    assert "mathTab" in html_text
    assert "btn-mathTab" in html_text
    assert "Quantitative Formulas &amp; Proofs" in html_text or "Quantitative Formulas & Proofs" in html_text
    assert "RELIANCE.NS" in html_text


def test_chart_data_endpoint():
    """Verifies that chart data returns valid OHLCV candles for TradingView and detects currency."""
    response = api_chart_data(ticker="NVDA")
    assert response.status_code == 200
    data = json.loads(response.body.decode("utf-8"))
    assert "ticker" in data
    assert data["ticker"] == "NVDA"
    assert data.get("currency_symbol") == "$"
    assert "candles" in data
    assert len(data["candles"]) >= 5
    c0 = data["candles"][0]
    for k in ["time", "open", "high", "low", "close"]:
        assert k in c0

    # Test Indian NSE currency detection
    response_in = api_chart_data(ticker="RELIANCE.NS")
    assert response_in.status_code == 200
    data_in = json.loads(response_in.body.decode("utf-8"))
    assert data_in.get("currency_symbol") == "₹"


def test_ledger_endpoint():
    """Verifies that forward tracking ledger returns summary KPIs."""
    response = api_ledger()
    assert response.status_code == 200
    data = json.loads(response.body.decode("utf-8"))
    assert "summary" in data
    assert "predictions" in data
    assert "total_predictions" in data["summary"]


def test_regime_data_endpoint():
    """Verifies that regime stress-testing endpoint returns models."""
    response = api_regime_data()
    assert response.status_code == 200
    data = json.loads(response.body.decode("utf-8"))
    assert "models" in data
    assert "random_forest" in data["models"]


def test_ablation_data_endpoint():
    """Verifies that ablation matrix endpoint returns ablations list."""
    response = api_ablation_data()
    assert response.status_code == 200
    data = json.loads(response.body.decode("utf-8"))
    assert "ablations" in data
    assert len(data["ablations"]) >= 8


def test_historical_samples_endpoint():
    """Verifies that historical benchmark samples return valid non-UNK records."""
    response = api_historical_samples(limit=10)
    assert response.status_code == 200
    data = json.loads(response.body.decode("utf-8"))
    assert "samples" in data
    assert len(data["samples"]) > 0
    s0 = data["samples"][0]
    assert s0["ticker"] != "UNK"
    assert s0["date"] != "UNK"
    assert len(s0["prompt"]) > 20
    assert s0["ground_truth"] in [0, 1]
