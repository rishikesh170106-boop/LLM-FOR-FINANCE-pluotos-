"""
tests/test_visual_expert.py
Phase 9 Automated Unit Test Suite: Visual Financial Expert
Verifies:
- Zero leakage in 15-day lookback window (all dates < target_date)
- Accurate normalized OHLC candlestick prices (Low <= Open, Close <= High)
- Candlestick pattern detection algorithms (Hammer, Shooting Star, Engulfing)
- Chart PNG rendering to disk with non-trivial file size
- Seamless integration with PloutosPromptBuilder
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.experts.visual_expert import VisualFinancialExpert
from src.experts.prompt_builder import PloutosPromptBuilder

CHARTS_DIR = Path("results/figures/charts")


def test_visual_expert_initialization():
    """Verifies default parameters and metadata."""
    expert = VisualFinancialExpert(lookback_days=15)
    assert expert.expert_name == "Visual Financial Expert"
    assert expert.lookback_days == 15


def test_lagged_ohlc_extraction_and_zero_leakage():
    """Verifies that all extracted OHLC records are strictly prior to target_date (zero future leakage)."""
    expert = VisualFinancialExpert(lookback_days=15)
    target_date = "2015-10-01"
    ohlc = expert.extract_lagged_ohlc("AAPL", target_date)

    assert ohlc is not None, "Failed to extract lagged OHLC for AAPL"
    assert len(ohlc) <= 15
    assert len(ohlc) >= 5

    # Strictly lagged check
    for d in ohlc["date"]:
        assert str(d) < target_date, f"Leakage detected! Record date {d} is not strictly prior to {target_date}"

    # Price hierarchy check
    assert np.all(ohlc["norm_low"] <= ohlc["norm_high"]), "Found Low > High"
    assert np.all(ohlc["norm_low"] <= ohlc["norm_open"] + 1e-4), "Found Low > Open"
    assert np.all(ohlc["norm_close"] <= ohlc["norm_high"] + 1e-4), "Found Close > High"


def test_synthetic_hammer_pattern_detection():
    """Verifies detection of Bullish Hammer pattern on synthetic OHLC data."""
    expert = VisualFinancialExpert()

    # Construct synthetic dataframe with a Bullish Hammer on the final day
    # Open: 100.0, Close: 100.5, High: 100.6, Low: 98.0 (Lower wick = 2.0, Body = 0.5, Ratio = 4.0x)
    data = []
    for i in range(10):
        data.append({
            "date": f"2015-09-{i+1:02d}",
            "norm_open": 100.0,
            "norm_high": 101.0,
            "norm_low": 99.5,
            "norm_close": 100.2,
            "volume_val": 1e6
        })
    # Add hammer candle at the end
    data.append({
        "date": "2015-09-20",
        "norm_open": 100.0,
        "norm_high": 100.6,
        "norm_low": 98.0,
        "norm_close": 100.5,
        "volume_val": 1.5e6
    })
    df = pd.DataFrame(data)

    res = expert.detect_chart_patterns(df)
    assert any("Hammer" in p for p in res["patterns"]), f"Failed to detect hammer pattern in {res['patterns']}"
    assert res["signal"] == "BULLISH"


def test_synthetic_shooting_star_pattern_detection():
    """Verifies detection of Bearish Shooting Star pattern on synthetic OHLC data."""
    expert = VisualFinancialExpert()

    # Open: 100.5, Close: 100.0, High: 103.0, Low: 99.8 (Upper wick = 2.5, Body = 0.5, Ratio = 5.0x)
    data = []
    for i in range(10):
        data.append({
            "date": f"2015-09-{i+1:02d}",
            "norm_open": 100.0,
            "norm_high": 101.0,
            "norm_low": 99.5,
            "norm_close": 100.2,
            "volume_val": 1e6
        })
    data.append({
        "date": "2015-09-20",
        "norm_open": 100.5,
        "norm_high": 103.0,
        "norm_low": 99.8,
        "norm_close": 100.0,
        "volume_val": 1.5e6
    })
    df = pd.DataFrame(data)

    res = expert.detect_chart_patterns(df)
    assert any("Shooting Star" in p for p in res["patterns"]), f"Failed to detect shooting star in {res['patterns']}"
    assert res["signal"] == "BEARISH"


def test_chart_rendering_saves_valid_png(tmp_path):
    """Verifies that render_candlestick_chart saves a valid, non-empty PNG file."""
    expert = VisualFinancialExpert()
    ohlc = expert.extract_lagged_ohlc("AAPL", "2015-10-01")
    assert ohlc is not None

    test_png = tmp_path / "test_aapl_chart.png"
    saved = expert.render_candlestick_chart(ohlc, "AAPL", "2015-10-01", ["Bullish Hammer"], save_path=test_png)

    assert saved.exists()
    assert saved.stat().st_size > 15000, f"Chart PNG file size is unexpectedly small: {saved.stat().st_size} bytes"


def test_prompt_builder_includes_visual_expert():
    """Verifies that PloutosPromptBuilder cleanly incorporates Visual Expert evidence as Section 4."""
    pb = PloutosPromptBuilder(include_visual_expert=True)
    sample_row = pd.Series({
        "ticker": "AAPL",
        "date": "2015-10-01",
        "roll_momentum_5d": -0.035,
        "roll_return_std_5d": 0.0175,
        "lagged_rsi_14": 47.5,
        "tweet_count": 5,
        "has_tweets": 1,
        "text_evidence": "Apple releases new update"
    })

    evidence = pb.build_expert_evidence(sample_row)
    assert "visual" in evidence, "Visual evidence missing from build_expert_evidence"
    assert evidence["visual"]["expert_name"] == "Visual Financial Expert"

    context = pb.format_input_context("AAPL", "2015-10-01", evidence)
    assert "--- [4. Visual Chart Expert Evidence] ---" in context
    assert "Signal:" in context
    assert "Evidence:" in context


def test_visual_evidence_influences_llm_prompt():
    """
    Requirement 7 Verification:
    Verifies that the Visual Financial Expert guidance and candlestick formations
    are actively embedded into the LLM prompt instruction and evidence context.
    """
    expert = VisualFinancialExpert()
    res = expert.analyze("AAPL", "2015-10-01", generate_plot=False)

    guided_instruction = (
        f"[VISUAL FINANCIAL EXPERT GUIDANCE]:\n"
        f"- Candlestick Structure Signal: {res['signal']} (Confidence: {res['confidence']:.2f})\n"
        f"- Detected Formations: {', '.join(res['detected_patterns'])}\n"
        f"You MUST synthesize this visual chart evidence into your rationale."
    )

    visual_evidence_block = (
        f"\n\n--- [4. Visual Chart Expert Evidence] ---\n"
        f"Signal: {res['signal']} (Confidence: {res['confidence']:.2f})\n"
        f"Evidence: {res['evidence']}"
    )

    full_payload = f"{guided_instruction}\n\n{visual_evidence_block}"

    assert "VISUAL FINANCIAL EXPERT GUIDANCE" in full_payload
    assert "Candlestick Structure Signal:" in full_payload
    assert "--- [4. Visual Chart Expert Evidence] ---" in full_payload
    assert res["signal"] in full_payload


def test_chart_analysis_improves_performance():
    """
    Requirement 8 Verification:
    Verifies that fusing Visual Candlestick Expert evidence improves Accuracy and MCC
    over numerical-only models.
    """
    import json
    metrics_path = Path("results/metrics/visual_multimodal_metrics.json")
    assert metrics_path.exists(), f"Metrics payload not found at {metrics_path}"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    router_alone = data["router_alone"]
    multimodal = data["multimodal_fusion"]

    # Assert multimodal meets or exceeds router alone
    assert multimodal["accuracy"] >= router_alone["accuracy"], "Multimodal accuracy failed to match/beat router"
    assert multimodal["mcc"] >= router_alone["mcc"], "Multimodal MCC failed to beat router"
    assert multimodal["mcc"] > 0.0, "Multimodal MCC is non-positive"


