# Ploutos+ Phase 12: Interactive Web Dashboard Verification Report

**Authors:** Rishikesh (24BDS030), Vineel (24BDS010), Arjun (24BDS039) | IIIT Dharwad  
**Course:** LLM Applications in Finance  
**Date:** 2026-11-02 23:05:00  
**Status:** **100% Implemented, Tested & Operational**

---

## 1. Executive Summary
Phase 12 delivers an institutional-grade, full-stack interactive web application built with **Streamlit** and **Plotly** (`app.py`). The dashboard bridges every layer of the Ploutos+ research system—historical benchmarking, dynamic expert gating, multimodal candlestick visualization, real-time data ingestion, and forward performance tracking—into a unified, reactive user interface.

The application adheres to an institutional dark-themed visual design with high-contrast quantitative cards, interactive donut and grouped bar charts, expandable prompt inspection panels, and one-click forward tracking auditing.

---

## 2. Implemented Dashboard Modules & Capabilities

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             PLOUTOS+ STREAMLIT APP                               │
│                         (5 Reactive Operational Modes)                           │
└──────────────────────────────────────┬───────────────────────────────────────────┘
                                       │
      ┌────────────────────────────────┼────────────────────────────────┐
      ▼                                ▼                                ▼
[Mode 1: Live Predictor]    [Mode 2: Historical Audit]    [Mode 3: Forward Ledger]
 ├─ Ticker Search & Select   ├─ 3,720 Test Records         ├─ SQLite Forward DB
 ├─ Real-Time OHLCV & Sent   ├─ Ground Truth vs Model      ├─ Win Rate % & Realized Ret
 ├─ Interactive Router Donut ├─ Injected Prompt Inspector  ├─ Full Historical Audit Table
 ├─ 15d Candlestick PNG      └─ Historical Candlestick     └─ Settle Pending Button
 └─ Synthesized Rationale
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
         [Mode 4: Regime Stress Tests]       [Mode 5: System Architecture]
          ├─ Bull / Bear / Vol Breakdown      ├─ Multimodal Pipeline Diagram
          ├─ RF Collapse vs Ploutos+ Gain     ├─ Mathematical Specifications
          └─ Interactive Plotly Comparisons   └─ Zero-Leakage Proofs
```

### Module Breakdown:
1. **Mode 1: Real-Time Live Stock Predictor (`Ploutos-Live`):**
   - **User Input:** Bellwether presets (`NVDA`, `AAPL`, `TSLA`, `MSFT`, `GOOGL`, `AMZN`, `META`) or custom equity symbol.
   - **Executive Forecast Card:** Large directional badge (`▲ BULLISH` or `▼ BEARISH`), forecast confidence percentage, current market price, and 5-day price change.
   - **Adaptive Router Donut Chart:** Interactive Plotly donut chart depicting dynamic allocation across Technical ($w_{tech}$), Retail Sentiment ($w_{sent}$), and Fundamental/Sector ($w_{fund}$).
   - **Multimodal Evidence Panel:** 3-column breakdown displaying Momentum, RSI, Volatility, Volume Surge, Retail Bull/Bear Cues, Recent Tweet Quote, Corporate Beta, and Detected Candlestick Formations.
   - **Visual Candlestick Display:** High-resolution 15-day candlestick chart with moving average alignment and volume bars.
   - **Chain-of-Thought Rationale:** Synthesized multi-expert qualitative analysis.

2. **Mode 2: Historical Benchmark Audit:**
   - Browse the 3,720 out-of-sample StockNet evaluation records (2015-10-01 to 2016-01-01).
   - Side-by-side ground truth label verification with expandable raw prompt inspection.

3. **Mode 3: Forward Tracking Audit & Ledger:**
   - Real-time connection to SQLite database (`results/live_tracker.db`).
   - Summary KPIs: Total Predictions Tracked, Settled Predictions, Forward Hit Rate %, and Average Realized Return %.
   - Interactive data table showing prediction dates, target prices, forecast directions, realized outcomes, and settlement statuses.

4. **Mode 4: Market-Regime Robustness & Stress-Testing:**
   - Displays empirical stress-test results across Bull, Bear, Sideways, and High-Volatility market conditions.
   - Interactive Plotly charts illustrating the collapse of Random Forest in High-Volatility (49.4% Acc, -0.0103 MCC) versus the universal resilience of Ploutos+ Adaptive Router ($\min(\text{MCC}) = +0.0287$).

5. **Mode 5: System Architecture & User Guide:**
   - Complete technical pipeline flowcharts, mathematical formulas, and methodology notes.

---

## 3. Automated Test Verification

| Test Suite | Module Under Test | Tests Executed | Passed | Status |
| :--- | :--- | :---: | :---: | :--- |
| `tests/test_dashboard.py` | `app.py` Cached Resource Loaders & Figures | 5 | 5 | **PASSED** |
| `tests/test_live_fetcher.py`| Live Ingestion, Router & SQLite Tracker | 5 | 5 | **PASSED** |
| `tests/test_visual_expert.py`| Candlestick Detection & Chart Generation | 8 | 8 | **PASSED** |
| `tests/test_regime_analysis.py`| Partitioning & Stress-Testing Engine | 4 | 4 | **PASSED** |
| `tests/test_expert_router.py`| Gated Router Architecture & Gating | 8 | 8 | **PASSED** |
| `tests/test_llm_inference.py`| Prompt Construction & LLM Generation | 4 | 4 | **PASSED** |
| `tests/test_baselines.py` | Majority & Random Forest Baselines | 4 | 4 | **PASSED** |
| `tests/test_leakage.py` | Zero Temporal Lookahead Verification | 3 | 3 | **PASSED** |
| `tests/test_text_processing.py`| Tweet Cleaning & Tokenization | 5 | 5 | **PASSED** |
| `tests/test_temporal_splits.py`| Chronological Train/Val/Test Splits | 4 | 4 | **PASSED** |
| `tests/test_labels.py` | Stock Movement Return & Label Logic | 5 | 5 | **PASSED** |
| `tests/test_experts.py` | Baseline Technical & Sentiment Experts | 5 | 5 | **PASSED** |
| **Total Test Suite** | **Entire Ploutos+ Codebase (Phases 1-12)** | **60** | **60** | **100% PASSED** (7.48s) |

---

## 4. How to Launch the Application
Run the following command from the project root:
```powershell
.\.venv\Scripts\streamlit run app.py
```
*The interactive dashboard will open automatically in the default web browser at `http://localhost:8501`.*

