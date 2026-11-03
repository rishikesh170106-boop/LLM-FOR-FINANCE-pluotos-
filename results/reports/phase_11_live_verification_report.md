# Ploutos+ Phase 11: Real-Time Live Market Ingestion & Prediction Audit

**Author:** K. Rishikesh (Roll: 24BDS030), B.Tech DSAI, IIIT Dharwad  
**Course:** LLM Applications in Finance  
**Date:** 2026-11-02 22:51:16  
**Status:** **100% Verified & Fully Functional**

---

## 1. Executive Summary
Phase 11 extends the historical StockNet benchmark pipeline into an operational, production-ready **Real-Time Live Market Ingestion & Forward Auditing Engine (Ploutos-Live)**. The engine connects modern market data (`yfinance` OHLCV streams, `StockTwits` public sentiment feeds, and corporate sector profiles) with the Adaptive Expert Router, Visual Candlestick Pattern Expert, and fine-tuned Qwen2.5-1.5B LLM.

Live predictions are automatically logged to a persistent SQLite audit database (`results/live_tracker.db`), enabling forward out-of-sample directional accuracy and realized return tracking as next-day markets settle.

---

## 2. Live Bellwether Predictions Summary Table

| Ticker | Last Close | Forecast Direction | Dominant Expert | Dynamic Weights (Tech / Sent / Fund) | Visual Formations | Chart File |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$AAPL** | $338.26 | **BEARISH** (59%) | Human/Sector Proxy | 25% / 28% / 47% | Neutral range-bound price action with no extreme breakout patterns | [`AAPL_live.png`](../figures/charts/AAPL_live.png) |
| **$NVDA** | $206.75 | **BEARISH** (82%) | Human/Sector Proxy | 27% / 30% / 43% | Bearish Shooting Star (Upper wick resistance rejection), Short-term Moving Average Bearish Alignment (SMA-5 < SMA-15) | [`NVDA_live.png`](../figures/charts/NVDA_live.png) |
| **$TSLA** | $411.23 | **BULLISH** (80%) | Technical Expert | 35% / 33% / 33% | Short-term Moving Average Bullish Alignment (SMA-5 > SMA-15) | [`TSLA_live.png`](../figures/charts/TSLA_live.png) |
| **$MSFT** | $521.22 | **BEARISH** (62%) | Human/Sector Proxy | 25% / 28% / 46% | Bearish Shooting Star (Upper wick resistance rejection) | [`MSFT_live.png`](../figures/charts/MSFT_live.png) |

---

## 3. Forward Tracking Audit Summary

- **Total Live Predictions Tracked:** 12
- **Total Settled Predictions:** 12
- **Settled Directional Hits:** 7
- **Forward Hit Rate (Accuracy):** **58.33%**
- **Average Realized Return:** **+7.11%**
- **Database Path:** `results/live_tracker.db`
- **Audit Metrics File:** `results/metrics/live_audit_summary.json`

---

## 4. Architectural Verification Checklist

| Requirement | Implementation Component | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **Real-time Price Stream** | `LiveMarketFetcher.fetch_price_history()` | Ingests 30-day OHLCV via `yfinance` with offline synthetic fallback | **VERIFIED** |
| **Retail Sentiment Ingestion** | `LiveMarketFetcher.fetch_social_sentiment()` | Queries StockTwits Public API & Yahoo News headlines | **VERIFIED** |
| **16-D Context Vector Construction**| `LiveMarketFetcher.construct_live_context_vector()` | Normalizes momentum, RSI, volatility, and sentiment metrics | **VERIFIED** |
| **Live Candlestick Chart Rendering**| `VisualFinancialExpert.render_candlestick_chart()` | Dark-themed PNG candlestick charts with volume bars | **VERIFIED** |
| **Live Candlestick Pattern Detection**| `VisualFinancialExpert.detect_chart_patterns()` | Identifies Hammer, Engulfing, SMA crossovers, Vol surges | **VERIFIED** |
| **Live Adaptive Router Execution** | `AdaptiveRouterEngine.route_single()` | Dynamic weighting based on real-time market context | **VERIFIED** |
| **Forward Prediction Persistence** | `PredictionTracker.log_prediction()` | SQLite database (`results/live_tracker.db`) with PENDING status | **VERIFIED** |
| **Forward Hit/Miss Settlement** | `PredictionTracker.update_realized_outcome()` | Realized return and directional accuracy audit on next-day close | **VERIFIED** |
