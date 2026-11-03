# PLOUTOS+ PHASE 9 RESEARCH REPORT: VISUAL FINANCIAL EXPERT
**Multimodal Candlestick Extension & Chart Pattern Recognition**  
**Author:** K. Rishikesh (24BDS030) | IIIT Dharwad  
**Course:** LLM Applications in Finance  
**Date:** 2026-11-02  

---

## 1. Executive Summary
In Phase 9, we developed and verified the **Visual Financial Expert**, the 4th expert modality in Ploutos+. 
While traditional financial LLMs (including the original *Ploutos* paper) are limited to tabular numerical indicators and social text, professional market analysts rely heavily on visual price charts to identify support/resistance levels, order flow absorption, and candlestick reversal formations.

The Visual Financial Expert:
1. Reconstructs strictly lagged 15-day Open, High, Low, Close (OHLC) price action with **zero lookahead bias** (strictly dates $< t$).
2. Algorithmic pattern recognition detects classic Japanese candlestick formations (Bullish Hammer Pin-Bars, Bearish Shooting Stars, Bullish/Bearish Engulfing, Moving Average slopes, and Volume breakouts).
3. Renders publication-grade dark-themed financial candlestick charts with volume histograms.
4. Synthesizes a structured natural-language evidence block incorporated as **Section 4** in Ploutos multi-expert prompt payloads.

---

## 2. Technical Architecture & Zero-Leakage Formulation

$$\text{Lookback Window: } [t - 15, t - 1] \implies \text{Strictly excludes target observation date } t$$

Given historical return series $\{r_\tau\}$ and OHLC ratios from StockNet, normalized price levels $P_\tau$ are computed forward across the lookback slice from a base of 100.0:
$$P_\tau = P_{\tau-1} \cdot (1 + \text{movement}_\tau)$$
$$O_\tau = P_{\tau-1} \cdot (1 + \text{open\_return}_\tau)$$
$$H_\tau = \max(P_{\tau-1} \cdot (1 + \text{high\_return}_\tau), O_\tau, P_\tau)$$
$$L_\tau = \min(P_{\tau-1} \cdot (1 + \text{low\_return}_\tau), O_\tau, P_\tau)$$

Automated assertions in [`tests/test_visual_expert.py`](../../tests/test_visual_expert.py) mathematically verify:
$$L_\tau \le O_\tau, C_\tau \le H_\tau \quad \forall \tau < t$$

---

## 3. Detected Candlestick Formations & Price Action Rules

| Chart Pattern | Structural Criteria | Market Interpretation | Directional Weight |
| :--- | :--- | :--- | :---: |
| **Bullish Hammer Pin-Bar** | Lower shadow $\ge 1.8 \times \text{body}$, Upper shadow $\le 0.5 \times \text{body}$ | Sellers rejected intraday; buyers stepped in at lows | **BULLISH** (+0.45) |
| **Bearish Shooting Star** | Upper shadow $\ge 1.8 \times \text{body}$, Lower shadow $\le 0.5 \times \text{body}$ | Buyers rejected at session highs; distribution | **BEARISH** (-0.45) |
| **Bullish Engulfing** | Green body completely covers previous red body | Immediate buyer dominance over past supply | **BULLISH** (+0.40) |
| **Bearish Engulfing** | Red body completely covers previous green body | Immediate seller dominance over past demand | **BEARISH** (-0.40) |
| **Support / Resistance Test** | Price tests 15-day minimum/maximum within 1.5% | Range boundary test indicating bounce or breakout | Calibrated ($\pm 0.35$) |
| **SMA Alignment** | 5-day SMA relative to 15-day SMA | Short-term trend momentum direction | Calibrated ($\pm 0.25$) |
| **Volume Expansion** | Volume $\ge 1.30 \times$ 15-day mean volume | Institutional order flow confirmation | Calibrated ($\pm 0.20$) |

---

## 4. Multimodal Prompt Integration

The Visual Expert injects an authentic 4th evidence block into the PloutosGen prompt schema:

```text
--- [4. Visual Chart Expert Evidence] ---
Signal: BEARISH (Confidence: 0.62)
Evidence: 15-day candlestick chart analysis indicates bearish price structure. 
Primary pattern: Short-term Moving Average Bearish Alignment (SMA-5: 101.95 vs SMA-15: 103.29) 
with volume ratio at 1.21x normal levels.
```

---

## 5. Artifacts Generated
* **Visual Expert Implementation:** `src/experts/visual_expert.py`
* **Prompt Builder Extension:** `src/experts/prompt_builder.py`
* **Inference CLI Integration:** `src/inference/predict.py`
* **Rendered Candlestick Charts:** `results/figures/charts/AAPL_2015-10-01.png`
* **Unit Test Suite:** `tests/test_visual_expert.py` (**6 / 6 PASSED**)
* **Full Repository Health:** **44 / 44 pytest unit tests PASSED** in 58.88s.

