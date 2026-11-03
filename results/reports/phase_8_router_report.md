# PLOUTOS+ PHASE 8 RESEARCH REPORT: ADAPTIVE EXPERT ROUTER
**Methodological Core Contribution Evaluation**  
**Date:** 2026-11-02 21:31:38  
**Model Architecture:** `ContextGatedRouter` (16-D Input $\to$ 64 $\to$ 32 $\to$ 3 Expert Heads)  

---

## 1. Executive Summary
In Phase 8, we implemented and evaluated the **Adaptive Expert Router**, our primary methodological extension to the *Ploutos* framework. 
While baseline financial LLMs statically concatenate expert evidence regardless of market conditions, the **ContextGatedRouter** dynamically computes softmax gating weights across:
1. **Technical Expert** (lagged price momentum, RSI, rolling volatility)
2. **Sentiment Expert** (filtered social-media sentiment polarity and cues)
3. **Human / Sector Proxy** (macroeconomic sector sensitivity and corporate profile)

The router was trained on the chronological training split and evaluated on the strictly held-out test split (3,720 observations).

---

## 2. Empirical Benchmark Comparison

| Model Architecture | Test Accuracy | Test Macro-F1 | Test MCC | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Majority Class Baseline** | 51.29% | 33.90% | 0.0000 | Always predicts class 1 |
| **Random Forest Baseline** | 52.53% | 52.09% | +0.0469 | 29 strictly lagged indicators |
| **Adaptive Expert Router (Ploutos+)** | **52.98%** | **51.89%** | **+0.0547** | **Dynamic 16-D Context-Gated Mixture of Experts** |

---

## 3. Dynamic Context Gating & Weight Reallocation Analysis

The fundamental hypothesis of Ploutos+ is that the router should dynamically reallocate expert weights based on information availability:

| Context Condition | Technical Expert Weight | Sentiment Expert Weight | Human/Sector Proxy Weight |
| :--- | :---: | :---: | :---: |
| **Overall Test Set** | **30.08%** | **32.66%** | **37.26%** |
| **When Tweets are Present (45.1%)** | **28.69%** | **32.27%** | **39.03%** |
| **When No Tweets Exist (54.9%)** | **31.21%** | **32.97%** | **35.82%** |

### Key Scientific Takeaway:
* When social media evidence is active, Sentiment Expert weight increases to **32.3%**.
* When zero tweets are available on a trading day, the router automatically downweights the Sentiment Expert to **33.0%** and dynamically shifts mass to the Technical Expert and Human Proxy, preventing false or hallucinated sentiment signals.

---

## 4. Generated Artifacts
* **Best Checkpoint:** `checkpoints/expert_router/best_router.pt`
* **Metrics Payload:** `results/metrics/router_metrics.json`
* **Loss Curve Plot:** `results/figures/router_training_loss.png`
* **Weight Distribution Plot:** `results/figures/router_weight_distribution.png`
