# Ploutos+: Interpretable Stock Movement Prediction with Financial LLMs

[![Python 3.10](https://img.shields.io/badge/python-3.10.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1%20%2B%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-69%20%2F%2069%20passed-brightgreen.svg)]()

> **Institutional Quantitative Trading & Multimodal AI Research Platform**  
> *Adaptive Context-Gated Mixture-of-Experts, Candlestick Vision Geometry & Dynamic Risk Guardrails*

---

### Authors & Academic Affiliation
* **Rishikesh** (`24BDS030`) — *System Architect & Data Pipeline Lead*
* **Vineel** (`24BDS010`) — *PEFT LoRA Engineer & Mathematical Modeler*
* **Arjun** (`24BDS039`) — *Quantitative Strategist & Full-Stack Terminal Lead*
* **Institution**: Indian Institute of Information Technology, Dharwad (IIIT Dharwad)
* **Course**: LLM Applications in Finance
* **Core Research Paper**: *"Ploutos: Towards Interpretable Stock Movement Prediction with Financial Large Language Model"* (Tong et al., ACL / arXiv:2407.13524, 2024)

---

## 1. Project Overview & Key Contributions

Ploutos+ addresses the core challenge of financial machine learning: **predicting next-day directional price movement ($t+1$) while eliminating black-box opacity and surviving non-stationary market regime shifts**.

### Core Scientific Innovations
1. **Adaptive Neural Mixture-of-Experts (MoE) Router**: Replaces the static 1/3 heuristics of the original paper with a 16-dimensional MLP gating network that dynamically shifts modal weights $[\alpha_{\text{Tech}}, \alpha_{\text{Sent}}, \alpha_{\text{Fund}}]$ based on volatility, RSI divergence, and news volume.
2. **Visual Financial Expert (Candlestick Geometry)**: Employs computer vision spatial pattern recognition (hammers, shooting stars, engulfing bars, MA crosses) over 15-day OHLC candlestick charts with strict zero lookahead leakage ($t < t+1$).
3. **Actionable Risk Guardrail Engine**: Translates binary directional outputs into hedge-fund capital allocation verdicts (`STRONG BUY`, `SCALE IN`, `HOLD CASH`, `DO NOT INVEST`) and volatility-calibrated Stop-Loss ($1.5\sigma$) and Take-Profit ($2.5\sigma$) price levels.
4. **Institutional Trading Terminal**: Full-stack asynchronous FastAPI backend (`server.py`) with TradingView Lightweight Charts, ApexCharts MoE donut visualizations, multi-currency detection (USD and INR ₹), and an automated forward tracking audit ledger.

---

## 2. Experimental Benchmark Results

Evaluated across **3,720 strictly held-out out-of-sample observations** from the **StockNet benchmark** (87 equities, 2015-10-01 to 2015-12-31):

### Comprehensive Ablation Study Matrix
| Architecture Configuration | Accuracy | Macro F1 | MCC | $\Delta$ MCC vs. RF |
| :--- | :---: | :---: | :---: | :---: |
| Majority Class Baseline | 51.29% | 33.90% | 0.0000 | -100.0% |
| **Stage 1 Random Forest Baseline** | 52.53% | 51.73% | +0.0469 | Base (0.0%) |
| **Stage 1 Static Ploutos (Paper)** | 52.19% | 51.10% | +0.0274 | -41.6% |
| Stage 2 Technical Expert Only | 52.42% | 51.57% | +0.0441 | -6.0% |
| Stage 2 Sentiment Expert Only | 51.77% | 49.34% | +0.0332 | -29.2% |
| Stage 2 Visual Expert Only | 52.12% | 50.89% | +0.0398 | -15.1% |
| Ploutos+ (Adaptive Router Only) | 52.96% | 51.86% | +0.0541 | +15.4% |
| **Ploutos+ Full Multimodal MoE System** | **53.32%** | **52.88%** | **+0.0596** | **+27.1%** |

* **+1.13% Absolute Accuracy Gain** over Stage 1 Static Ploutos ($52.19\% \to 53.32\%$).
* **+117.5% Relative MCC Gain** over Stage 1 ($+0.0274 \to +0.0596$).
* **+27.1% Relative MCC Gain** over Random Forest ($+0.0469 \to +0.0596$).

### Market-Regime Robustness Stress-Tests
| Economic Regime (Sample Size) | Random Forest Acc | Random Forest MCC | Ploutos+ Acc | Ploutos+ MCC |
| :--- | :---: | :---: | :---: | :---: |
| **Bull Regime (956 days)** | 47.91% | -0.0444 *(Collapse)* | **51.99%** | **+0.0349** |
| **Bear Regime (629 days)** | 58.19% | +0.1129 | **58.82%** | **+0.1230** |
| **Sideways Regime (1,205 days)** | 54.69% | +0.1066 | **50.71%** | **+0.0287** |
| **High-Volatility Regime (930 days)** | 49.35% | -0.0103 *(Collapse)* | **52.90%** | **+0.0401** |
| **OVERALL TEST SPLIT (3,720 days)** | **52.53%** | **+0.0469** | **52.98%** | **+0.0547** |

> **Critical Proof**: Random Forest collapses into negative correlation during market shocks (MCC = $-0.0444$ in bull markets, $-0.0103$ in high volatility). **Ploutos+ maintains strictly positive MCC across ALL 4 market regimes.**

---

## 3. Key Submission Deliverables

* **Presentation Slide Deck**: [`results/Ploutos_Plus_Presentation.pptx`](results/Ploutos_Plus_Presentation.pptx)  
  *Widescreen 16:9, 8 slides (7 core + 1 allowed results slide), dark theme, embeds both compulsory block diagrams.*
* **Comprehensive Project Report (PDF)**: [`results/reports/PLOUTOS_PLUS_COMPREHENSIVE_FINAL_DEFENSE_REPORT.pdf`](results/reports/PLOUTOS_PLUS_COMPREHENSIVE_FINAL_DEFENSE_REPORT.pdf)  
  *7-page formal academic report strictly formatted according to Part B guidelines with full equations and tables.*
* **Defense Spoken Guide (PDF)**: [`results/reports/PLOUTOS_PLUS_BEGINNER_TO_EXPERT_DEFENSE_GUIDE.pdf`](results/reports/PLOUTOS_PLUS_BEGINNER_TO_EXPERT_DEFENSE_GUIDE.pdf)  
  *Word-for-word spoken scripts and Q&A defense answers for Rishikesh, Vineel, and Arjun.*

---

## 4. Quickstart & Execution Guide

### 1. Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Complete Automated Test Suite (69 Tests)
```powershell
pytest tests/
```

### 3. Launch the Institutional Trading Terminal
```powershell
python server.py
```
Open your browser at `http://localhost:8000` to interact with the real-time TradingView chart, multi-currency detection, ApexCharts MoE donut, and forward audit ledger.

### 4. Regenerate Deliverables
```powershell
# Generate compulsory block diagram figures
python scripts/generate_block_diagrams.py

# Generate official 8-slide presentation
python scripts/generate_defense_presentation.py

# Generate official 7-page defense PDF report
python scripts/generate_comprehensive_pdf_report.py
```

---

## 5. Repository Structure
```
LLM-PROJECT/
├── configs/               # System and hyperparameter YAML configurations
├── data/
│   └── processed/         # Held-out benchmark test prompts (ploutos_test_prompts.jsonl)
├── results/
│   ├── figures/           # Publication charts & compulsory block diagrams
│   ├── metrics/           # Regime and ablation JSON evaluation matrices
│   ├── reports/           # Official academic defense PDF reports
│   └── Ploutos_Plus_Presentation.pptx # Official Part A presentation deck
├── scripts/               # Deliverable and benchmark generation scripts
├── src/
│   ├── data/              # Label construction, temporal splitting & tweet parsing
│   ├── evaluation/        # Macro F1, MCC, accuracy & regime metrics
│   ├── experts/           # Technical, sentiment, visual & human expert modules
│   ├── features/          # Lagged technical indicator & leakage checks
│   ├── inference/         # Prediction routines & prompt builders
│   ├── live/              # Real-time yfinance fetcher, ledger & investment engine
│   └── models/            # Adaptive neural router & baseline architectures
├── tests/                 # 69 automated unit and integration tests
├── server.py              # Institutional FastAPI web application
├── app.py                 # Streamlit analytical dashboard
└── requirements.txt       # Frozen environment dependencies
```

---

## License
MIT License. Developed for academic research at IIIT Dharwad.
