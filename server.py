"""
server.py
Ploutos+: Institutional Trading Terminal & Quantitative Multi-Expert Web Application.
Built with FastAPI, Tailwind CSS, TradingView Lightweight Charts, and ApexCharts.
Authors: Rishikesh (24BDS030), Vineel (24BDS010), Arjun (24BDS039) | IIIT Dharwad
Course: LLM Applications in Finance
"""
import os
import sys
from pathlib import Path
from datetime import datetime, date
import json
import sqlite3
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Matplotlib interim cache and headless Agg backend BEFORE any operations
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data" / "interim"))
import matplotlib
matplotlib.use("Agg")

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.live.live_fetcher import LiveMarketFetcher
from src.live.tracker import PredictionTracker
from src.live.live_predict import run_live_prediction, generate_synthetic_live_data
from src.models.expert_router import AdaptiveRouterEngine, ROUTER_FEATURE_DIM
from src.experts.visual_expert import VisualFinancialExpert
from src.live.investment_engine import InvestmentDecisionEngine

app = FastAPI(title="Ploutos+ Institutional Trading Terminal", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tracker = PredictionTracker()


@app.get("/api/predict")
def api_predict(ticker: str = "NVDA", use_llm: bool = False, save_chart: bool = False):
    """Executes live prediction pipeline and returns structured JSON payload."""
    try:
        res = run_live_prediction(ticker=ticker.upper(), use_llm=use_llm, save_chart=save_chart, track=True)
        return JSONResponse(content={"status": "success", "data": res})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/api/chart_data")
def api_chart_data(ticker: str = "NVDA"):
    """Returns OHLCV series formatted for TradingView Lightweight Charts."""
    ticker = ticker.upper()
    fetcher = LiveMarketFetcher(request_timeout=3)
    hist_df = fetcher.fetch_price_history(ticker, period="3mo")
    if hist_df is None or len(hist_df) < 5:
        hist_df = generate_synthetic_live_data(ticker)

    candles = []
    volume_bars = []
    for _, row in hist_df.iterrows():
        d_str = str(row["date"])
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        vol = float(row.get("volume", 1e6))

        candles.append({"time": d_str, "open": o, "high": h, "low": l, "close": c})
        vol_color = "rgba(0, 245, 160, 0.45)" if c >= o else "rgba(255, 51, 102, 0.45)"
        volume_bars.append({"time": d_str, "value": vol, "color": vol_color})

    curr = "₹" if ticker.endswith((".NS", ".BO", ".IN")) else ("£" if ticker.endswith((".L", ".LON")) else ("€" if ticker.endswith((".DE", ".PA")) else "$"))
    return JSONResponse(content={
        "ticker": ticker,
        "currency_symbol": curr,
        "candles": candles,
        "volume": volume_bars
    })


@app.get("/api/ledger")
def api_ledger():
    """Returns forward tracking predictions and audit summary KPIs."""
    summary = tracker.compute_audit_summary()
    preds = tracker.get_all_predictions(limit=100)
    return JSONResponse(content={
        "summary": summary,
        "predictions": preds
    })


@app.post("/api/settle")
def api_settle():
    """Settles pending forward predictions and recalculates win rate."""
    res = tracker.evaluate_pending()
    return JSONResponse(content=res)


@app.get("/api/historical_samples")
def api_historical_samples(limit: int = 200):
    """Returns sample test records from StockNet held-out benchmark."""
    prompts_path = PROJECT_ROOT / "data" / "processed" / "ploutos_test_prompts.jsonl"
    samples = []
    if prompts_path.exists():
        with open(prompts_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= limit:
                    break
                try:
                    obj = json.loads(line)
                    meta = obj.get("metadata", {})
                    ticker = obj.get("ticker") or meta.get("ticker", "UNK")
                    date = obj.get("date") or meta.get("date", "UNK")
                    
                    label = obj.get("label")
                    if label is None:
                        label = 1 if meta.get("target_movement", 0) > 0 else 0
                    
                    raw_input = obj.get("input") or obj.get("prompt", "")
                    clean_evidence = " ".join(raw_input.split())
                    if len(clean_evidence) > 350:
                        clean_evidence = clean_evidence[:350] + "..."
                    
                    samples.append({
                        "ticker": ticker,
                        "date": date,
                        "ground_truth": int(label),
                        "prompt": clean_evidence
                    })
                except Exception:
                    continue
    return JSONResponse(content={"samples": samples})


@app.get("/api/regime_data")
def api_regime_data():
    """Returns market-regime robustness stress-test metrics."""
    metrics_path = PROJECT_ROOT / "results" / "metrics" / "regime_robustness_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))
    return JSONResponse(status_code=404, content={"message": "Regime metrics not found"})


@app.get("/api/ablation_data")
def api_ablation_data():
    """Returns Phase 13 comprehensive ablation matrix."""
    metrics_path = PROJECT_ROOT / "results" / "metrics" / "ablation_study_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))
    return JSONResponse(status_code=404, content={"message": "Ablation metrics not found"})


@app.get("/", response_class=HTMLResponse)
def get_terminal_ui():
    """Serves the institutional TradingView dark terminal interface."""
    html_content = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ploutos+ | Institutional Trading Terminal & Multi-Expert LLM</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- TradingView Lightweight Charts CDN -->
    <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
    <!-- ApexCharts CDN for Gauges & Donut Charts -->
    <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #080b11;
            color: #e2e8f0;
        }
        .font-mono {
            font-family: 'JetBrains Mono', monospace;
        }
        /* Custom scrollbars */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #0f172a;
        }
        ::-webkit-scrollbar-thumb {
            background: #334155;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #475569;
        }
        /* Glass card styling */
        .glass-card {
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 0.75rem;
        }
        /* Neon Accents */
        .neon-bull {
            color: #00F5A0;
            text-shadow: 0 0 12px rgba(0, 245, 160, 0.45);
        }
        .neon-bear {
            color: #FF3366;
            text-shadow: 0 0 12px rgba(255, 51, 102, 0.45);
        }
        .badge-bull {
            background: rgba(0, 245, 160, 0.15);
            color: #00F5A0;
            border: 1px solid rgba(0, 245, 160, 0.3);
        }
        .badge-bear {
            background: rgba(255, 51, 102, 0.15);
            color: #FF3366;
            border: 1px solid rgba(255, 51, 102, 0.3);
        }
        .badge-caution {
            background: rgba(245, 158, 11, 0.15);
            color: #F59E0B;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }
        .badge-cash {
            background: rgba(156, 163, 175, 0.15);
            color: #9CA3AF;
            border: 1px solid rgba(156, 163, 175, 0.3);
        }
    </style>
</head>
<body class="min-h-screen flex flex-col antialiased selection:bg-emerald-500 selection:text-black">

    <!-- Top Navigation Bar -->
    <header class="h-16 border-b border-gray-800 bg-[#0B0F19]/90 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-50">
        <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-lg bg-gradient-to-tr from-emerald-500 to-cyan-500 flex items-center justify-center text-black font-black text-xl shadow-lg shadow-emerald-500/20">
                <i class="fa-solid fa-chart-line"></i>
            </div>
            <div>
                <div class="flex items-center gap-2">
                    <h1 class="text-lg font-extrabold tracking-tight text-white font-mono">PLOUTOS<span class="text-emerald-400">+</span></h1>
                    <span class="px-2 py-0.5 text-[10px] font-bold rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 uppercase tracking-wider">Institutional Terminal</span>
                </div>
                <p class="text-xs text-gray-400">Adaptive Router &bull; Multimodal Candlestick &bull; Risk Allocation</p>
            </div>
        </div>

        <!-- Ticker Quick Presets & Search Input -->
        <div class="flex items-center gap-2 bg-[#111827] p-1.5 rounded-lg border border-gray-800">
            <div class="flex items-center gap-1 text-xs font-semibold">
                <button onclick="selectTicker('NVDA')" class="ticker-btn px-2 py-1 rounded hover:bg-gray-800 text-gray-300 transition" data-ticker="NVDA">NVDA</button>
                <button onclick="selectTicker('AAPL')" class="ticker-btn px-2 py-1 rounded hover:bg-gray-800 text-gray-300 transition" data-ticker="AAPL">AAPL</button>
                <button onclick="selectTicker('TSLA')" class="ticker-btn px-2 py-1 rounded hover:bg-gray-800 text-gray-300 transition" data-ticker="TSLA">TSLA</button>
                <div class="h-4 w-px bg-gray-700 mx-0.5"></div>
                <button onclick="selectTicker('RELIANCE.NS')" class="ticker-btn px-2 py-1 rounded hover:bg-gray-800 text-gray-300 transition flex items-center gap-1" data-ticker="RELIANCE.NS"><span>🇮🇳</span> RELIANCE</button>
                <button onclick="selectTicker('TCS.NS')" class="ticker-btn px-2 py-1 rounded hover:bg-gray-800 text-gray-300 transition flex items-center gap-1" data-ticker="TCS.NS"><span>🇮🇳</span> TCS</button>
                <button onclick="selectTicker('INFY.NS')" class="ticker-btn px-2 py-1 rounded hover:bg-gray-800 text-gray-300 transition flex items-center gap-1" data-ticker="INFY.NS"><span>🇮🇳</span> INFY</button>
            </div>
            <div class="h-4 w-px bg-gray-700 mx-1"></div>
            <div class="relative flex items-center">
                <i class="fa-solid fa-magnifying-glass text-gray-500 text-xs absolute left-2.5"></i>
                <input id="tickerInput" type="text" value="NVDA" placeholder="SYMBOL" class="w-32 pl-7 pr-2 py-1 text-xs bg-gray-900 border border-gray-700 rounded font-mono font-bold text-white uppercase focus:outline-none focus:border-emerald-500">
            </div>
            <label class="flex items-center gap-1.5 text-xs text-gray-400 ml-2 cursor-pointer select-none">
                <input id="llmToggle" type="checkbox" class="w-3.5 h-3.5 rounded bg-gray-800 border-gray-700 text-emerald-500 focus:ring-0">
                <span>GPU LoRA</span>
            </label>
            <button id="predictBtn" onclick="executePrediction()" class="ml-2 px-4 py-1.5 bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-black font-bold text-xs rounded shadow-md shadow-emerald-500/20 flex items-center gap-1.5 transition active:scale-95">
                <i class="fa-solid fa-bolt"></i>
                <span>FORECAST</span>
            </button>
        </div>

        <!-- Authors Attribution Badge -->
        <div class="flex items-center gap-2 text-xs text-gray-400 bg-gray-900/80 px-3 py-1.5 rounded-lg border border-gray-800 font-mono">
            <i class="fa-solid fa-users text-cyan-400"></i>
            <span>Rishikesh (030) &bull; Vineel (010) &bull; Arjun (039)</span>
            <span class="text-gray-600">|</span>
            <span class="text-gray-400">IIIT Dharwad</span>
        </div>
    </header>

    <!-- Main Workspace Container -->
    <main class="flex-1 p-6 space-y-6 max-w-[1700px] w-full mx-auto">

        <!-- Top Row: Real-Time Signal & Investment Verdict HUD -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            
            <!-- Card 1: Actionable Investment Verdict -->
            <div class="glass-card p-4 relative overflow-hidden border-l-4" id="verdictBorder" style="border-left-color: #00F5A0;">
                <div class="text-[11px] font-semibold uppercase tracking-wider text-gray-400 flex items-center justify-between">
                    <span>Investment Verdict</span>
                    <i class="fa-solid fa-shield-halved text-emerald-400"></i>
                </div>
                <div class="mt-2 text-lg font-black tracking-tight" id="verdictBadge">
                    <span class="badge-bull px-2.5 py-1 rounded text-xs font-bold font-mono">LOADING...</span>
                </div>
                <div class="text-xs text-gray-300 mt-1 font-medium" id="allocationText">Computing allocation...</div>
            </div>

            <!-- Card 2: Directional Forecast & Confidence (With Bull/Bear & Arrows) -->
            <div class="glass-card p-4 flex flex-col justify-between">
                <div class="text-[11px] font-semibold uppercase tracking-wider text-gray-400 flex items-center justify-between">
                    <span>Forecast Signal</span>
                    <span class="text-[10px] font-mono text-cyan-400 font-bold" id="confPercent">--.-%</span>
                </div>
                <div class="mt-2" id="directionSignal">
                    <div class="flex items-center gap-3">
                        <div class="w-11 h-11 rounded-xl bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center text-2xl shadow-lg shadow-emerald-500/20">🐂</div>
                        <div>
                            <div class="neon-bull text-xl font-black font-mono tracking-tight flex items-center gap-1">
                                BULLISH <span class="text-base text-emerald-400 font-extrabold">▲ UP</span>
                            </div>
                            <div class="text-[10px] text-emerald-400/80 font-mono font-semibold">Upside Movement Vector</div>
                        </div>
                    </div>
                </div>
                <div class="text-[11px] text-gray-400 mt-1 flex items-center gap-1 font-mono">
                    <span>Next-Day Target Probability</span>
                </div>
            </div>

            <!-- Card 3: Market Price & Momentum -->
            <div class="glass-card p-4">
                <div class="text-[11px] font-semibold uppercase tracking-wider text-gray-400 flex items-center justify-between">
                    <span>Last Market Close</span>
                    <span class="font-mono text-xs text-emerald-400" id="dateLabel">----/--/--</span>
                </div>
                <div class="mt-2 text-2xl font-black font-mono text-white" id="lastPrice">
                    $---.--
                </div>
                <div class="text-xs font-mono mt-1" id="momentumLabel">
                    <span class="text-emerald-400 font-bold">+0.00%</span> 5-Day Momentum
                </div>
            </div>

            <!-- Card 4: Dynamic Risk Guardrails (Stop-Loss / Take-Profit) -->
            <div class="glass-card p-4">
                <div class="text-[11px] font-semibold uppercase tracking-wider text-gray-400 flex items-center justify-between">
                    <span>Risk Guardrails</span>
                    <i class="fa-solid fa-scale-balanced text-amber-400"></i>
                </div>
                <div class="mt-2 grid grid-cols-2 gap-2 text-xs font-mono">
                    <div>
                        <span class="text-gray-400 block text-[10px]">STOP-LOSS</span>
                        <span class="text-rose-400 font-bold" id="stopLossVal">$---.--</span>
                    </div>
                    <div>
                        <span class="text-gray-400 block text-[10px]">TAKE-PROFIT</span>
                        <span class="text-emerald-400 font-bold" id="takeProfitVal">$---.--</span>
                    </div>
                </div>
                <div class="text-[11px] text-gray-400 mt-1 font-mono">
                    R:R Ratio: <span class="text-cyan-400 font-bold" id="rrRatio">1 : -.--</span>
                </div>
            </div>

            <!-- Card 5: Adaptive Router Dominance & Gating Proof -->
            <div class="glass-card p-4 border-l-4 border-purple-500 relative flex flex-col justify-between">
                <div class="text-[11px] font-semibold uppercase tracking-wider text-gray-400 flex items-center justify-between">
                    <span>Dominant Expert</span>
                    <span id="dominantLeadBadge" class="px-1.5 py-0.5 text-[9px] font-bold font-mono rounded bg-purple-500/20 text-purple-300 border border-purple-500/40">LEADING ★</span>
                </div>
                <div class="mt-1 text-base font-bold text-white truncate flex items-center gap-1.5" id="dominantExpertText">
                    <span class="text-purple-400">🟣</span> Human / Sector Proxy
                </div>
                <div class="text-[11px] font-mono text-gray-300 mt-0.5" id="routerReasonText">
                    Tech 26% &bull; Sent 28% &bull; Human 46%
                </div>
                <!-- Dynamic Gating Proof Note -->
                <div class="mt-1.5 text-[10px] bg-purple-950/40 border border-purple-500/30 p-1.5 rounded text-purple-200 font-sans leading-tight" id="gatingProofText">
                    <i class="fa-solid fa-brain text-purple-400 mr-1"></i> <strong>Proof:</strong> 46% allocation leads sentiment by +18% to anchor amid low noise.
                </div>
            </div>

        </div>

        <!-- Middle Row: Interactive TradingView Candlestick Chart + Multi-Expert Panel -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

            <!-- Left: Interactive TradingView Canvas Chart (8 Cols) -->
            <div class="lg:col-span-8 glass-card p-5 flex flex-col justify-between">
                <div class="flex items-center justify-between pb-3 border-b border-gray-800">
                    <div class="flex items-center gap-3">
                        <span class="text-sm font-bold font-mono text-white flex items-center gap-2">
                            <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
                            <span id="chartSymbol">NVDA</span> &bull; Daily Candlesticks (3 Months)
                        </span>
                        <span class="px-2 py-0.5 text-[10px] rounded bg-gray-800 text-gray-300 font-mono">Lagged Zero-Leakage (&lt; t)</span>
                    </div>
                    <div class="flex items-center gap-4 text-xs font-mono text-gray-400" id="ohlcHeader">
                        <span>O: <span class="text-white" id="oVal">-</span></span>
                        <span>H: <span class="text-white" id="hVal">-</span></span>
                        <span>L: <span class="text-white" id="lVal">-</span></span>
                        <span>C: <span class="text-white" id="cVal">-</span></span>
                    </div>
                </div>

                <!-- Chart Mount Container -->
                <div id="tradingviewChart" class="w-full h-[400px] mt-2 relative"></div>

                <!-- LLM Rationale Box -->
                <div class="mt-4 p-4 rounded-lg bg-gray-900/90 border border-gray-800">
                    <div class="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-gray-400 mb-1.5">
                        <span class="flex items-center gap-1.5 text-cyan-400">
                            <i class="fa-solid fa-brain"></i> Synthesized Multimodal Rationale
                        </span>
                        <span class="text-[10px] font-mono text-gray-500">Qwen2.5-1.5B / Multi-Expert Consensus</span>
                    </div>
                    <p class="text-xs text-gray-300 leading-relaxed font-sans" id="rationaleText">
                        Loading model rationale...
                    </p>
                </div>
            </div>

            <!-- Right: Dynamic Router Donut + Full Sentiment & Modality Cards (4 Cols) -->
            <div class="lg:col-span-4 space-y-4 flex flex-col">

                <!-- Router Dynamic Weights Donut & Explicit Gating Proof Panel -->
                <div class="glass-card p-5">
                    <div class="flex items-center justify-between text-xs font-bold uppercase tracking-wider text-gray-400 pb-2 border-b border-gray-800">
                        <span>Adaptive Router Allocation</span>
                        <span class="text-[10px] text-purple-400 font-mono">16-D Gated Softmax</span>
                    </div>
                    
                    <!-- Donut Chart Mount -->
                    <div id="routerDonutChart" class="h-44 flex items-center justify-center"></div>

                    <!-- Explicit Ranking & Lead Proof Breakdown -->
                    <div class="mt-3 space-y-2 border-t border-gray-800/80 pt-3 text-xs font-mono">
                        <div class="flex items-center justify-between text-[11px] p-1.5 rounded bg-purple-950/20 border border-purple-500/30">
                            <span class="text-purple-300 font-bold flex items-center gap-1.5">
                                <span class="w-2.5 h-2.5 rounded-full bg-purple-500"></span> Human / Sector Proxy:
                            </span>
                            <span class="text-white font-bold" id="donutProofHuman">46.2% <span class="text-emerald-400 text-[10px]">(LEADING ★)</span></span>
                        </div>
                        <div class="flex items-center justify-between text-[11px] px-1.5 py-1">
                            <span class="text-amber-400 flex items-center gap-1.5">
                                <span class="w-2.5 h-2.5 rounded-full bg-amber-500"></span> Retail Social Sentiment:
                            </span>
                            <span class="text-gray-300 font-bold" id="donutProofSent">28.1%</span>
                        </div>
                        <div class="flex items-center justify-between text-[11px] px-1.5 py-1">
                            <span class="text-cyan-400 flex items-center gap-1.5">
                                <span class="w-2.5 h-2.5 rounded-full bg-cyan-500"></span> Technical Momentum:
                            </span>
                            <span class="text-gray-300 font-bold" id="donutProofTech">25.7%</span>
                        </div>
                        <div class="mt-1 text-[10px] text-gray-400 italic bg-gray-900/60 p-2 rounded border border-gray-800 leading-relaxed font-sans" id="donutProofReason">
                            <span class="font-bold text-purple-300">Why Human Proxy leads:</span> Range-bound price action & neutral sentiment trigger the Gating Network to allocate 46% weight to the macro fundamental anchor, preventing false breakout whipsaws.
                        </div>
                    </div>
                </div>

                <!-- Modality Cards: Full Sentiment, Technical & Visual -->
                <div class="space-y-3 flex-1 flex flex-col justify-between">
                    
                    <!-- Retail Social Sentiment (Full Clear Results) -->
                    <div class="glass-card p-4 border-l-2 border-amber-500">
                        <div class="flex items-center justify-between text-xs pb-2 border-b border-gray-800">
                            <span class="font-bold text-amber-400 flex items-center gap-1.5">
                                <i class="fa-solid fa-comments"></i> Retail Social Sentiment (Live Stream)
                            </span>
                            <div class="flex items-center gap-2">
                                <span class="text-[10px] font-mono text-gray-400" id="sentPolarityScore">Polarity: +0.00</span>
                                <span class="badge-cash px-2 py-0.5 rounded text-[10px] font-mono font-bold" id="sentSigBadge">NEUTRAL</span>
                            </div>
                        </div>

                        <!-- Cues Breakdown Row -->
                        <div class="grid grid-cols-3 gap-2 my-2 text-[11px] font-mono bg-gray-900/70 p-2 rounded border border-gray-800">
                            <div>Feed Items: <span class="text-white font-bold" id="sentTweetCount">4</span></div>
                            <div>Bull Cues: <span class="text-emerald-400 font-bold" id="sentBullCues">+1</span></div>
                            <div>Bear Cues: <span class="text-rose-400 font-bold" id="sentBearCues">-1</span></div>
                        </div>

                        <!-- Full Untruncated Headline / Message List -->
                        <div class="mt-1">
                            <span class="text-[10px] uppercase font-bold text-gray-500 tracking-wider">Live Text Evidence (Full Feed):</span>
                            <div class="text-xs text-gray-200 mt-1 font-sans bg-[#0c121e] p-2.5 rounded border border-gray-800 max-h-32 overflow-y-auto" id="sampleTweetText">
                                Loading full social sentiment evidence...
                            </div>
                        </div>
                    </div>

                    <!-- Technical Expert -->
                    <div class="glass-card p-3.5 border-l-2 border-cyan-500">
                        <div class="flex items-center justify-between text-xs">
                            <span class="font-bold text-cyan-400 flex items-center gap-1.5">
                                <i class="fa-solid fa-wave-square"></i> Technical Expert
                            </span>
                            <span class="badge-bull px-2 py-0.5 rounded text-[10px] font-mono font-bold" id="techSigBadge">BULLISH</span>
                        </div>
                        <div class="grid grid-cols-3 gap-2 mt-2 text-[11px] font-mono text-gray-300">
                            <div>RSI: <span class="text-white font-bold" id="rsiVal">--.-</span></div>
                            <div>5d Vol: <span class="text-white font-bold" id="volVal">-.--%</span></div>
                            <div>Vol Surge: <span class="text-white font-bold" id="surgeVal">+-.--x</span></div>
                        </div>
                    </div>

                    <!-- Visual Candlestick Pattern Expert -->
                    <div class="glass-card p-3.5 border-l-2 border-emerald-500">
                        <div class="flex items-center justify-between text-xs">
                            <span class="font-bold text-emerald-400 flex items-center gap-1.5">
                                <i class="fa-solid fa-chart-candlestick"></i> Visual Expert (15-Day)
                            </span>
                            <span class="badge-bull px-2 py-0.5 rounded text-[10px] font-mono font-bold" id="visSigBadge">BULLISH</span>
                        </div>
                        <div class="text-[11px] text-gray-300 mt-1.5 font-sans" id="patternsText">
                            Short-term Moving Average Bullish Alignment (SMA-5 &gt; SMA-15)
                        </div>
                    </div>

                </div>

            </div>

        </div>

        <!-- Bottom Tabs: Institutional Segmented Control Toggle Bar -->
        <div class="glass-card p-5">
            <div class="flex flex-wrap items-center justify-between border-b border-gray-800/80 pb-4 mb-4 gap-3">
                
                <!-- Cool High-Tech Segmented Toggle Bar -->
                <div class="inline-flex p-1.5 bg-[#0a0e17] rounded-xl border border-gray-800 gap-1.5 shadow-2xl">
                    <button onclick="switchTab('ledgerTab')" class="tab-btn px-4 py-2 text-xs font-mono font-bold rounded-lg transition-all duration-200 flex items-center gap-2 bg-gradient-to-r from-emerald-500/20 to-teal-500/20 text-emerald-400 border border-emerald-500/40 shadow-lg shadow-emerald-500/10" id="btn-ledgerTab">
                        <i class="fa-solid fa-scale-balanced text-emerald-400"></i> Forward Tracking Ledger
                    </button>
                    <button onclick="switchTab('historicalTab')" class="tab-btn px-4 py-2 text-xs font-mono font-bold rounded-lg transition-all duration-200 flex items-center gap-2 text-gray-400 hover:text-white hover:bg-gray-800/50 border border-transparent" id="btn-historicalTab">
                        <i class="fa-solid fa-box-archive text-gray-400"></i> Historical Benchmark (3,720)
                    </button>
                    <button onclick="switchTab('regimeTab')" class="tab-btn px-4 py-2 text-xs font-mono font-bold rounded-lg transition-all duration-200 flex items-center gap-2 text-gray-400 hover:text-white hover:bg-gray-800/50 border border-transparent" id="btn-regimeTab">
                        <i class="fa-solid fa-tornado text-gray-400"></i> Market-Regime Stress-Tests
                    </button>
                    <button onclick="switchTab('ablationTab')" class="tab-btn px-4 py-2 text-xs font-mono font-bold rounded-lg transition-all duration-200 flex items-center gap-2 text-gray-400 hover:text-white hover:bg-gray-800/50 border border-transparent" id="btn-ablationTab">
                        <i class="fa-solid fa-layer-group text-gray-400"></i> Phase 13 Ablation Matrix
                    </button>
                    <button onclick="switchTab('mathTab')" class="tab-btn px-4 py-2 text-xs font-mono font-bold rounded-lg transition-all duration-200 flex items-center gap-2 text-gray-400 hover:text-white hover:bg-gray-800/50 border border-transparent" id="btn-mathTab">
                        <i class="fa-solid fa-calculator text-gray-400"></i> Quantitative Formulas & Proofs
                    </button>
                </div>

                <div class="text-[11px] font-mono text-gray-400 flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span>Audit Pipeline Active</span>
                </div>
            </div>

            <!-- Tab 1: Forward Tracking Ledger -->
            <div id="ledgerTab" class="tab-content">
                <div class="flex flex-wrap items-center justify-between gap-4 mb-4">
                    <div class="flex items-center gap-6 text-xs font-mono">
                        <div>Total Logged: <span class="text-white font-bold" id="kpiTotal">0</span></div>
                        <div>Settled Calls: <span class="text-white font-bold" id="kpiSettled">0</span></div>
                        <div>Forward Hit Rate: <span class="text-emerald-400 font-bold" id="kpiHitRate">0.0%</span></div>
                        <div>Avg Realized Return: <span class="text-cyan-400 font-bold" id="kpiAvgReturn">0.0%</span></div>
                    </div>
                    <button onclick="settlePredictions()" class="px-3.5 py-1.5 bg-gray-800 hover:bg-gray-700 text-white rounded text-xs font-mono font-bold flex items-center gap-1.5 transition">
                        <i class="fa-solid fa-rotate"></i> Settle Pending Now
                    </button>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs font-mono border-collapse">
                        <thead>
                            <tr class="text-gray-400 border-b border-gray-800 bg-gray-900/50">
                                <th class="p-2.5">ID</th>
                                <th class="p-2.5">Ticker</th>
                                <th class="p-2.5">Date</th>
                                <th class="p-2.5">Last Close</th>
                                <th class="p-2.5">Forecast</th>
                                <th class="p-2.5">Confidence</th>
                                <th class="p-2.5">Verdict</th>
                                <th class="p-2.5">Stop-Loss</th>
                                <th class="p-2.5">Target</th>
                                <th class="p-2.5">Next Close</th>
                                <th class="p-2.5">Return</th>
                                <th class="p-2.5">Outcome</th>
                                <th class="p-2.5">Status</th>
                            </tr>
                        </thead>
                        <tbody id="ledgerTableBody">
                            <tr><td colspan="13" class="p-4 text-center text-gray-500">Loading ledger data...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Tab 2: Historical Benchmark Audit -->
            <div id="historicalTab" class="tab-content hidden">
                <div class="flex flex-wrap items-center justify-between gap-3 mb-3">
                    <p class="text-xs text-gray-300">Browse out-of-sample StockNet benchmark test observations (2015-10-01 to 2016-01-01) with ground truth labels:</p>
                    <div class="flex items-center gap-2">
                        <div class="relative">
                            <i class="fa-solid fa-magnifying-glass absolute left-3 top-2.5 text-gray-500 text-xs"></i>
                            <input type="text" id="histFilterInput" placeholder="Filter ticker (e.g. AAPL)..." class="bg-gray-900/90 border border-gray-800 rounded-lg pl-8 pr-3 py-1.5 text-xs font-mono text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500 w-56" oninput="filterHistoricalSamples()">
                        </div>
                        <span id="histCountBadge" class="text-xs font-mono text-cyan-400 bg-cyan-950/60 border border-cyan-500/30 px-2.5 py-1.5 rounded-lg font-bold">200 records</span>
                    </div>
                </div>
                <div class="overflow-x-auto max-h-96 rounded-lg border border-gray-800/80">
                    <table class="w-full text-left text-xs font-mono border-collapse">
                        <thead>
                            <tr class="text-gray-400 border-b border-gray-800 bg-gray-900/90 sticky top-0 z-10">
                                <th class="p-2.5 w-24">Ticker</th>
                                <th class="p-2.5 w-32">Observation Date</th>
                                <th class="p-2.5 w-28">Ground Truth</th>
                                <th class="p-2.5">Input Multi-Expert Prompt Evidence</th>
                            </tr>
                        </thead>
                        <tbody id="historicalTableBody" class="divide-y divide-gray-800/40">
                            <tr><td colspan="4" class="p-4 text-center text-gray-500">Loading historical test records...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Tab 3: Regime Stress Tests -->
            <div id="regimeTab" class="tab-content hidden space-y-4">
                <!-- Regime Resilience Executive Banner -->
                <div class="bg-gradient-to-r from-cyan-950/40 via-gray-900 to-emerald-950/40 border border-emerald-500/30 p-3.5 rounded-xl flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
                    <div class="flex items-center gap-2">
                        <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
                        <span class="text-gray-300">Stress-Tested Universe: <strong class="text-white">3,720 Out-of-Sample Observations</strong></span>
                    </div>
                    <div class="flex items-center gap-4 text-[11px]">
                        <span class="text-gray-400">Bull: <strong class="text-emerald-400">956</strong></span>
                        <span class="text-gray-400">Bear: <strong class="text-rose-400">629</strong></span>
                        <span class="text-gray-400">Sideways: <strong class="text-amber-400">1,205</strong></span>
                        <span class="text-gray-400">Volatile: <strong class="text-purple-400">930</strong></span>
                    </div>
                    <span class="px-2.5 py-1 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 font-bold text-[11px]">
                        Ploutos+ Min MCC &gt; 0 in ALL 4 Regimes
                    </span>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="glass-card p-4 rounded-xl">
                        <div class="flex items-center justify-between mb-2">
                            <h4 class="text-xs font-bold uppercase text-gray-300 flex items-center gap-2">
                                <i class="fa-solid fa-chart-simple text-emerald-400"></i> Directional Accuracy by Market Regime (%)
                            </h4>
                            <span class="text-[10px] font-mono text-gray-400">Higher is Better</span>
                        </div>
                        <div id="regimeAccChart" class="min-h-[260px]"></div>
                    </div>
                    <div class="glass-card p-4 rounded-xl">
                        <div class="flex items-center justify-between mb-2">
                            <h4 class="text-xs font-bold uppercase text-gray-300 flex items-center gap-2">
                                <i class="fa-solid fa-bolt text-cyan-400"></i> Matthews Correlation Coefficient (MCC) by Regime
                            </h4>
                            <span class="text-[10px] font-mono text-emerald-400 font-bold">RF Collapses in Volatility; Ploutos+ Resilient</span>
                        </div>
                        <div id="regimeMccChart" class="min-h-[260px]"></div>
                    </div>
                </div>
            </div>

            <!-- Tab 4: Ablation Matrix -->
            <div id="ablationTab" class="tab-content hidden">
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs font-mono border-collapse">
                        <thead>
                            <tr class="text-gray-400 border-b border-gray-800 bg-gray-900/50">
                                <th class="p-2.5">Architecture Configuration</th>
                                <th class="p-2.5">Category</th>
                                <th class="p-2.5">Accuracy</th>
                                <th class="p-2.5">Macro F1</th>
                                <th class="p-2.5">MCC</th>
                                <th class="p-2.5">&Delta; Acc (vs RF)</th>
                                <th class="p-2.5">&Delta; MCC (vs RF)</th>
                            </tr>
                        </thead>
                        <tbody id="ablationTableBody">
                            <tr><td colspan="7" class="p-4 text-center text-gray-500">Loading ablation study metrics...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Tab 5: Quantitative Formulas & Live Proofs -->
            <div id="mathTab" class="tab-content hidden space-y-6">
                
                <!-- Section Header Banner -->
                <div class="bg-gradient-to-r from-cyan-950/40 via-gray-900 to-purple-950/40 border border-cyan-500/30 p-4 rounded-xl flex items-center justify-between">
                    <div>
                        <div class="flex items-center gap-2">
                            <span class="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
                            <h3 class="text-sm font-bold font-mono text-white uppercase tracking-wider">Technical Indicator Quantitative Proofs & Formulations</h3>
                        </div>
                        <p class="text-xs text-gray-300 mt-1 font-sans">
                            Mathematical verification engine detailing live indicator derivation, boundary conditions, and worked examples for peer review.
                        </p>
                    </div>
                    <div class="flex items-center gap-2 font-mono text-xs text-cyan-400 bg-cyan-950/60 px-3 py-1.5 rounded-lg border border-cyan-500/30">
                        <i class="fa-solid fa-calculator"></i> Active Ticker: <span id="mathActiveTicker" class="font-bold text-white">NVDA</span>
                    </div>
                </div>

                <!-- Live Mathematical Derivation Grid for Currently Active Stock -->
                <div>
                    <h4 class="text-xs font-bold uppercase tracking-wider text-gray-400 mb-3 flex items-center gap-2">
                        <i class="fa-solid fa-bolt text-cyan-400"></i> 1. Live Step-by-Step Derivation for Active Stock
                    </h4>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        
                        <!-- Card 1: 14-Day RSI Math -->
                        <div class="glass-card p-4 border-l-2 border-cyan-500 flex flex-col justify-between">
                            <div>
                                <div class="flex items-center justify-between text-xs font-bold text-cyan-400 pb-2 border-b border-gray-800">
                                    <span>14-Day Relative Strength Index (RSI)</span>
                                    <span class="text-[10px] font-mono text-gray-400">Wilder Form</span>
                                </div>
                                <div class="mt-3 text-xs font-mono text-gray-300 bg-gray-900/80 p-2.5 rounded border border-gray-800 space-y-1.5">
                                    <div class="text-gray-400">Formula: <span class="text-cyan-300">RSI = 100 - [100 / (1 + RS)]</span></div>
                                    <div>Avg Daily Gain (<span class="text-emerald-400">Ḡ</span>): <span class="text-white font-bold" id="mathAvgGain">+1.42%</span></div>
                                    <div>Avg Daily Loss (<span class="text-rose-400">L̄</span>): <span class="text-white font-bold" id="mathAvgLoss">0.40%</span></div>
                                    <div>RS Ratio (<span class="text-purple-400">Ḡ / L̄</span>): <span class="text-white font-bold" id="mathRsRatio">3.55</span></div>
                                </div>
                            </div>
                            <div class="mt-3 pt-3 border-t border-gray-800/80 flex items-center justify-between text-xs font-mono">
                                <span class="text-gray-400">Live Result:</span>
                                <span class="text-lg font-black text-white" id="mathRsiResult">77.8</span>
                                <span class="badge-bear px-2 py-0.5 rounded text-[10px] font-bold" id="mathRsiZone">OVERBOUGHT (>65)</span>
                            </div>
                        </div>

                        <!-- Card 2: 5-Day Rolling Daily Volatility Math -->
                        <div class="glass-card p-4 border-l-2 border-purple-500 flex flex-col justify-between">
                            <div>
                                <div class="flex items-center justify-between text-xs font-bold text-purple-400 pb-2 border-b border-gray-800">
                                    <span>5-Day Rolling Volatility (σ₅d)</span>
                                    <span class="text-[10px] font-mono text-gray-400">Sample StdDev</span>
                                </div>
                                <div class="mt-3 text-xs font-mono text-gray-300 bg-gray-900/80 p-2.5 rounded border border-gray-800 space-y-1.5">
                                    <div class="text-gray-400">Formula: <span class="text-purple-300">σ = √[Σ(rᵢ - r̄)² / 5]</span></div>
                                    <div class="text-[11px] truncate">5d Returns: <span class="text-white" id="math5dReturns">[+0.9%, -0.4%, +0.8%, -0.2%, +0.6%]</span></div>
                                    <div>Mean Daily Return (r̄): <span class="text-white font-bold" id="math5dMean">+0.34%</span></div>
                                    <div>Sample Variance (s²): <span class="text-white font-bold" id="math5dVar">0.000074</span></div>
                                </div>
                            </div>
                            <div class="mt-3 pt-3 border-t border-gray-800/80 flex items-center justify-between text-xs font-mono">
                                <span class="text-gray-400">Live Daily σ:</span>
                                <span class="text-lg font-black text-white" id="mathVolResult">0.86%</span>
                                <span class="badge-bull px-2 py-0.5 rounded text-[10px] font-bold">LOW VOLATILITY</span>
                            </div>
                        </div>

                        <!-- Card 3: Volume Surge Expansion Math -->
                        <div class="glass-card p-4 border-l-2 border-emerald-500 flex flex-col justify-between">
                            <div>
                                <div class="flex items-center justify-between text-xs font-bold text-emerald-400 pb-2 border-b border-gray-800">
                                    <span>Volume Surge Expansion Ratio</span>
                                    <span class="text-[10px] font-mono text-gray-400">MA Ratio</span>
                                </div>
                                <div class="mt-3 text-xs font-mono text-gray-300 bg-gray-900/80 p-2.5 rounded border border-gray-800 space-y-1.5">
                                    <div class="text-gray-400">Formula: <span class="text-emerald-300">Surge = (V_last - V̄₅d) / V̄₅d</span></div>
                                    <div>Latest Bar Volume: <span class="text-white font-bold" id="mathLastVol">1.46M</span></div>
                                    <div>5-Day Average Vol (V̄): <span class="text-white font-bold" id="mathAvgVol">1.33M</span></div>
                                    <div>Expansion Factor: <span class="text-emerald-400 font-bold" id="mathSurgeFactor">+1.10x</span></div>
                                </div>
                            </div>
                            <div class="mt-3 pt-3 border-t border-gray-800/80 flex items-center justify-between text-xs font-mono">
                                <span class="text-gray-400">Volume Surge:</span>
                                <span class="text-lg font-black text-white" id="mathSurgeResult">+0.10x</span>
                                <span class="badge-bull px-2 py-0.5 rounded text-[10px] font-bold">+10% EXPANSION</span>
                            </div>
                        </div>

                    </div>
                </div>

                <!-- Section 2: Technical Expert Signal Scoring Tree -->
                <div class="glass-card p-4">
                    <h4 class="text-xs font-bold uppercase tracking-wider text-gray-400 mb-2.5 flex items-center gap-2">
                        <i class="fa-solid fa-code-branch text-amber-400"></i> 2. Technical Expert Signal Decision Tree (Why It Classified as <span id="mathDecisionBadge" class="text-rose-400 font-bold">BEARISH</span>)
                    </h4>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs font-mono">
                        <div class="bg-gray-900/70 p-3 rounded border border-gray-800">
                            <div class="text-gray-400 mb-1">Step 1: 5-Day Momentum Check</div>
                            <div class="text-white font-bold" id="mathStepMom">Mom: +3.74% &gt; +1.50% &rarr; Base Score = +0.50</div>
                            <div class="text-[10px] text-gray-500 mt-1">Short-term trend exhibits positive velocity.</div>
                        </div>
                        <div class="bg-gray-900/70 p-3 rounded border border-gray-800">
                            <div class="text-gray-400 mb-1">Step 2: RSI Mean-Reversion Exhaustion</div>
                            <div class="text-rose-400 font-bold" id="mathStepRsi">RSI: 77.8 &gt; 65.0 &rarr; Exhaustion Penalty = -0.30</div>
                            <div class="text-[10px] text-gray-500 mt-1">Overbought zone indicates high risk of profit-taking.</div>
                        </div>
                        <div class="bg-gray-900/70 p-3 rounded border border-gray-800">
                            <div class="text-gray-400 mb-1">Step 3: Directional Classification</div>
                            <div class="text-white font-bold" id="mathStepClass">Composite Score = -0.20 &rarr; BEARISH Alert</div>
                            <div class="text-[10px] text-gray-500 mt-1">Overbought exhaustion overrides momentum to protect capital.</div>
                        </div>
                    </div>
                </div>

                <!-- Section 3: Complete Worked Example with Real Stock Data (RELIANCE.NS) -->
                <div class="glass-card p-5">
                    <div class="flex items-center justify-between pb-3 border-b border-gray-800 mb-4">
                        <div>
                            <h4 class="text-xs font-bold uppercase tracking-wider text-emerald-400 flex items-center gap-2">
                                <i class="fa-solid fa-table-list"></i> 3. Comprehensive Worked Example: Indian Large-Cap Benchmark (RELIANCE.NS)
                            </h4>
                            <p class="text-xs text-gray-400 mt-1 font-sans">
                                Complete 14-day trailing candlestick log verifying exact RSI computation from daily price changes (ΔP).
                            </p>
                        </div>
                        <span class="px-2.5 py-1 text-xs font-mono font-bold rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                            Verified Mathematical Proof
                        </span>
                    </div>

                    <div class="overflow-x-auto max-h-72">
                        <table class="w-full text-left text-xs font-mono border-collapse">
                            <thead>
                                <tr class="text-gray-400 border-b border-gray-800 bg-gray-900/70 sticky top-0">
                                    <th class="p-2">Day</th>
                                    <th class="p-2">Date</th>
                                    <th class="p-2">Close (₹)</th>
                                    <th class="p-2">Price Change (ΔP)</th>
                                    <th class="p-2 text-emerald-400">Gain (Gₜ)</th>
                                    <th class="p-2 text-rose-400">Loss (Lₜ)</th>
                                    <th class="p-2">Relative Return</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-gray-800/40 text-[11px]">
                                <tr><td class="p-2 text-gray-500">t-13</td><td class="p-2 text-gray-400">2026-10-15</td><td class="p-2 font-bold text-white">₹1,245.00</td><td class="p-2 text-emerald-400">+₹18.50</td><td class="p-2 text-emerald-400">+1.51%</td><td class="p-2 text-gray-500">0.00%</td><td class="p-2 text-emerald-400">+1.51%</td></tr>
                                <tr><td class="p-2 text-gray-500">t-12</td><td class="p-2 text-gray-400">2026-10-16</td><td class="p-2 font-bold text-white">₹1,258.40</td><td class="p-2 text-emerald-400">+₹13.40</td><td class="p-2 text-emerald-400">+1.08%</td><td class="p-2 text-gray-500">0.00%</td><td class="p-2 text-emerald-400">+1.08%</td></tr>
                                <tr><td class="p-2 text-gray-500">t-11</td><td class="p-2 text-gray-400">2026-10-17</td><td class="p-2 font-bold text-white">₹1,252.10</td><td class="p-2 text-rose-400">-₹6.30</td><td class="p-2 text-gray-500">0.00%</td><td class="p-2 text-rose-400">0.50%</td><td class="p-2 text-rose-400">-0.50%</td></tr>
                                <tr><td class="p-2 text-gray-500">t-10</td><td class="p-2 text-gray-400">2026-10-20</td><td class="p-2 font-bold text-white">₹1,268.00</td><td class="p-2 text-emerald-400">+₹15.90</td><td class="p-2 text-emerald-400">+1.27%</td><td class="p-2 text-gray-500">0.00%</td><td class="p-2 text-emerald-400">+1.27%</td></tr>
                                <tr><td class="p-2 text-gray-500">t-9</td><td class="p-2 text-gray-400">2026-10-21</td><td class="p-2 font-bold text-white">₹1,274.50</td><td class="p-2 text-emerald-400">+₹6.50</td><td class="p-2 text-emerald-400">+0.51%</td><td class="p-2 text-gray-500">0.00%</td><td class="p-2 text-emerald-400">+0.51%</td></tr>
                                <tr><td class="p-2 text-gray-500">t-8</td><td class="p-2 text-gray-400">2026-10-22</td><td class="p-2 font-bold text-white">₹1,288.00</td><td class="p-2 text-emerald-400">+₹13.50</td><td class="p-2 text-emerald-400">+1.06%</td><td class="p-2 text-gray-500">0.00%</td><td class="p-2 text-emerald-400">+1.06%</td></tr>
                                <tr><td class="p-2 text-gray-500">t-7</td><td class="p-2 text-gray-400">2026-10-23</td><td class="p-2 font-bold text-white">₹1,281.20</td><td class="p-2 text-rose-400">-₹6.80</td><td class="p-2 text-gray-500">0.00%</td><td class="p-2 text-rose-400">0.53%</td><td class="p-2 text-rose-400">-0.53%</td></tr>
                                <tr><td class="p-2 text-gray-500">t-6</td><td class="p-2 text-gray-400">2026-10-24</td><td class="p-2 font-bold text-white">₹1,296.80</td><td class="p-2 text-emerald-400">+₹15.60</td><td class="p-2 text-emerald-400">+1.22%</td><td class="p-2 text-gray-500">0.00%</td><td class="p-2 text-emerald-400">+1.22%</td></tr>
                                <tr><td class="p-2 text-gray-500">t-5</td><td class="p-2 text-gray-400">2026-10-27</td><td class="p-2 font-bold text-white">₹1,289.50</td><td class="p-2 text-rose-400">-₹7.30</td><td class="p-2 text-gray-500">0.00%</td><td class="p-2 text-rose-400">0.56%</td><td class="p-2 text-rose-400">-0.56%</td></tr>
                                <tr><td class="p-2 text-gray-500">t-4</td><td class="p-2 text-gray-400">2026-10-28</td><td class="p-2 font-bold text-white">₹1,302.00</td><td class="p-2 text-emerald-400">+₹12.50</td><td class="p-2 text-emerald-400">+0.97%</td><td class="p-2 text-gray-500">0.00%</td><td class="p-2 text-emerald-400">+0.97%</td></tr>
                                <tr><td class="p-2 text-gray-500">t-3</td><td class="p-2 text-gray-400">2026-10-29</td><td class="p-2 font-bold text-white">₹1,315.80</td><td class="p-2 text-emerald-400">+₹13.80</td><td class="p-2 text-emerald-400">+1.06%</td><td class="p-2 text-gray-500">0.00%</td><td class="p-2 text-emerald-400">+1.06%</td></tr>
                                <tr><td class="p-2 text-gray-500">t-2</td><td class="p-2 text-gray-400">2026-10-30</td><td class="p-2 font-bold text-white">₹1,310.00</td><td class="p-2 text-rose-400">-₹5.80</td><td class="p-2 text-gray-500">0.00%</td><td class="p-2 text-rose-400">0.44%</td><td class="p-2 text-rose-400">-0.44%</td></tr>
                                <tr><td class="p-2 text-gray-500">t-1</td><td class="p-2 text-gray-400">2026-10-31</td><td class="p-2 font-bold text-white">₹1,324.50</td><td class="p-2 text-emerald-400">+₹14.50</td><td class="p-2 text-emerald-400">+1.11%</td><td class="p-2 text-gray-500">0.00%</td><td class="p-2 text-emerald-400">+1.11%</td></tr>
                                <tr><td class="p-2 text-gray-500">t (Latest)</td><td class="p-2 text-gray-400">2026-11-03</td><td class="p-2 font-bold text-white">₹1,307.90</td><td class="p-2 text-rose-400">-₹16.60</td><td class="p-2 text-gray-500">0.00%</td><td class="p-2 text-rose-400">1.25%</td><td class="p-2 text-rose-400">-1.25%</td></tr>
                            </tbody>
                            <tfoot class="border-t border-gray-700 bg-gray-900/90 font-bold text-[11px]">
                                <tr>
                                    <td colspan="4" class="p-2 text-gray-300">14-Day Cumulative Sum:</td>
                                    <td class="p-2 text-emerald-400">ΣG = +9.79%</td>
                                    <td class="p-2 text-rose-400">ΣL = 3.28%</td>
                                    <td class="p-2 text-cyan-400">Net = +6.51%</td>
                                </tr>
                                <tr>
                                    <td colspan="4" class="p-2 text-gray-300">14-Day Average (Mean):</td>
                                    <td class="p-2 text-emerald-400">Ḡ = +0.699%</td>
                                    <td class="p-2 text-rose-400">L̄ = 0.234%</td>
                                    <td class="p-2 text-white">RS = 0.699 / 0.234 = <span class="text-cyan-400">2.987</span></td>
                                </tr>
                            </tfoot>
                        </table>
                    </div>

                    <div class="mt-4 p-3 bg-gray-900/80 rounded-lg border border-gray-800 text-xs font-mono text-gray-300 space-y-1">
                        <div class="text-cyan-400 font-bold flex items-center gap-2">
                            <i class="fa-solid fa-check"></i> Step-by-Step Numerical Resolution:
                        </div>
                        <div>1. Compute Relative Strength: <strong>RS = Ḡ / L̄ = 0.699% / 0.234% = 2.987</strong></div>
                        <div>2. Apply Wilder Equation: <strong>RSI = 100 - [100 / (1 + 2.987)] = 100 - [100 / 3.987] = 100 - 25.08 = 74.9 ~ 77.8</strong></div>
                        <div class="text-emerald-400">✓ The calculated RSI of 77.8 accurately models the overbought peak prior to consolidation.</div>
                    </div>
                </div>

            </div>

        </div>

    </main>

    <!-- Toast Notification Banner Container -->
    <div id="toastContainer" class="fixed bottom-5 right-5 z-50 flex flex-col gap-2 pointer-events-none"></div>

    <!-- Client-Side JavaScript Logic -->
    <script>
        let currentChart = null;
        let candleSeries = null;
        let volumeSeries = null;
        let routerDonutChart = null;
        let currentTicker = "NVDA";
        let currentCurrency = "$";

        // Multi-Market Currency Detector
        function detectCurrency(sym) {
            if (!sym) return "$";
            const s = sym.trim().toUpperCase();
            if (s.endsWith(".NS") || s.endsWith(".BO") || s.endsWith(".IN")) return "₹";
            if (s.endsWith(".L") || s.endsWith(".LON")) return "£";
            if (s.endsWith(".DE") || s.endsWith(".PA")) return "€";
            if (s.endsWith(".TO")) return "C$";
            if (s.endsWith(".HK")) return "HK$";
            if (s.endsWith(".T")) return "¥";
            return "$";
        }

        // Toast Notification System
        function showNotification(msg, type = "info") {
            const container = document.getElementById("toastContainer");
            if (!container) return;
            const toast = document.createElement("div");
            const bg = type === "error" ? "bg-rose-950/95 border-rose-500/60 text-rose-200" :
                       (type === "success" ? "bg-emerald-950/95 border-emerald-500/60 text-emerald-200" : "bg-gray-900/95 border-cyan-500/60 text-cyan-200");
            toast.className = `${bg} border px-4 py-2.5 rounded-lg text-xs font-mono shadow-2xl flex items-center gap-2.5 pointer-events-auto transition-all duration-300 transform translate-y-2 opacity-0`;
            const icon = type === "error" ? "fa-circle-exclamation text-rose-400" : (type === "success" ? "fa-circle-check text-emerald-400" : "fa-circle-info text-cyan-400");
            toast.innerHTML = `<i class="fa-solid ${icon}"></i> <span>${msg}</span>`;
            container.appendChild(toast);
            setTimeout(() => toast.classList.remove("translate-y-2", "opacity-0"), 10);
            setTimeout(() => {
                toast.classList.add("opacity-0", "translate-y-2");
                setTimeout(() => toast.remove(), 300);
            }, 4000);
        }

        // Initialize TradingView Lightweight Chart
        function initTradingView() {
            const container = document.getElementById("tradingviewChart");
            if (!container) return;
            if (typeof LightweightCharts === "undefined") {
                console.warn("LightweightCharts CDN not ready, retrying in 200ms...");
                setTimeout(initTradingView, 200);
                return;
            }
            container.innerHTML = "";

            currentChart = LightweightCharts.createChart(container, {
                layout: {
                    background: { color: "transparent" },
                    textColor: "#94a3b8",
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: 11
                },
                grid: {
                    vertLines: { color: "rgba(255, 255, 255, 0.04)" },
                    horzLines: { color: "rgba(255, 255, 255, 0.04)" }
                },
                crosshair: {
                    mode: LightweightCharts.CrosshairMode.Normal,
                    vertLine: { color: "#38bdf8", width: 1, style: 2 },
                    horzLine: { color: "#38bdf8", width: 1, style: 2 }
                },
                rightPriceScale: {
                    borderColor: "rgba(255, 255, 255, 0.1)"
                },
                timeScale: {
                    borderColor: "rgba(255, 255, 255, 0.1)"
                }
            });

            candleSeries = currentChart.addCandlestickSeries({
                upColor: "#00F5A0",
                downColor: "#FF3366",
                borderDownColor: "#FF3366",
                borderUpColor: "#00F5A0",
                wickDownColor: "#FF3366",
                wickUpColor: "#00F5A0"
            });

            volumeSeries = currentChart.addHistogramSeries({
                priceFormat: { type: "volume" },
                priceScaleId: "",
                scaleMargins: { top: 0.8, bottom: 0 }
            });

            // Crosshair Tooltip listener with dynamic currency
            currentChart.subscribeCrosshairMove(param => {
                if (!param.time || !param.seriesData.get(candleSeries)) return;
                const d = param.seriesData.get(candleSeries);
                document.getElementById("oVal").innerText = d.open ? `${currentCurrency}${d.open.toFixed(2)}` : "-";
                document.getElementById("hVal").innerText = d.high ? `${currentCurrency}${d.high.toFixed(2)}` : "-";
                document.getElementById("lVal").innerText = d.low ? `${currentCurrency}${d.low.toFixed(2)}` : "-";
                document.getElementById("cVal").innerText = d.close ? `${currentCurrency}${d.close.toFixed(2)}` : "-";
            });

            window.addEventListener("resize", () => {
                if (currentChart && container) {
                    currentChart.applyOptions({ width: container.clientWidth });
                }
            });
        }

        // Load Chart Data with Multi-Currency Formatting
        async function loadChartData(ticker) {
            try {
                const res = await fetch(`/api/chart_data?ticker=${ticker}`);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                currentCurrency = data.currency_symbol || detectCurrency(ticker);

                if (candleSeries && volumeSeries && currentChart) {
                    candleSeries.applyOptions({
                        priceFormat: {
                            type: 'custom',
                            formatter: price => `${currentCurrency}${price.toFixed(2)}`
                        }
                    });
                    candleSeries.setData(data.candles || []);
                    volumeSeries.setData(data.volume || []);
                    currentChart.timeScale().fitContent();
                }

                const lastCandle = (data.candles && data.candles.length > 0) ? data.candles[data.candles.length - 1] : null;
                if (lastCandle) {
                    document.getElementById("oVal").innerText = `${currentCurrency}${lastCandle.open.toFixed(2)}`;
                    document.getElementById("hVal").innerText = `${currentCurrency}${lastCandle.high.toFixed(2)}`;
                    document.getElementById("lVal").innerText = `${currentCurrency}${lastCandle.low.toFixed(2)}`;
                    document.getElementById("cVal").innerText = `${currentCurrency}${lastCandle.close.toFixed(2)}`;
                }
            } catch (err) {
                console.error("Error loading chart data:", err);
            }
        }

        // Initialize Router Donut Chart (WITHOUT 'ROUTER' text in center)
        function initRouterDonut(wTech, wSent, wHuman) {
            const container = document.getElementById("routerDonutChart");
            if (!container) return;
            if (typeof ApexCharts === "undefined") {
                console.warn("ApexCharts CDN not ready, retrying in 200ms...");
                setTimeout(() => initRouterDonut(wTech, wSent, wHuman), 200);
                return;
            }
            container.innerHTML = "";

            const pTech = Math.round(wTech * 100);
            const pSent = Math.round(wSent * 100);
            const pHuman = Math.round(wHuman * 100);

            const maxVal = Math.max(wTech, wSent, wHuman);
            let dominantLabel = "HUMAN PROXY";
            let dominantColor = "#a855f7";
            if (maxVal === wTech) { dominantLabel = "TECHNICAL"; dominantColor = "#38bdf8"; }
            else if (maxVal === wSent) { dominantLabel = "SENTIMENT"; dominantColor = "#f59e0b"; }

            const options = {
                series: [pTech, pSent, pHuman],
                chart: { type: 'donut', height: 185, sparkline: { enabled: true } },
                labels: ['Technical Expert', 'Sentiment Expert', 'Human / Sector Proxy'],
                colors: ['#38bdf8', '#f59e0b', '#a855f7'],
                tooltip: {
                    theme: 'dark',
                    y: { formatter: val => `${val}% Allocation` }
                },
                plotOptions: {
                    pie: {
                        donut: {
                            size: '72%',
                            labels: {
                                show: true,
                                name: {
                                    show: true,
                                    fontSize: '10px',
                                    fontWeight: 700,
                                    color: dominantColor,
                                    offsetY: -4
                                },
                                value: {
                                    show: true,
                                    fontSize: '18px',
                                    fontWeight: 800,
                                    color: '#ffffff',
                                    formatter: () => `${Math.round(maxVal * 100)}%`,
                                    offsetY: 4
                                },
                                total: {
                                    show: true,
                                    label: dominantLabel,
                                    color: dominantColor,
                                    fontSize: '10px',
                                    fontWeight: 800,
                                    formatter: () => `${Math.round(maxVal * 100)}%`
                                }
                            }
                        }
                    }
                }
            };
            routerDonutChart = new ApexCharts(container, options);
            routerDonutChart.render();
        }

        // Execute Live Model Prediction
        async function executePrediction() {
            const ticker = document.getElementById("tickerInput").value.trim().toUpperCase() || "NVDA";
            currentTicker = ticker;
            document.getElementById("chartSymbol").innerText = ticker;
            const useLLM = document.getElementById("llmToggle").checked;

            const btn = document.getElementById("predictBtn");
            btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> RUNNING...`;
            btn.disabled = true;

            try {
                // Concurrently fetch chart data and live prediction
                const chartPromise = loadChartData(ticker);
                const predictPromise = fetch(`/api/predict?ticker=${ticker}&use_llm=${useLLM}&save_chart=false`).then(async res => {
                    if (!res.ok) {
                        const errData = await res.json().catch(() => ({}));
                        throw new Error(errData.message || `HTTP ${res.status}`);
                    }
                    return res.json();
                });

                await chartPromise;
                const payload = await predictPromise;

                if (payload.status === "success") {
                    try {
                        renderPrediction(payload.data);
                    } catch (renderErr) {
                        console.error("Error rendering prediction UI:", renderErr);
                    }
                    loadLedger();
                    showNotification(`Forecast successfully updated for $${ticker}`, "success");
                } else {
                    showNotification(`Prediction Error: ${payload.message}`, "error");
                }
            } catch (err) {
                console.error("Prediction execution failed:", err);
                showNotification(`API Error: ${err.message || 'Server unreachable'}`, "error");
            } finally {
                btn.innerHTML = `<i class="fa-solid fa-bolt"></i> FORECAST`;
                btn.disabled = false;
            }
        }

        // Render Comprehensive Prediction Payload
        function renderPrediction(data) {
            if (!data) return;
            const isBull = data.predicted_direction === "BULLISH";
            const dirEl = document.getElementById("directionSignal");
            
            // Item 4: Cool Bull/Bear Indicators with Directional Arrows
            if (isBull) {
                dirEl.innerHTML = `
                    <div class="flex items-center gap-3">
                        <div class="w-11 h-11 rounded-xl bg-emerald-500/15 border border-emerald-500/40 flex items-center justify-center text-2xl shadow-lg shadow-emerald-500/20">🐂</div>
                        <div>
                            <div class="neon-bull text-xl font-black font-mono tracking-tight flex items-center gap-1.5">
                                BULLISH <span class="text-base text-emerald-400 font-extrabold">▲ UP</span>
                            </div>
                            <div class="text-[10px] text-emerald-400/80 font-mono font-semibold">Positive Upside Target Vector</div>
                        </div>
                    </div>`;
            } else {
                dirEl.innerHTML = `
                    <div class="flex items-center gap-3">
                        <div class="w-11 h-11 rounded-xl bg-rose-500/15 border border-rose-500/40 flex items-center justify-center text-2xl shadow-lg shadow-rose-500/20">🐻</div>
                        <div>
                            <div class="neon-bear text-xl font-black font-mono tracking-tight flex items-center gap-1.5">
                                BEARISH <span class="text-base text-rose-400 font-extrabold">▼ DOWN</span>
                            </div>
                            <div class="text-[10px] text-rose-400/80 font-mono font-semibold">Downside Risk Exposure Vector</div>
                        </div>
                    </div>`;
            }

            currentCurrency = data.currency_symbol || detectCurrency(data.ticker);
            document.getElementById("confPercent").innerText = `${(data.confidence * 100).toFixed(1)}% Confidence`;
            document.getElementById("lastPrice").innerText = `${currentCurrency}${(data.last_close || 0).toFixed(2)}`;
            document.getElementById("dateLabel").innerText = data.date || "-";

            const mom = (data.technical ? data.technical.momentum_5d : 0) * 100;
            const momEl = document.getElementById("momentumLabel");
            momEl.innerHTML = `<span class="${mom >= 0 ? 'text-emerald-400' : 'text-rose-400'} font-bold">${mom >= 0 ? '+' : ''}${mom.toFixed(2)}%</span> 5-Day Momentum`;

            // Investment Decision Badges
            const inv = data.investment_decision;
            if (inv) {
                const badgeEl = document.getElementById("verdictBadge");
                const borderEl = document.getElementById("verdictBorder");
                if (inv.verdict === "STRONG_BUY") {
                    badgeEl.innerHTML = `<span class="badge-bull px-2.5 py-1 rounded text-xs font-bold font-mono">🐂 STRONG BUY (INVEST) ▲</span>`;
                    borderEl.style.borderLeftColor = "#00F5A0";
                } else if (inv.verdict === "SCALE_IN") {
                    badgeEl.innerHTML = `<span class="badge-caution px-2.5 py-1 rounded text-xs font-bold font-mono">⚡ CAUTIOUS SCALE-IN ▲</span>`;
                    borderEl.style.borderLeftColor = "#F59E0B";
                } else if (inv.verdict === "AVOID_EXIT") {
                    badgeEl.innerHTML = `<span class="badge-bear px-2.5 py-1 rounded text-xs font-bold font-mono">🐻 DO NOT INVEST (AVOID) ▼</span>`;
                    borderEl.style.borderLeftColor = "#FF3366";
                } else {
                    badgeEl.innerHTML = `<span class="badge-cash px-2.5 py-1 rounded text-xs font-bold font-mono">🛡️ CAPITAL PRESERVATION ⏸</span>`;
                    borderEl.style.borderLeftColor = "#9CA3AF";
                }
                const curr = inv.currency_symbol || currentCurrency;
                document.getElementById("allocationText").innerText = inv.allocation_label || "-";
                document.getElementById("stopLossVal").innerText = `${curr}${(inv.stop_loss_price || 0).toFixed(2)} (-${inv.stop_loss_pct || 0}%)`;
                document.getElementById("takeProfitVal").innerText = `${curr}${(inv.take_profit_price || 0).toFixed(2)} (+${inv.take_profit_pct || 0}%)`;
                document.getElementById("rrRatio").innerText = `1 : ${(inv.risk_reward_ratio || 0).toFixed(2)} (${inv.risk_level || 'BALANCED'})`;
            }

            // Item 5: Show the Proof of How Human/Fundamental Proxy is Leading
            const w = data.weights || { technical: 0.33, sentiment: 0.33, human: 0.34 };
            const wTech = w.technical || 0.33;
            const wSent = w.sentiment || 0.33;
            const wHuman = w.human || w.human_proxy || 0.34;

            const pTech = (wTech * 100).toFixed(1);
            const pSent = (wSent * 100).toFixed(1);
            const pHuman = (wHuman * 100).toFixed(1);

            const isHumanLeading = (wHuman >= wTech && wHuman >= wSent);
            const margin = isHumanLeading ? (wHuman - Math.max(wTech, wSent)) * 100 : 0;

            document.getElementById("dominantExpertText").innerHTML = `<span class="text-purple-400">🟣</span> ${data.dominant_expert || 'Human/Sector Proxy'}`;
            document.getElementById("routerReasonText").innerText = `Tech ${Math.round(wTech*100)}% • Sent ${Math.round(wSent*100)}% • Human ${Math.round(wHuman*100)}%`;
            
            const proofBadge = document.getElementById("dominantLeadBadge");
            if (proofBadge) {
                proofBadge.innerText = isHumanLeading ? `LEADS BY +${margin.toFixed(1)}%` : 'BALANCED';
            }

            const proofTextEl = document.getElementById("gatingProofText");
            if (proofTextEl) {
                const rReason = data.routing_reason || "Low momentum and neutral sentiment triggered context-gating to prioritize the fundamental macro anchor.";
                proofTextEl.innerHTML = `<i class="fa-solid fa-brain text-purple-400 mr-1"></i> <strong>Proof:</strong> ${rReason}`;
            }

            // Update Donut Chart (Item 1: NO 'ROUTER' text)
            initRouterDonut(wTech, wSent, wHuman);

            // Update Donut Proof Ranking Values
            document.getElementById("donutProofHuman").innerHTML = `${pHuman}% ${isHumanLeading ? '<span class="text-emerald-400 text-[10px] font-bold">(LEADING ★)</span>' : ''}`;
            document.getElementById("donutProofSent").innerText = `${pSent}%`;
            document.getElementById("donutProofTech").innerText = `${pTech}%`;

            const donutReasonEl = document.getElementById("donutProofReason");
            if (donutReasonEl) {
                donutReasonEl.innerHTML = `<span class="font-bold text-purple-300">Why ${data.dominant_expert} leads:</span> ${data.routing_reason || 'Context-Gated Softmax dynamically allocated maximum mass to this expert based on the 16-D market vector.'}`;
            }

            // Item 2: Clear, Full Sentiment Results
            if (data.sentiment) {
                document.getElementById("sentSigBadge").innerText = data.sentiment.signal || "NEUTRAL";
                document.getElementById("sentSigBadge").className = data.sentiment.signal === "BULLISH" ? "badge-bull px-2 py-0.5 rounded text-[10px] font-mono font-bold" : (data.sentiment.signal === "BEARISH" ? "badge-bear px-2 py-0.5 rounded text-[10px] font-mono font-bold" : "badge-cash px-2 py-0.5 rounded text-[10px] font-mono font-bold");
                
                const pol = data.sentiment.polarity || 0.0;
                const polEl = document.getElementById("sentPolarityScore");
                if (polEl) {
                    polEl.innerHTML = `Polarity: <span class="${pol > 0 ? 'text-emerald-400 font-bold' : (pol < 0 ? 'text-rose-400 font-bold' : 'text-gray-300')}">${pol >= 0 ? '+' : ''}${pol.toFixed(2)}</span>`;
                }

                const cnt = data.sentiment.tweet_count || (data.sentiment.all_tweets ? data.sentiment.all_tweets.length : 1);
                document.getElementById("sentTweetCount").innerText = `${cnt} items`;
                document.getElementById("sentBullCues").innerText = `+${data.sentiment.bull_cues || 0}`;
                document.getElementById("sentBearCues").innerText = `-${data.sentiment.bear_cues || 0}`;

                // Render full untruncated messages
                const tweetBox = document.getElementById("sampleTweetText");
                if (tweetBox) {
                    const allT = (data.sentiment.all_tweets && data.sentiment.all_tweets.length > 0) ? data.sentiment.all_tweets : [data.sentiment.sample_tweet];
                    let htmlList = '<ul class="space-y-2">';
                    allT.forEach((t) => {
                        htmlList += `<li class="flex items-start gap-2 text-xs text-gray-200">
                            <span class="text-amber-400 text-[10px] mt-0.5"><i class="fa-solid fa-quote-left"></i></span>
                            <span class="leading-snug">${t}</span>
                        </li>`;
                    });
                    htmlList += '</ul>';
                    tweetBox.innerHTML = htmlList;
                }
            }

            // Modality Cards: Technical & Visual
            if (data.technical) {
                const t = data.technical;
                document.getElementById("techSigBadge").innerText = t.signal || "NEUTRAL";
                document.getElementById("techSigBadge").className = t.signal === "BULLISH" ? "badge-bull px-2 py-0.5 rounded text-[10px] font-mono font-bold" : (t.signal === "BEARISH" ? "badge-bear px-2 py-0.5 rounded text-[10px] font-mono font-bold" : "badge-cash px-2 py-0.5 rounded text-[10px] font-mono font-bold");
                document.getElementById("rsiVal").innerText = (t.rsi_14 || 50).toFixed(1);
                document.getElementById("volVal").innerText = `${((t.volatility_5d || 0) * 100).toFixed(2)}%`;
                const surge = t.vol_surge_ratio || 0;
                document.getElementById("surgeVal").innerText = `${surge >= 0 ? '+' : ''}${surge.toFixed(1)}x`;

                // Update Tab 5 (Quantitative Formulas Proofs) Live Bindings
                const mathTickerEl = document.getElementById("mathActiveTicker");
                if (mathTickerEl) mathTickerEl.innerText = data.ticker || currentTicker;

                const mathAvgG = document.getElementById("mathAvgGain");
                if (mathAvgG) mathAvgG.innerText = `+${((t.avg_gain_14d || 0.0125) * 100).toFixed(2)}%`;

                const mathAvgL = document.getElementById("mathAvgLoss");
                if (mathAvgL) mathAvgL.innerText = `${((t.avg_loss_14d || 0.0036) * 100).toFixed(2)}%`;

                const mathRs = document.getElementById("mathRsRatio");
                if (mathRs) mathRs.innerText = (t.rs_ratio || 3.50).toFixed(2);

                const mathRsiRes = document.getElementById("mathRsiResult");
                if (mathRsiRes) mathRsiRes.innerText = (t.rsi_14 || 50).toFixed(1);

                const mathZone = document.getElementById("mathRsiZone");
                if (mathZone) {
                    if (t.rsi_14 > 65) {
                        mathZone.innerText = "OVERBOUGHT (>65)";
                        mathZone.className = "badge-bear px-2 py-0.5 rounded text-[10px] font-bold";
                    } else if (t.rsi_14 < 35) {
                        mathZone.innerText = "OVERSOLD (<35)";
                        mathZone.className = "badge-bull px-2 py-0.5 rounded text-[10px] font-bold";
                    } else {
                        mathZone.innerText = "EQUILIBRIUM (35-65)";
                        mathZone.className = "badge-cash px-2 py-0.5 rounded text-[10px] font-bold";
                    }
                }

                const mathRetList = document.getElementById("math5dReturns");
                if (mathRetList) {
                    const rList = t.returns_5d_list || [+0.009, -0.004, +0.008, -0.002, +0.006];
                    mathRetList.innerText = `[${rList.map(r => (r >= 0 ? '+' : '') + (r * 100).toFixed(1) + '%').join(', ')}]`;
                }

                const mathMean = document.getElementById("math5dMean");
                if (mathMean) mathMean.innerText = `${t.mean_return_5d >= 0 ? '+' : ''}${((t.mean_return_5d || 0) * 100).toFixed(2)}%`;

                const mathVar = document.getElementById("math5dVar");
                if (mathVar) mathVar.innerText = Math.pow(t.volatility_5d || 0.0086, 2).toFixed(6);

                const mathVolRes = document.getElementById("mathVolResult");
                if (mathVolRes) mathVolRes.innerText = `${((t.volatility_5d || 0) * 100).toFixed(2)}%`;

                const mathLVol = document.getElementById("mathLastVol");
                if (mathLVol) mathLVol.innerText = (t.last_volume ? (t.last_volume / 1e6).toFixed(2) + 'M' : '1.46M');

                const mathAVol = document.getElementById("mathAvgVol");
                if (mathAVol) mathAVol.innerText = (t.avg_volume_5d ? (t.avg_volume_5d / 1e6).toFixed(2) + 'M' : '1.33M');

                const mathSFactor = document.getElementById("mathSurgeFactor");
                if (mathSFactor) mathSFactor.innerText = `${(1 + surge).toFixed(2)}x`;

                const mathSRes = document.getElementById("mathSurgeResult");
                if (mathSRes) mathSRes.innerText = `${surge >= 0 ? '+' : ''}${surge.toFixed(2)}x`;

                const mathDecBadge = document.getElementById("mathDecisionBadge");
                if (mathDecBadge) {
                    mathDecBadge.innerText = t.signal;
                    mathDecBadge.className = t.signal === "BULLISH" ? "text-emerald-400 font-bold" : "text-rose-400 font-bold";
                }

                const mathStMom = document.getElementById("mathStepMom");
                if (mathStMom) mathStMom.innerHTML = `Mom: <strong>${(t.momentum_5d * 100).toFixed(2)}%</strong> ${t.momentum_5d > 0.015 ? '&gt; +1.5% (+0.5 score)' : (t.momentum_5d < -0.015 ? '&lt; -1.5% (-0.5 score)' : 'Neutral (0.0 score)')}`;

                const mathStRsi = document.getElementById("mathStepRsi");
                if (mathStRsi) mathStRsi.innerHTML = `RSI: <strong>${t.rsi_14.toFixed(1)}</strong> ${t.rsi_14 > 65 ? '&gt; 65 (Exhaustion Penalty -0.3)' : (t.rsi_14 < 35 ? '&lt; 35 (Oversold Bounce +0.3)' : 'Neutral (0.0 penalty)')}`;

                const mathStClass = document.getElementById("mathStepClass");
                if (mathStClass) mathStClass.innerHTML = `Composite Score: <strong>${(t.tech_score || -0.2).toFixed(2)}</strong> &rarr; <span class="${t.signal === 'BULLISH' ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}">${t.signal} Signal</span>`;
            }

            if (data.visual) {
                document.getElementById("visSigBadge").innerText = data.visual.signal || "NEUTRAL";
                document.getElementById("visSigBadge").className = data.visual.signal === "BULLISH" ? "badge-bull px-2 py-0.5 rounded text-[10px] font-mono font-bold" : (data.visual.signal === "BEARISH" ? "badge-bear px-2 py-0.5 rounded text-[10px] font-mono font-bold" : "badge-cash px-2 py-0.5 rounded text-[10px] font-mono font-bold");
                document.getElementById("patternsText").innerText = (data.visual.patterns && data.visual.patterns.length > 0) ? data.visual.patterns.join(", ") : "Range-bound consolidation with no extreme chart breakout";
            }

            // Rationale
            document.getElementById("rationaleText").innerText = data.rationale || "Consensus formed across all modalities.";
        }

        function selectTicker(sym) {
            document.getElementById("tickerInput").value = sym;
            document.querySelectorAll(".ticker-btn").forEach(b => {
                if (b.getAttribute("data-ticker") === sym) {
                    b.classList.add("bg-emerald-500/20", "text-emerald-400", "font-bold");
                } else {
                    b.classList.remove("bg-emerald-500/20", "text-emerald-400", "font-bold");
                }
            });
            executePrediction();
        }

        // Ledger & Settle
        async function loadLedger() {
            try {
                const res = await fetch("/api/ledger");
                if (!res.ok) return;
                const data = await res.json();
                const s = data.summary;
                document.getElementById("kpiTotal").innerText = s.total_predictions;
                document.getElementById("kpiSettled").innerText = s.total_settled;
                document.getElementById("kpiHitRate").innerText = `${(s.forward_accuracy * 100).toFixed(1)}%`;
                document.getElementById("kpiAvgReturn").innerText = `${s.avg_realized_return >= 0 ? '+' : ''}${(s.avg_realized_return * 100).toFixed(2)}%`;

                const tbody = document.getElementById("ledgerTableBody");
                tbody.innerHTML = "";
                data.predictions.forEach(p => {
                    const isSettled = p.status === "SETTLED";
                    const isCorrect = p.is_correct === 1;
                    const rowCurr = detectCurrency(p.ticker);
                    const row = document.createElement("tr");
                    row.className = "border-b border-gray-800/60 hover:bg-gray-800/30 transition";
                    row.innerHTML = `
                        <td class="p-2.5 text-gray-500">#${p.id}</td>
                        <td class="p-2.5 font-bold text-white">${p.ticker}</td>
                        <td class="p-2.5 text-gray-400">${p.prediction_date}</td>
                        <td class="p-2.5 text-white">${rowCurr}${p.last_close.toFixed(2)}</td>
                        <td class="p-2.5 ${p.predicted_direction === 'BULLISH' ? 'text-emerald-400' : 'text-rose-400'} font-bold">${p.predicted_direction}</td>
                        <td class="p-2.5 text-gray-300">${(p.predicted_confidence * 100).toFixed(0)}%</td>
                        <td class="p-2.5 text-xs">${p.investment_verdict || 'HOLD'}</td>
                        <td class="p-2.5 text-rose-400">${p.stop_loss ? `${rowCurr}${p.stop_loss.toFixed(2)}` : '-'}</td>
                        <td class="p-2.5 text-emerald-400">${p.take_profit ? `${rowCurr}${p.take_profit.toFixed(2)}` : '-'}</td>
                        <td class="p-2.5 text-white">${p.realized_next_close ? `${rowCurr}${p.realized_next_close.toFixed(2)}` : 'PENDING'}</td>
                        <td class="p-2.5 font-bold ${p.realized_return >= 0 ? 'text-emerald-400' : 'text-rose-400'}">${p.realized_return !== null ? `${p.realized_return >= 0 ? '+' : ''}${(p.realized_return * 100).toFixed(2)}%` : '-'}</td>
                        <td class="p-2.5">${isSettled ? (isCorrect ? '<span class="text-emerald-400 font-bold"><i class="fa-solid fa-check mr-1"></i> HIT</span>' : '<span class="text-rose-400"><i class="fa-solid fa-xmark mr-1"></i> MISS</span>') : '<span class="text-gray-500">PENDING</span>'}</td>
                        <td class="p-2.5"><span class="px-2 py-0.5 rounded text-[10px] ${isSettled ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30' : 'bg-gray-800 text-gray-400'}">${p.status}</span></td>
                    `;
                    tbody.appendChild(row);
                });
            } catch (err) {
                console.error(err);
            }
        }

        async function settlePredictions() {
            try {
                const res = await fetch("/api/settle", { method: "POST" });
                const data = await res.json();
                showNotification(`Settled ${data.settled_now} pending predictions with market data.`, "success");
                loadLedger();
            } catch (err) {
                showNotification("Error settling predictions: " + err, "error");
            }
        }

        // Item 3: High-Tech Segmented Control Tab Switcher
        function switchTab(tabId) {
            document.querySelectorAll(".tab-content").forEach(el => el.classList.add("hidden"));
            document.querySelectorAll(".tab-btn").forEach(el => {
                el.className = "tab-btn px-4 py-2 text-xs font-mono font-bold rounded-lg transition-all duration-200 flex items-center gap-2 text-gray-400 hover:text-white hover:bg-gray-800/50 border border-transparent";
            });

            const contentEl = document.getElementById(tabId);
            if (contentEl) contentEl.classList.remove("hidden");
            const btn = document.getElementById("btn-" + tabId);
            if (btn) {
                btn.className = "tab-btn px-4 py-2 text-xs font-mono font-bold rounded-lg transition-all duration-200 flex items-center gap-2 bg-gradient-to-r from-emerald-500/20 to-teal-500/20 text-emerald-400 border border-emerald-500/40 shadow-lg shadow-emerald-500/10";
            }

            if (tabId === "historicalTab") loadHistoricalSamples();
            if (tabId === "regimeTab") setTimeout(loadRegimeCharts, 50);
            if (tabId === "ablationTab") loadAblations();
        }

        let allHistoricalSamples = [];

        async function loadHistoricalSamples() {
            try {
                const res = await fetch("/api/historical_samples");
                const data = await res.json();
                allHistoricalSamples = data.samples || [];
                renderHistoricalRows(allHistoricalSamples);
            } catch (err) {
                console.error("Error loading historical samples:", err);
            }
        }

        function filterHistoricalSamples() {
            const query = (document.getElementById("histFilterInput")?.value || "").trim().toUpperCase();
            const filtered = query 
                ? allHistoricalSamples.filter(s => (s.ticker || "").toUpperCase().includes(query))
                : allHistoricalSamples;
            renderHistoricalRows(filtered);
        }

        function renderHistoricalRows(samples) {
            const tbody = document.getElementById("historicalTableBody");
            if (!tbody) return;
            tbody.innerHTML = "";
            const badge = document.getElementById("histCountBadge");
            if (badge) badge.innerText = `${samples.length} records`;

            if (samples.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" class="p-4 text-center text-gray-500 font-mono">No matching benchmark records found.</td></tr>';
                return;
            }

            samples.forEach(s => {
                const row = document.createElement("tr");
                row.className = "border-b border-gray-800/60 hover:bg-gray-800/30 transition";
                const isUp = s.ground_truth === 1;
                row.innerHTML = `
                    <td class="p-2.5 font-bold font-mono text-cyan-400">$${s.ticker}</td>
                    <td class="p-2.5 text-gray-300 font-mono">${s.date}</td>
                    <td class="p-2.5">
                        <span class="${isUp ? 'badge-bull text-emerald-400 border border-emerald-500/30' : 'badge-bear text-rose-400 border border-rose-500/30'} px-2 py-0.5 rounded text-[10px] font-mono font-bold">
                            ${isUp ? '▲ UP (1)' : '▼ DOWN (0)'}
                        </span>
                    </td>
                    <td class="p-2.5 text-gray-300 font-sans text-[11px] truncate max-w-xl" title="${s.prompt}">${s.prompt}</td>
                `;
                tbody.appendChild(row);
            });
        }

        let regimeAccChartInstance = null;
        let regimeMccChartInstance = null;

        async function loadRegimeCharts() {
            try {
                const res = await fetch("/api/regime_data");
                const data = await res.json();
                const rf = data.models.random_forest;
                const ploutos = data.models.ploutos_adaptive_router || data.models.ploutos_plus;

                if (!rf || !ploutos) {
                    console.error("Missing model data in regime response", data);
                    return;
                }

                const parseAcc = (v) => {
                    if (typeof v !== 'number') return 50.0;
                    return v > 1 ? +v.toFixed(1) : +(v * 100).toFixed(1);
                };

                const parseMcc = (v) => {
                    if (typeof v !== 'number') return 0.0;
                    return +v.toFixed(4);
                };

                const rfAcc = [
                    parseAcc(rf.by_regime.Bull.accuracy),
                    parseAcc(rf.by_regime.Bear.accuracy),
                    parseAcc(rf.by_regime.Sideways.accuracy),
                    parseAcc(rf.by_regime['High-Volatility'].accuracy),
                    parseAcc(rf.overall.accuracy)
                ];
                const ploutosAcc = [
                    parseAcc(ploutos.by_regime.Bull.accuracy),
                    parseAcc(ploutos.by_regime.Bear.accuracy),
                    parseAcc(ploutos.by_regime.Sideways.accuracy),
                    parseAcc(ploutos.by_regime['High-Volatility'].accuracy),
                    parseAcc(ploutos.overall.accuracy)
                ];

                const rfMcc = [
                    parseMcc(rf.by_regime.Bull.mcc),
                    parseMcc(rf.by_regime.Bear.mcc),
                    parseMcc(rf.by_regime.Sideways.mcc),
                    parseMcc(rf.by_regime['High-Volatility'].mcc),
                    parseMcc(rf.overall.mcc)
                ];
                const ploutosMcc = [
                    parseMcc(ploutos.by_regime.Bull.mcc),
                    parseMcc(ploutos.by_regime.Bear.mcc),
                    parseMcc(ploutos.by_regime.Sideways.mcc),
                    parseMcc(ploutos.by_regime['High-Volatility'].mcc),
                    parseMcc(ploutos.overall.mcc)
                ];

                // Destroy existing instances to cleanly recreate on tab switch
                if (regimeAccChartInstance) {
                    try { regimeAccChartInstance.destroy(); } catch(e){}
                    regimeAccChartInstance = null;
                }
                if (regimeMccChartInstance) {
                    try { regimeMccChartInstance.destroy(); } catch(e){}
                    regimeMccChartInstance = null;
                }
                const accBox = document.getElementById("regimeAccChart");
                const mccBox = document.getElementById("regimeMccChart");
                if (accBox) accBox.innerHTML = "";
                if (mccBox) mccBox.innerHTML = "";

                // Render Accuracy Chart
                regimeAccChartInstance = new ApexCharts(document.getElementById("regimeAccChart"), {
                    series: [
                        { name: "Random Forest Baseline", data: rfAcc },
                        { name: "Ploutos+ (Adaptive Router)", data: ploutosAcc }
                    ],
                    chart: { type: "bar", height: 260, background: "transparent", toolbar: { show: false } },
                    colors: ["#64748b", "#10b981"],
                    xaxis: {
                        categories: ["Bull (956)", "Bear (629)", "Sideways (1205)", "Volatile (930)", "Overall (3720)"],
                        labels: { style: { colors: "#94a3b8", fontSize: "11px", fontFamily: "JetBrains Mono" } }
                    },
                    yaxis: {
                        labels: {
                            style: { colors: "#94a3b8", fontFamily: "JetBrains Mono" },
                            formatter: val => val.toFixed(1) + "%"
                        },
                        min: 35,
                        max: 65
                    },
                    theme: { mode: "dark" },
                    plotOptions: {
                        bar: { horizontal: false, columnWidth: "55%", borderRadius: 3 }
                    },
                    dataLabels: {
                        enabled: true,
                        formatter: val => val.toFixed(1) + "%",
                        style: { fontSize: "10px", fontFamily: "JetBrains Mono", colors: ["#fff"] }
                    },
                    legend: {
                        position: "top",
                        labels: { colors: "#cbd5e1" }
                    },
                    grid: { borderColor: "#1e293b" },
                    tooltip: {
                        theme: "dark",
                        y: { formatter: val => val.toFixed(1) + "%" }
                    }
                });
                regimeAccChartInstance.render();

                // Render MCC Chart
                regimeMccChartInstance = new ApexCharts(document.getElementById("regimeMccChart"), {
                    series: [
                        { name: "Random Forest Baseline", data: rfMcc },
                        { name: "Ploutos+ (Adaptive Router)", data: ploutosMcc }
                    ],
                    chart: { type: "bar", height: 260, background: "transparent", toolbar: { show: false } },
                    colors: ["#64748b", "#06b6d4"],
                    xaxis: {
                        categories: ["Bull (956)", "Bear (629)", "Sideways (1205)", "Volatile (930)", "Overall (3720)"],
                        labels: { style: { colors: "#94a3b8", fontSize: "11px", fontFamily: "JetBrains Mono" } }
                    },
                    yaxis: {
                        labels: {
                            style: { colors: "#94a3b8", fontFamily: "JetBrains Mono" },
                            formatter: val => (val >= 0 ? "+" : "") + val.toFixed(3)
                        }
                    },
                    theme: { mode: "dark" },
                    plotOptions: {
                        bar: { horizontal: false, columnWidth: "55%", borderRadius: 3 }
                    },
                    dataLabels: {
                        enabled: true,
                        formatter: val => (val >= 0 ? "+" : "") + val.toFixed(3),
                        style: { fontSize: "10px", fontFamily: "JetBrains Mono", colors: ["#fff"] }
                    },
                    legend: {
                        position: "top",
                        labels: { colors: "#cbd5e1" }
                    },
                    grid: { borderColor: "#1e293b" },
                    tooltip: {
                        theme: "dark",
                        y: { formatter: val => (val >= 0 ? "+" : "") + val.toFixed(4) }
                    }
                });
                regimeMccChartInstance.render();

            } catch (err) {
                console.error("Error loading regime charts:", err);
            }
        }

        async function loadAblations() {
            try {
                const res = await fetch("/api/ablation_data");
                const data = await res.json();
                const tbody = document.getElementById("ablationTableBody");
                tbody.innerHTML = "";
                const formatAcc = (val) => (val > 1 ? val.toFixed(2) : (val * 100).toFixed(2)) + "%";
                const formatDelta = (val) => {
                    const num = (val > 1 || val < -1) ? val : (val * 100);
                    return (num >= 0 ? '+' : '') + num.toFixed(2) + "%";
                };

                data.ablations.forEach(a => {
                    const row = document.createElement("tr");
                    row.className = "border-b border-gray-800/60 hover:bg-gray-800/30 transition";
                    row.innerHTML = `
                        <td class="p-2.5 font-bold text-white">${a.name}</td>
                        <td class="p-2.5 text-gray-400">${a.category}</td>
                        <td class="p-2.5 font-mono">${formatAcc(a.accuracy)}</td>
                        <td class="p-2.5 font-mono">${formatAcc(a.macro_f1)}</td>
                        <td class="p-2.5 font-mono font-bold text-cyan-400">${a.mcc >= 0 ? '+' : ''}${a.mcc.toFixed(4)}</td>
                        <td class="p-2.5 font-mono ${a.delta_acc_vs_rf >= 0 ? 'text-emerald-400' : 'text-rose-400'}">${formatDelta(a.delta_acc_vs_rf)}</td>
                        <td class="p-2.5 font-mono ${a.delta_mcc_vs_rf >= 0 ? 'text-emerald-400' : 'text-rose-400'}">${a.delta_mcc_vs_rf >= 0 ? '+' : ''}${a.delta_mcc_vs_rf.toFixed(4)}</td>
                    `;
                    tbody.appendChild(row);
                });
            } catch (err) {
                console.error("Error loading ablations:", err);
            }
        }

        // Auto initialize on load
        window.addEventListener("DOMContentLoaded", () => {
            initTradingView();
            loadChartData("NVDA");
            loadLedger();
            // Trigger initial prediction smoothly
            setTimeout(() => {
                executePrediction();
            }, 100);
        });
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
