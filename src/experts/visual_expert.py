"""
src/experts/visual_expert.py
Phase 9: Visual Financial Expert (Multimodal Extension)
- Reconstructs strictly lagged 15-day OHLC candlestick price series (zero leakage: dates < t).
- Detects core technical chart patterns:
    * Bullish Hammer / Pin Bar
    * Bearish Shooting Star
    * Bullish & Bearish Engulfing
    * Support / Resistance Tests
    * 5-day vs 15-day Moving Average (SMA) Trend
    * Institutional Volume Expansion
- Renders high-resolution dark-themed candlestick charts with volume subpanels.
- Formulates structured visual evidence blocks for Ploutos multimodal prompts.
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure interim directory for matplotlib cache BEFORE importing matplotlib
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data" / "interim"))

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for thread-safety
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

STOCKNET_PRICE_DIR = PROJECT_ROOT / "data" / "raw" / "stocknet-dataset" / "price" / "preprocessed"
CHARTS_DIR = PROJECT_ROOT / "results" / "figures" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


class VisualFinancialExpert:
    """
    Multimodal visual expert that renders candlestick charts and extracts
    spatial price action and candlestick reversal patterns.
    """
    def __init__(self, lookback_days: int = 15):
        self.expert_name = "Visual Financial Expert"
        self.lookback_days = lookback_days
        self._price_cache: Dict[str, pd.DataFrame] = {}

    def _load_ticker_price_history(self, ticker: str) -> pd.DataFrame:
        """Loads and sorts historical prices for a single ticker."""
        if ticker in self._price_cache:
            return self._price_cache[ticker]

        price_file = STOCKNET_PRICE_DIR / f"{ticker}.txt"
        if not price_file.exists():
            return pd.DataFrame()

        df = pd.read_csv(price_file, sep=r'\s+', header=None)
        df.columns = ["date", "movement", "open_return", "high_return", "low_return", "price_change", "volume"]
        df["date"] = df["date"].astype(str)
        df = df.sort_values("date").reset_index(drop=True)
        self._price_cache[ticker] = df
        return df

    def extract_lagged_ohlc(
        self, ticker: str, target_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Extracts strictly lagged OHLC prices for the lookback window preceding target_date.
        All records strictly have date < target_date. Zero future leakage.
        """
        df = self._load_ticker_price_history(ticker)
        if df.empty:
            return None

        # Filter strictly prior to target_date
        prior_df = df[df["date"] < str(target_date)].tail(self.lookback_days).copy()
        if len(prior_df) < 5:
            return None

        # Reconstruct normalized cumulative price levels (base = 100.0)
        # Close_t = Close_{t-1} * (1 + movement_t)
        n = len(prior_df)
        close_prices = np.zeros(n)
        open_prices = np.zeros(n)
        high_prices = np.zeros(n)
        low_prices = np.zeros(n)

        current_close = 100.0
        # Reconstruct forward across the lookback slice
        for i in range(n):
            mov = prior_df["movement"].iloc[i]
            op_ret = prior_df["open_return"].iloc[i]
            hi_ret = prior_df["high_return"].iloc[i]
            lo_ret = prior_df["low_return"].iloc[i]

            prev_close = current_close
            current_close = prev_close * (1.0 + mov)

            close_prices[i] = current_close
            open_prices[i] = prev_close * (1.0 + op_ret)
            high_prices[i] = max(prev_close * (1.0 + hi_ret), open_prices[i], close_prices[i])
            low_prices[i] = min(prev_close * (1.0 + lo_ret), open_prices[i], close_prices[i])

        prior_df["norm_open"] = open_prices
        prior_df["norm_high"] = high_prices
        prior_df["norm_low"] = low_prices
        prior_df["norm_close"] = close_prices
        prior_df["volume_val"] = prior_df["volume"].fillna(1e6).values

        return prior_df.reset_index(drop=True)

    def detect_chart_patterns(self, ohlc_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detects core candlestick patterns and technical levels on the lagged series.
        """
        patterns = []
        signal = "NEUTRAL"
        confidence = 0.50
        bullish_score = 0.0
        bearish_score = 0.0

        if len(ohlc_df) < 5:
            return {
                "signal": "NEUTRAL",
                "confidence": 0.50,
                "patterns": ["Insufficient historical chart data"],
                "metrics": {}
            }

        # Most recent closed candle (t-1)
        last = ohlc_df.iloc[-1]
        prev = ohlc_df.iloc[-2] if len(ohlc_df) >= 2 else last

        o, h, l, c = last["norm_open"], last["norm_high"], last["norm_low"], last["norm_close"]
        body_size = abs(c - o)
        full_range = max(h - l, 1e-6)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        # 1. Bullish Hammer / Pin Bar
        if lower_wick >= 1.8 * body_size and upper_wick <= 0.5 * body_size + 1e-4:
            patterns.append("Bullish Hammer Pin-Bar (Strong lower shadow rejection)")
            bullish_score += 0.45

        # 2. Bearish Shooting Star
        if upper_wick >= 1.8 * body_size and lower_wick <= 0.5 * body_size + 1e-4:
            patterns.append("Bearish Shooting Star (Upper wick resistance rejection)")
            bearish_score += 0.45

        # 3. Bullish Engulfing
        prev_o, prev_c = prev["norm_open"], prev["norm_close"]
        if prev_c < prev_o and c > o and c > prev_o and o < prev_c:
            patterns.append("Bullish Engulfing Pattern (Buyers overwhelm previous selling candle)")
            bullish_score += 0.40

        # 4. Bearish Engulfing
        if prev_c > prev_o and c < o and c < prev_o and o > prev_c:
            patterns.append("Bearish Engulfing Pattern (Sellers overwhelm previous buying candle)")
            bearish_score += 0.40

        # 5. Support / Resistance Levels
        min_15d = ohlc_df["norm_low"].min()
        max_15d = ohlc_df["norm_high"].max()

        if abs(l - min_15d) / min_15d < 0.015 and c > o:
            patterns.append(f"Support Bounce (Tested 15-day range low at {min_15d:.2f} and rebounded)")
            bullish_score += 0.35
        elif abs(h - max_15d) / max_15d < 0.015 and c < o:
            patterns.append(f"Resistance Rejection (Rejected at 15-day ceiling of {max_15d:.2f})")
            bearish_score += 0.35

        # 6. Moving Average Slope
        sma_5 = ohlc_df["norm_close"].tail(5).mean()
        sma_15 = ohlc_df["norm_close"].mean()
        if sma_5 > sma_15 * 1.01:
            patterns.append("Short-term Moving Average Bullish Alignment (SMA-5 > SMA-15)")
            bullish_score += 0.25
        elif sma_5 < sma_15 * 0.99:
            patterns.append("Short-term Moving Average Bearish Alignment (SMA-5 < SMA-15)")
            bearish_score += 0.25

        # 7. Volume Expansion
        avg_vol = ohlc_df["volume_val"].mean()
        last_vol = last["volume_val"]
        vol_ratio = last_vol / (avg_vol + 1e-6)
        if vol_ratio > 1.30:
            direction_tag = "green" if c >= o else "red"
            patterns.append(f"Volume Surge (+{int((vol_ratio-1)*100)}% on {direction_tag} candle)")
            if c >= o:
                bullish_score += 0.20
            else:
                bearish_score += 0.20

        # Formulate final signal
        net_score = bullish_score - bearish_score
        if net_score >= 0.25:
            signal = "BULLISH"
            confidence = min(0.85, 0.55 + bullish_score * 0.3)
        elif net_score <= -0.25:
            signal = "BEARISH"
            confidence = min(0.85, 0.55 + bearish_score * 0.3)
        else:
            signal = "NEUTRAL"
            confidence = 0.52

        if not patterns:
            patterns.append("Neutral range-bound price action with no extreme breakout patterns")

        return {
            "signal": signal,
            "confidence": round(confidence, 2),
            "patterns": patterns,
            "metrics": {
                "last_close": round(float(c), 2),
                "sma_5": round(float(sma_5), 2),
                "sma_15": round(float(sma_15), 2),
                "support_15d": round(float(min_15d), 2),
                "resistance_15d": round(float(max_15d), 2),
                "vol_ratio": round(float(vol_ratio), 2)
            }
        }

    def render_candlestick_chart(
        self,
        ohlc_df: pd.DataFrame,
        ticker: str,
        target_date: str,
        patterns: List[str],
        save_path: Optional[Path] = None
    ) -> Path:
        """
        Renders a publication-grade dark-themed financial candlestick chart
        with moving averages and volume subpanels using pure Matplotlib.
        """
        if save_path is None:
            save_path = CHARTS_DIR / f"{ticker}_{target_date}.png"

        # Dark theme palette
        BG_COLOR = "#0F0F1A"
        PANEL_COLOR = "#1A1A2E"
        GRID_COLOR = "#2A2A40"
        TEXT_COLOR = "#EEEEEE"
        GREEN_COLOR = "#00D4AA"
        RED_COLOR = "#FF4D6A"
        SMA5_COLOR = "#FFB800"
        SMA15_COLOR = "#4E8CFF"

        fig = plt.figure(figsize=(9, 5.5), facecolor=BG_COLOR)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.15)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        ax1.set_facecolor(PANEL_COLOR)
        ax2.set_facecolor(PANEL_COLOR)

        n = len(ohlc_df)
        x_indices = np.arange(n)

        # Plot Candlesticks
        candle_width = 0.55
        for i in range(n):
            row = ohlc_df.iloc[i]
            op = row["norm_open"]
            cl = row["norm_close"]
            hi = row["norm_high"]
            lo = row["norm_low"]

            is_up = cl >= op
            color = GREEN_COLOR if is_up else RED_COLOR

            # Wick line
            ax1.plot([i, i], [lo, hi], color=color, linewidth=1.2, zorder=2)

            # Candle body
            rect_bottom = min(op, cl)
            rect_height = max(abs(cl - op), 0.05)
            rect = patches.Rectangle(
                (i - candle_width / 2, rect_bottom),
                candle_width,
                rect_height,
                facecolor=color,
                edgecolor=color,
                zorder=3
            )
            ax1.add_patch(rect)

            # Volume Bar
            ax2.bar(i, row["volume_val"] / 1e6, color=color, alpha=0.75, width=candle_width)

        # Overlay Moving Averages
        if n >= 5:
            sma_5 = ohlc_df["norm_close"].rolling(5).mean()
            ax1.plot(x_indices, sma_5, color=SMA5_COLOR, linewidth=1.5, label="5-Day SMA")
        if n >= 10:
            sma_15 = ohlc_df["norm_close"].rolling(min(15, n)).mean()
            ax1.plot(x_indices, sma_15, color=SMA15_COLOR, linewidth=1.5, label="15-Day SMA")

        # Styling Ax1 (Price)
        ax1.set_title(f"Ploutos+ Visual Expert: ${ticker} 15-Day Chart (Prior to {target_date})",
                      color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=10)
        ax1.set_ylabel("Normalized Price", color=TEXT_COLOR, fontsize=9)
        ax1.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax1.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.6)
        ax1.legend(loc="upper left", facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

        # Top pattern annotation banner
        if patterns:
            top_p = patterns[0]
            ax1.text(0.98, 0.05, f"Pattern: {top_p}", transform=ax1.transAxes,
                     color=TEXT_COLOR, fontsize=8, ha="right",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR, edgecolor=GREEN_COLOR, alpha=0.9))

        # Styling Ax2 (Volume)
        ax2.set_ylabel("Vol (M)", color=TEXT_COLOR, fontsize=8)
        ax2.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax2.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.4)

        # Date X-ticks
        step = max(1, n // 5)
        ax2.set_xticks(x_indices[::step])
        ax2.set_xticklabels([ohlc_df["date"].iloc[i][5:] for i in range(0, n, step)], color=TEXT_COLOR, fontsize=8)

        plt.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.10)
        fig.savefig(save_path, dpi=180, facecolor=BG_COLOR)
        plt.close(fig)

        return save_path

    def analyze(
        self,
        ticker: str,
        target_date: str,
        generate_plot: bool = True
    ) -> Dict[str, Any]:
        """
        Full Visual Expert pipeline:
        1. Extracts strictly lagged 15-day OHLC.
        2. Detects candlestick and support/resistance chart patterns.
        3. Optionally renders the chart PNG.
        4. Synthesizes structured textual evidence.
        """
        ohlc = self.extract_lagged_ohlc(ticker, target_date)
        if ohlc is None or len(ohlc) < 5:
            return {
                "expert_name": self.expert_name,
                "signal": "NEUTRAL",
                "confidence": 0.50,
                "detected_patterns": ["No historical price series available for visual rendering"],
                "chart_path": None,
                "evidence": f"No chart evidence available for ${ticker} prior to {target_date}."
            }

        pattern_res = self.detect_chart_patterns(ohlc)
        patterns = pattern_res["patterns"]
        signal = pattern_res["signal"]
        confidence = pattern_res["confidence"]
        m = pattern_res["metrics"]

        chart_path = None
        if generate_plot:
            chart_path = str(self.render_candlestick_chart(ohlc, ticker, target_date, patterns))

        # Synthesize concise natural-language evidence
        primary_pattern = patterns[0]
        secondary = f" Secondary signal: {patterns[1]}." if len(patterns) > 1 else ""
        evidence_text = (
            f"15-day candlestick chart analysis indicates {signal.lower()} price structure. "
            f"Primary pattern: {primary_pattern}.{secondary} "
            f"SMA alignment (SMA-5: {m.get('sma_5', 'N/A')} vs SMA-15: {m.get('sma_15', 'N/A')}) "
            f"with volume ratio at {m.get('vol_ratio', 1.0)}x normal levels."
        )

        return {
            "expert_name": self.expert_name,
            "signal": signal,
            "confidence": confidence,
            "detected_patterns": patterns,
            "metrics": m,
            "chart_path": chart_path,
            "evidence": evidence_text
        }


if __name__ == "__main__":
    expert = VisualFinancialExpert()
    res = expert.analyze(ticker="AAPL", target_date="2015-10-01", generate_plot=True)
    print("=" * 70)
    print("VISUAL FINANCIAL EXPERT SMOKE TEST")
    print("=" * 70)
    print(f"Signal: {res['signal']} (Confidence: {res['confidence']})")
    print(f"Patterns: {res['detected_patterns']}")
    print(f"Evidence: {res['evidence']}")
    print(f"Saved Chart: {res['chart_path']}")
