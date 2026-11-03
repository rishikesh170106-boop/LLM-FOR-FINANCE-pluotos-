"""
src/experts/technical_expert.py
Technical Expert component of PloutosGen:
Extracts strictly lagged price action indicators (5-day momentum, RSI, volatility)
and generates a structured technical evidence summary with directional signal.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd

class TechnicalExpert:
    def __init__(self):
        self.expert_name = "Technical Expert"

    def analyze(self, row: pd.Series) -> Dict[str, Any]:
        """
        Analyzes lagged numerical indicators for a single stock-day observation.
        All inputs in 'row' must be strictly lagged (derived from dates < t).
        """
        ticker = row.get("ticker", "UNKNOWN")
        date = str(row.get("date", "UNKNOWN"))

        # Extract lagged metrics
        mom_5d = float(row.get("roll_momentum_5d", 0.0))
        vol_5d = float(row.get("roll_return_std_5d", 0.01))
        rsi_14 = float(row.get("lagged_rsi_14", 50.0))
        if np.isnan(rsi_14):
            rsi_14 = 50.0

        # Determine directional signal
        # RSI < 35 -> Oversold (reversal upward), RSI > 65 -> Overbought (potential reversal down)
        # Momentum > 0.01 -> Bullish trend, Momentum < -0.01 -> Bearish trend
        score = 0.0
        evidence_points = []

        if mom_5d > 0.01:
            score += 0.5
            evidence_points.append(f"Positive 5-day momentum of {mom_5d * 100:+.2f}% indicates upward price momentum.")
        elif mom_5d < -0.01:
            score -= 0.5
            evidence_points.append(f"Negative 5-day momentum of {mom_5d * 100:+.2f}% indicates persistent selling pressure.")
        else:
            evidence_points.append(f"Neutral 5-day momentum of {mom_5d * 100:+.2f}%.")

        if rsi_14 > 65:
            score -= 0.3
            evidence_points.append(f"14-day RSI at {rsi_14:.1f} indicates overbought conditions.")
        elif rsi_14 < 35:
            score += 0.3
            evidence_points.append(f"14-day RSI at {rsi_14:.1f} indicates oversold conditions with bounce potential.")
        else:
            evidence_points.append(f"14-day RSI at {rsi_14:.1f} reflects a neutral trading band.")

        if vol_5d > 0.025:
            evidence_points.append(f"Elevated price volatility (std: {vol_5d:.4f}) signals high uncertainty.")
        else:
            evidence_points.append(f"Contained volatility (std: {vol_5d:.4f}) reflects steady order flow.")

        if score > 0.2:
            signal = "BULLISH"
            confidence = min(0.5 + abs(score) * 0.4, 0.90)
        elif score < -0.2:
            signal = "BEARISH"
            confidence = min(0.5 + abs(score) * 0.4, 0.90)
        else:
            signal = "NEUTRAL"
            confidence = 0.50

        summary_text = " ".join(evidence_points)

        return {
            "expert_name": self.expert_name,
            "ticker": ticker,
            "as_of_date": date,
            "signal": signal,
            "confidence": round(confidence, 2),
            "evidence": summary_text,
            "metrics": {
                "momentum_5d": round(mom_5d, 4),
                "rsi_14": round(rsi_14, 2),
                "volatility_5d": round(vol_5d, 4)
            },
            "source": "Lagged OHLCV Technical Series",
            "availability": True
        }
