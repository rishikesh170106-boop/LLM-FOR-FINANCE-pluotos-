"""
src/live/investment_engine.py
Actionable Investment Decision & Quantitative Capital Allocation Engine.
Translates raw model directional forecasts and multi-expert evidence into
actionable, institutional risk-managed investment recommendations:
- Verdict: STRONG_BUY, SCALE_IN, HOLD_CASH, AVOID_EXIT
- Automated Risk Guardrails: Entry, Stop-Loss, Take-Profit, Risk/Reward Ratio
- Position Sizing Suggestion & Risk Classification
- Clear, plain-English capital preservation rationales
"""
from typing import Dict, Any, Optional
import numpy as np


class InvestmentDecisionEngine:
    """
    Evaluates multi-expert predictions to produce institutional-grade
    investment recommendations and risk management levels.
    """
    def __init__(
        self,
        min_confidence_strong_buy: float = 0.67,
        min_confidence_scale_in: float = 0.58,
        min_rr_ratio: float = 1.30
    ):
        self.min_strong_buy_conf = min_confidence_strong_buy
        self.min_scale_in_conf = min_confidence_scale_in
        self.min_rr_ratio = min_rr_ratio

    def evaluate(
        self,
        current_price: float,
        predicted_direction: str,
        confidence: float,
        technical_metrics: Dict[str, Any],
        sentiment_metrics: Dict[str, Any],
        visual_metrics: Dict[str, Any],
        dominant_expert: str,
        weights: Optional[Dict[str, float]] = None,
        currency_symbol: str = "$"
    ) -> Dict[str, Any]:
        """
        Computes an actionable investment recommendation with dynamic risk levels.
        """
        curr = currency_symbol
        direction = predicted_direction.upper()
        conf = float(confidence)
        price = float(current_price)

        vol_5d = float(technical_metrics.get("volatility_5d", 0.015))
        mom_5d = float(technical_metrics.get("momentum_5d", 0.0))
        rsi = float(technical_metrics.get("rsi_14", 50.0))
        tech_sig = technical_metrics.get("signal", "NEUTRAL")

        sent_sig = sentiment_metrics.get("signal", "NEUTRAL")
        sent_polarity = float(sentiment_metrics.get("polarity", 0.0))

        vis_sig = visual_metrics.get("signal", "NEUTRAL")
        patterns = visual_metrics.get("patterns", [])

        # 1. Compute Dynamic Support, Resistance, Stop-Loss, and Take-Profit
        # Default volatility buffer: 1.8x 5-day daily volatility (bounded between 1.5% and 5.0%)
        buffer_pct = np.clip(vol_5d * 1.8, 0.015, 0.050)
        
        # Upside projected target: scaled by momentum and volatility
        upside_pct = np.clip(max(abs(mom_5d) * 1.2, buffer_pct * 1.6), 0.025, 0.080)

        if direction in ["BULLISH", "UP"]:
            stop_loss = round(price * (1.0 - buffer_pct), 2)
            take_profit = round(price * (1.0 + upside_pct), 2)
            risk = max(price - stop_loss, 0.01)
            reward = max(take_profit - price, 0.01)
            rr_ratio = round(reward / risk, 2)
        else:
            # Bearish scenario: Invalidation level above, downside target below
            stop_loss = round(price * (1.0 + buffer_pct), 2)
            take_profit = round(price * (1.0 - upside_pct), 2)
            risk = max(stop_loss - price, 0.01)
            reward = max(price - take_profit, 0.01)
            rr_ratio = round(reward / risk, 2)

        # 2. Risk Profile Classification
        if vol_5d < 0.012 and 40.0 <= rsi <= 60.0:
            risk_level = "LOW RISK"
        elif vol_5d <= 0.025:
            risk_level = "MODERATE RISK"
        else:
            risk_level = "HIGH RISK / ELEVATED VOLATILITY"

        # 3. Determine Investment Verdict
        # Count confirming signals for Bullish
        bull_confirmations = 0
        if tech_sig == "BULLISH":
            bull_confirmations += 1
        if sent_sig == "BULLISH":
            bull_confirmations += 1
        if vis_sig == "BULLISH":
            bull_confirmations += 1

        bear_confirmations = 0
        if tech_sig == "BEARISH":
            bear_confirmations += 1
        if sent_sig == "BEARISH":
            bear_confirmations += 1
        if vis_sig == "BEARISH":
            bear_confirmations += 1

        has_conflict = (tech_sig == "BEARISH" and sent_sig == "BULLISH") or (tech_sig == "BULLISH" and sent_sig == "BEARISH")

        if direction in ["BULLISH", "UP"]:
            if conf >= self.min_strong_buy_conf and bull_confirmations >= 2 and not has_conflict and rr_ratio >= self.min_rr_ratio:
                verdict = "STRONG_BUY"
                verdict_label = "RECOMMENDED TO INVEST (STRONG BUY)"
                action_badge = "🟢 INVEST NOW"
                action_badge_ascii = "[INVEST NOW - STRONG BUY]"
                allocation_pct = 100
                allocation_label = "Standard Full Allocation (e.g. 3–5% of portfolio)"
                decision_reason = (
                    f"High conviction ({conf*100:.0f}% confidence) with multi-expert consensus. "
                    f"Attractive Risk/Reward ratio of 1:{rr_ratio:.2f} with Stop-Loss at {curr}{stop_loss:,.2f} (-{buffer_pct*100:.1f}%) "
                    f"and Take-Profit at {curr}{take_profit:,.2f} (+{upside_pct*100:.1f}%)."
                )
            elif conf >= self.min_scale_in_conf and not (tech_sig == "BEARISH" and vis_sig == "BEARISH"):
                verdict = "SCALE_IN"
                verdict_label = "CAUTIOUS ACCUMULATE (SCALE-IN)"
                action_badge = "🟡 CAUTIOUS INVEST"
                action_badge_ascii = "[CAUTIOUS INVEST - SCALE IN]"
                allocation_pct = 50
                allocation_label = "Half Allocation (e.g. 1.5–2.5% of portfolio with tight stop)"
                decision_reason = (
                    f"Moderate conviction ({conf*100:.0f}% confidence). Favorable directional tilt, "
                    f"but elevated risk/volatility warrants scaling in cautiously with disciplined stop-loss at {curr}{stop_loss:,.2f}."
                )
            else:
                verdict = "HOLD_CASH"
                verdict_label = "DO NOT INVEST (WAIT IN CASH)"
                action_badge = "⚪ DO NOT INVEST / WAIT"
                action_badge_ascii = "[DO NOT INVEST - WAIT IN CASH]"
                allocation_pct = 0
                allocation_label = "0% Allocation (Capital Preservation)"
                decision_reason = (
                    f"Confidence ({conf*100:.0f}%) or expert consensus is insufficient to risk capital. "
                    f"Market exhibits conflicting signals or high noise. Staying in cash avoids unnecessary drawdown."
                )
        else:
            # Bearish Forecast
            if conf >= 0.60:
                verdict = "AVOID_EXIT"
                verdict_label = "DO NOT INVEST (AVOID / PROTECT CAPITAL)"
                action_badge = "🔴 DO NOT INVEST / AVOID"
                action_badge_ascii = "[DO NOT INVEST - AVOID DOWNSIDE]"
                allocation_pct = 0
                allocation_label = "0% Allocation (Exit Existing Longs / AVOID)"
                decision_reason = (
                    f"Model identifies active downside risk ({conf*100:.0f}% confidence) driven by {dominant_expert}. "
                    f"Protecting capital is the optimal strategy. Do not enter new long positions."
                )
            else:
                verdict = "HOLD_CASH"
                verdict_label = "DO NOT INVEST (CHOPPY / NEUTRAL MARKET)"
                action_badge = "⚪ DO NOT INVEST / WAIT"
                action_badge_ascii = "[DO NOT INVEST - WAIT FOR CLEAR SETUP]"
                allocation_pct = 0
                allocation_label = "0% Allocation (Wait for Clear Setup)"
                decision_reason = (
                    f"Directional bias is negative but confidence is weak ({conf*100:.0f}%). "
                    f"Preserve cash until clear institutional accumulation or breakdown occurs."
                )

        return {
            "verdict": verdict,
            "verdict_label": verdict_label,
            "action_badge": action_badge,
            "action_badge_ascii": action_badge_ascii,
            "can_invest": verdict in ["STRONG_BUY", "SCALE_IN"],
            "allocation_pct": allocation_pct,
            "allocation_label": allocation_label,
            "risk_level": risk_level,
            "entry_price": price,
            "stop_loss_price": stop_loss,
            "take_profit_price": take_profit,
            "stop_loss_pct": round(buffer_pct * 100, 2),
            "take_profit_pct": round(upside_pct * 100, 2),
            "risk_reward_ratio": rr_ratio,
            "currency_symbol": curr,
            "decision_reason": decision_reason
        }
