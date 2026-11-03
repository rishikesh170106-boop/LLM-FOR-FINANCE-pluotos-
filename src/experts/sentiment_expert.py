"""
src/experts/sentiment_expert.py
Sentiment Expert component of PloutosGen:
Analyzes financial social-media text, extracts sentiment tone using financial polarity cues,
and handles missing tweets explicitly without hallucination.
"""
from typing import Dict, Any, List
import re
import pandas as pd

BULLISH_KEYWORDS = {
    "call", "long", "buy", "bull", "bullish", "breakout", "rally", "gain",
    "profit", "high", "upgrade", "outperform", "growth", "beat", "surge",
    "dividend", "upside", "target raised", "strong"
}

BEARISH_KEYWORDS = {
    "put", "short", "sell", "bear", "bearish", "breakdown", "drop", "loss",
    "fall", "downgrade", "underperform", "plunge", "miss", "crash", "risk",
    "weak", "target cut", "downside", "warning"
}

NO_TWEET_PLACEHOLDER = "[No textual evidence available for this trading day]"

class SentimentExpert:
    def __init__(self):
        self.expert_name = "Sentiment Expert"

    def analyze(self, row: pd.Series) -> Dict[str, Any]:
        """
        Analyzes textual evidence for a given stock-day.
        """
        ticker = row.get("ticker", "UNKNOWN")
        date = str(row.get("date", "UNKNOWN"))
        has_tweets = bool(row.get("has_tweets", False))
        text = str(row.get("text_evidence", ""))

        if not has_tweets or text.strip() == NO_TWEET_PLACEHOLDER or not text.strip():
            return {
                "expert_name": self.expert_name,
                "ticker": ticker,
                "as_of_date": date,
                "signal": "NEUTRAL",
                "confidence": 0.50,
                "evidence": "No social-media discourse or news text was published for this stock on this date.",
                "metrics": {"tweet_count": 0, "bull_score": 0, "bear_score": 0},
                "source": "StockNet Social Media Archive",
                "availability": False
            }

        # Clean individual tweet bullets
        raw_lines = [l.strip().lstrip("- ").strip() for l in text.split("\n") if l.strip()]
        tweet_count = len(raw_lines)

        bull_count = 0
        bear_count = 0

        for line in raw_lines:
            lower_line = line.lower()
            tokens = set(re.findall(r'\b\w+\b', lower_line))
            bull_hits = len(tokens.intersection(BULLISH_KEYWORDS))
            bear_hits = len(tokens.intersection(BEARISH_KEYWORDS))
            bull_count += bull_hits
            bear_count += bear_hits

        total_sentiment_tokens = bull_count + bear_count

        if total_sentiment_tokens == 0:
            signal = "NEUTRAL"
            confidence = 0.50
            tone_desc = f"{tweet_count} tweets were discussed with neutral/informational market commentary."
        elif bull_count > bear_count:
            diff = bull_count - bear_count
            signal = "BULLISH"
            confidence = min(0.55 + 0.05 * diff, 0.85)
            tone_desc = f"Predominantly bullish sentiment ({bull_count} positive cues vs {bear_count} negative cues) across {tweet_count} tweets."
        elif bear_count > bull_count:
            diff = bear_count - bull_count
            signal = "BEARISH"
            confidence = min(0.55 + 0.05 * diff, 0.85)
            tone_desc = f"Predominantly bearish sentiment ({bear_count} negative cues vs {bull_count} positive cues) across {tweet_count} tweets."
        else:
            signal = "NEUTRAL"
            confidence = 0.50
            tone_desc = f"Balanced sentiment volume with conflicting bull and bear opinions across {tweet_count} tweets."

        sample_snippet = raw_lines[0][:120] if raw_lines else ""
        evidence_text = f"{tone_desc} Key headline sample: \"{sample_snippet}\""

        return {
            "expert_name": self.expert_name,
            "ticker": ticker,
            "as_of_date": date,
            "signal": signal,
            "confidence": round(confidence, 2),
            "evidence": evidence_text,
            "metrics": {
                "tweet_count": tweet_count,
                "bull_score": bull_count,
                "bear_score": bear_count
            },
            "source": "StockNet Social Media Archive",
            "availability": True
        }
