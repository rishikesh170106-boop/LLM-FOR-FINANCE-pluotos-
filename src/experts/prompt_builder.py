"""
src/experts/prompt_builder.py
Synthesizes multi-expert evidence (Technical, Sentiment, Human Proxy) into structured
instruction-tuning prompts and rearview-mirror rationales for LLM training and inference.
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Dict, Any, List, Optional
import pandas as pd
from tqdm import tqdm

from src.experts.technical_expert import TechnicalExpert
from src.experts.sentiment_expert import SentimentExpert
from src.experts.human_expert import HumanExpert
from src.experts.visual_expert import VisualFinancialExpert

SYSTEM_PROMPT = (
    "You are Ploutos+, a specialized financial AI analyst predicting next-period stock movement "
    "(UP or DOWN) based on grounded multi-expert financial evidence, technical market dynamics, and candlestick chart patterns."
)

class PloutosPromptBuilder:
    def __init__(self, include_visual_expert: bool = True):
        self.technical_expert = TechnicalExpert()
        self.sentiment_expert = SentimentExpert()
        self.human_expert = HumanExpert()
        self.include_visual_expert = include_visual_expert
        if include_visual_expert:
            self.visual_expert = VisualFinancialExpert()

    def build_expert_evidence(self, row: pd.Series) -> Dict[str, Any]:
        """Queries all experts for a single observation."""
        tech_out = self.technical_expert.analyze(row)
        sent_out = self.sentiment_expert.analyze(row)
        human_out = self.human_expert.analyze(row)
        evidence_dict = {
            "technical": tech_out,
            "sentiment": sent_out,
            "human": human_out
        }
        if self.include_visual_expert:
            ticker = row.get("ticker", "UNKNOWN")
            date = str(row.get("date", "UNKNOWN"))
            evidence_dict["visual"] = self.visual_expert.analyze(ticker, date, generate_plot=False)
        return evidence_dict

    def format_input_context(self, ticker: str, date: str, experts: Dict[str, Any]) -> str:
        """Formats the multi-expert perspectives into a standardized prompt context."""
        tech = experts["technical"]
        sent = experts["sentiment"]
        human = experts["human"]

        context_lines = [
            f"Ticker: ${ticker}",
            f"Prediction Target Date: {date}",
            "",
            f"--- [1. Technical Expert Evidence] ---",
            f"Signal: {tech['signal']} (Confidence: {tech['confidence']:.2f})",
            f"Evidence: {tech['evidence']}",
            "",
            f"--- [2. Sentiment Expert Evidence] ---",
            f"Signal: {sent['signal']} (Confidence: {sent['confidence']:.2f})",
            f"Evidence: {sent['evidence']}",
            "",
            f"--- [3. Corporate & Sector Background (Human Expert Proxy)] ---",
            f"Context: {human['evidence']}"
        ]

        if "visual" in experts:
            vis = experts["visual"]
            context_lines.extend([
                "",
                f"--- [4. Visual Chart Expert Evidence] ---",
                f"Signal: {vis['signal']} (Confidence: {vis['confidence']:.2f})",
                f"Evidence: {vis['evidence']}"
            ])

        return "\n".join(context_lines)

    def generate_grounded_rationale(
        self, label: int, ticker: str, experts: Dict[str, Any]
    ) -> str:
        """
        Constructs a grounded rearview-mirror explanation for historical training examples.
        Synthesizes technical and sentiment factors explaining the observed target movement.
        """
        direction_str = "UP" if label == 1 else "DOWN"
        tech = experts["technical"]
        sent = experts["sentiment"]

        mom = tech["metrics"]["momentum_5d"]
        rsi = tech["metrics"]["rsi_14"]
        tweet_count = sent["metrics"]["tweet_count"]

        reasons = []
        if label == 1:
            if mom > 0:
                reasons.append(f"established positive momentum (+{mom*100:.2f}%) created favorable buying tailwinds")
            else:
                reasons.append(f"oversold conditions (RSI: {rsi:.1f}) triggered an attractive risk-reward technical bounce")

            if sent["signal"] == "BULLISH":
                reasons.append(f"constructive social-media sentiment reinforced market confidence")
            elif tweet_count == 0:
                reasons.append(f"steady institutional accumulation occurred amid low public noise")
            else:
                reasons.append(f"investor demand absorbed negative news sentiment")
        else:
            if mom < 0:
                reasons.append(f"persistent downward momentum ({mom*100:.2f}%) indicated continued distribution")
            else:
                reasons.append(f"overextended technical levels (RSI: {rsi:.1f}) prompted profit taking")

            if sent["signal"] == "BEARISH":
                reasons.append(f"bearish sentiment and caution weighed on near-term valuation")
            elif tweet_count == 0:
                reasons.append(f"lack of supportive catalyst allowed broader market pressure to dominate")
            else:
                reasons.append(f"bullish sentiment failed to overcome prevailing selling resistance")

        rationale_body = " Furthermore, ".join(reasons)
        return (
            f"Prediction: {direction_str}\n"
            f"Rationale: The stock demonstrated {direction_str} movement because {rationale_body}. "
            f"Overall technical and qualitative evidence supported this directional outcome."
        )

    def build_instruction_example(self, row: pd.Series) -> Dict[str, Any]:
        """Constructs a complete instruction tuning sample."""
        ticker = row["ticker"]
        date = str(row["date"])
        label = int(row["label"])

        experts = self.build_expert_evidence(row)
        input_context = self.format_input_context(ticker, date, experts)
        output_completion = self.generate_grounded_rationale(label, ticker, experts)

        instruction = (
            "Analyze the multi-expert financial evidence below and predict whether the stock will "
            "move UP or DOWN on the target date. Provide your prediction followed by a grounded rationale."
        )

        return {
            "id": f"{ticker}_{date}",
            "ticker": ticker,
            "date": date,
            "system": SYSTEM_PROMPT,
            "instruction": instruction,
            "input": input_context,
            "output": output_completion,
            "label": label,
            "target_movement": float(row.get("movement", 0.0))
        }

def generate_ploutos_prompt_splits(max_samples_per_split: Optional[int] = None):
    """Generates instruction-tuning JSONL datasets for Train, Val, and Test."""
    from src.features.technical_features import extract_all_technical_features

    builder = PloutosPromptBuilder()
    processed_dir = Path("data/processed")

    split_files = {
        "train": processed_dir / "train_with_text.csv",
        "val": processed_dir / "val_with_text.csv",
        "test": processed_dir / "test_with_text.csv"
    }

    train_df = pd.read_csv(split_files["train"])
    tickers = sorted(train_df["ticker"].unique())
    tech_features = extract_all_technical_features(tickers)

    for split_name, split_path in split_files.items():
        print(f"\n[Prompt Builder] Generating Ploutos prompts for {split_name}...")
        df = pd.read_csv(split_path)
        df["date"] = df["date"].astype(str)

        merged = df.merge(tech_features, on=["ticker", "date"], how="left")
        if max_samples_per_split and len(merged) > max_samples_per_split:
            merged = merged.sample(n=max_samples_per_split, random_state=42).reset_index(drop=True)

        output_jsonl = processed_dir / f"ploutos_{split_name}_prompts.jsonl"
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for _, row in tqdm(merged.iterrows(), total=len(merged), desc=split_name):
                sample = builder.build_instruction_example(row)
                f.write(json.dumps(sample) + "\n")

        print(f"  Saved {len(merged):,} instruction prompts to: {output_jsonl}")

if __name__ == "__main__":
    generate_ploutos_prompt_splits()
