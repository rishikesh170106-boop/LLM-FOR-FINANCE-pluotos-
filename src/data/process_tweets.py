"""
src/data/process_tweets.py
Processes and aligns StockNet historical tweets with labeled stock-days:
- Reads tokenized JSON tweet lines
- Joins tokens into readable text
- Applies configurable daily tweet cap (default: 10 tweets per stock-day)
- Preserves stock-days without tweets using a neutral missing-text placeholder
- Produces train_with_text.csv, val_with_text.csv, test_with_text.csv
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import yaml
from tqdm import tqdm

TWEET_DIR = Path("data/raw/stocknet-dataset/tweet/preprocessed")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("results/reports")
CONFIG_PATH = Path("configs/data.yaml")

NO_TWEET_PLACEHOLDER = "[No textual evidence available for this trading day]"

def load_tweet_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("text_processing", {"daily_tweet_cap": 10})
    return {"daily_tweet_cap": 10}

def clean_tweet_tokens(tokens: list) -> str:
    """Reconstructs readable text from StockNet's token list."""
    if not isinstance(tokens, list):
        return str(tokens)
    text = " ".join(tokens)
    # Basic cleanup: remove excess spaces around punctuation
    text = text.replace(" $ ", " $").replace(" '", "'")
    return text.strip()

def extract_tweets_for_stock_day(ticker: str, date: str, cap: int = 10) -> List[str]:
    """Reads tweets for given ticker and date up to the specified daily cap."""
    tweet_file = TWEET_DIR / ticker / date
    if not tweet_file.exists():
        return []

    tweets = []
    try:
        with open(tweet_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    obj = json.loads(line_str)
                    raw_tokens = obj.get("text", [])
                    cleaned = clean_tweet_tokens(raw_tokens)
                    if cleaned:
                        tweets.append(cleaned)
                except Exception:
                    continue
                if len(tweets) >= cap:
                    break
    except Exception:
        return []

    return tweets

def process_and_align_tweets() -> Dict[str, pd.DataFrame]:
    cfg = load_tweet_config()
    daily_cap = cfg.get("daily_tweet_cap", 10)
    print("=" * 65)
    print(f"PLOUTOS+ PHASE 4: TWEET PROCESSING & ALIGNMENT (Daily Cap: {daily_cap})")
    print("=" * 65)

    split_files = {
        "train": PROCESSED_DIR / "train_labels.csv",
        "val": PROCESSED_DIR / "val_labels.csv",
        "test": PROCESSED_DIR / "test_labels.csv"
    }

    results = {}
    stats = {
        "total_stock_days": 0,
        "stock_days_with_tweets": 0,
        "stock_days_without_tweets": 0,
        "total_tweets_after_cap": 0,
        "split_counts": {}
    }

    for split_name, split_path in split_files.items():
        if not split_path.exists():
            print(f"ERROR: Split file {split_path} not found! Run Phase 3 first.")
            return {}

        df = pd.read_csv(split_path)
        df["date"] = df["date"].astype(str)

        tweet_texts = []
        tweet_counts = []
        has_tweets_flags = []

        print(f"[Text Processing] Aligning tweets for {split_name} ({len(df):,} samples)...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split_name:>5}"):
            ticker = row["ticker"]
            date = str(row["date"])

            tweets = extract_tweets_for_stock_day(ticker, date, cap=daily_cap)
            n_tweets = len(tweets)

            if n_tweets > 0:
                # Format into numbered evidence bullets
                formatted_text = "\n".join([f"- {t}" for t in tweets])
                tweet_texts.append(formatted_text)
                tweet_counts.append(n_tweets)
                has_tweets_flags.append(True)
                stats["stock_days_with_tweets"] += 1
                stats["total_tweets_after_cap"] += n_tweets
            else:
                tweet_texts.append(NO_TWEET_PLACEHOLDER)
                tweet_counts.append(0)
                has_tweets_flags.append(False)
                stats["stock_days_without_tweets"] += 1

            stats["total_stock_days"] += 1

        df["tweet_count"] = tweet_counts
        df["has_tweets"] = has_tweets_flags
        df["text_evidence"] = tweet_texts

        output_path = PROCESSED_DIR / f"{split_name}_with_text.csv"
        df.to_csv(output_path, index=False)
        results[split_name] = df
        stats["split_counts"][split_name] = {
            "samples": len(df),
            "with_tweets": sum(has_tweets_flags),
            "without_tweets": len(df) - sum(has_tweets_flags),
            "coverage_pct": round(sum(has_tweets_flags) / len(df) * 100, 2)
        }
        print(f"  Saved: {output_path} (Coverage: {stats['split_counts'][split_name]['coverage_pct']}%)")

    # Combine all into master dataset
    all_with_text = pd.concat(list(results.values()), ignore_index=True)
    all_output_path = PROCESSED_DIR / "all_labeled_with_text.csv"
    all_with_text.to_csv(all_output_path, index=False)
    print(f"\n[Master Dataset] Saved combined {len(all_with_text):,} rows to {all_output_path}")

    # Generate Report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "text_processing_report.md"
    overall_coverage = stats["stock_days_with_tweets"] / stats["total_stock_days"] * 100

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# StockNet Tweet Processing & Alignment Report (Phase 4)\n\n")
        f.write(f"- **Daily Tweet Cap Applied:** {daily_cap} tweets/stock-day\n")
        f.write(f"- **Total Processed Stock-Days:** {stats['total_stock_days']:,}\n")
        f.write(f"- **Stock-Days With Tweets:** {stats['stock_days_with_tweets']:,} ({overall_coverage:.2f}%)\n")
        f.write(f"- **Stock-Days Without Tweets:** {stats['stock_days_without_tweets']:,}\n")
        f.write(f"- **Total Tweet Records Extracted (Capped):** {stats['total_tweets_after_cap']:,}\n")
        f.write(f"- **Missing Text Placeholder:** `{NO_TWEET_PLACEHOLDER}`\n\n")
        f.write("### Split-Level Coverage Breakdown\n\n")
        f.write("| Split | Total Samples | With Tweets | Without Tweets | Tweet Coverage (%) |\n")
        f.write("| :--- | :--- | :--- | :--- | :---: |\n")
        for sp, sp_stat in stats["split_counts"].items():
            f.write(f"| **{sp.capitalize()}** | {sp_stat['samples']:,} | {sp_stat['with_tweets']:,} | {sp_stat['without_tweets']:,} | **{sp_stat['coverage_pct']}%** |\n")

    print(f"Report saved to: {report_path}")
    print("=" * 65)
    print("PHASE 4: TWEET PROCESSING COMPLETE")
    print("=" * 65)
    return results

if __name__ == "__main__":
    process_and_align_tweets()
