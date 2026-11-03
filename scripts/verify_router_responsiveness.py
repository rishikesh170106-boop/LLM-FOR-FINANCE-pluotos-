"""
scripts/verify_router_responsiveness.py
Requirement 7 Verification:
Proves that the Context-Gated Router dynamically adapts and shifts expert weights
in response to changing market regimes (Shock vs Sentiment Surge vs Quiet Macro vs No Tweets).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.expert_router import (
    AdaptiveRouterEngine,
    ROUTER_FEATURE_DIM,
    ROUTER_FEATURE_NAMES
)

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "expert_router" / "best_router.pt"


def run_responsiveness_verification():
    print("=" * 75)
    print("PLOUTOS+ REQUIREMENT 7: ROUTER DYNAMIC MARKET RESPONSIVENESS AUDIT")
    print("=" * 75)

    engine = AdaptiveRouterEngine(model_path=CHECKPOINT_PATH, device="cpu")

    regimes = {}

    # 1. Regime 1: Extreme Volatility & Technical Oversold Shock
    # High rolling volatility, heavy negative momentum, oversold RSI
    v_shock = np.zeros(ROUTER_FEATURE_DIM, dtype=np.float32)
    v_shock[0] = 0.045   # high roll_return_std_5d
    v_shock[1] = -0.065  # severe negative momentum
    v_shock[2] = -0.015  # negative return mean
    v_shock[3] = -0.55   # oversold RSI ((22 - 50)/50)
    v_shock[4] = -0.035  # lag 1 return drop
    v_shock[9] = 0.0     # no tweets
    v_shock[10] = 0.0    # has_tweets = 0
    v_shock[13] = 0.90   # high technical confidence
    v_shock[14] = -1.0   # bearish technical signal
    regimes["1. Extreme Bearish / Volatility Shock"] = v_shock

    # 2. Regime 2: Social Media & Retail Sentiment Surge
    # 10 bullish tweets, strong sentiment polarity, normal technicals
    v_sent = np.zeros(ROUTER_FEATURE_DIM, dtype=np.float32)
    v_sent[0] = 0.012   # normal volatility
    v_sent[1] = 0.008   # modest momentum
    v_sent[3] = 0.05    # neutral RSI
    v_sent[9] = 1.0     # max tweet count (10/10)
    v_sent[10] = 1.0    # has_tweets = 1
    v_sent[11] = 0.95   # unanimous bullish polarity
    v_sent[12] = 0.90   # high sentiment confidence
    v_sent[13] = 0.55   # moderate technical confidence
    regimes["2. Retail Social Sentiment Surge (Bullish)"] = v_sent

    # 3. Regime 3: Zero Textual Information Day
    # Identical technicals to Regime 2, but ZERO tweets available
    v_notext = v_sent.copy()
    v_notext[9] = 0.0   # 0 tweets
    v_notext[10] = 0.0  # has_tweets = 0
    v_notext[11] = 0.0  # 0 polarity
    v_notext[12] = 0.0  # 0 confidence
    regimes["3. Zero Information Day (Missing Social Text)"] = v_notext

    # 4. Regime 4: Quiet Sideways Macro Market
    # Zero momentum, ultra-low volatility, RSI centered at 50
    v_macro = np.zeros(ROUTER_FEATURE_DIM, dtype=np.float32)
    v_macro[0] = 0.004  # very low volatility
    v_macro[1] = 0.001  # flat momentum
    v_macro[2] = 0.000  # flat mean return
    v_macro[3] = 0.00   # RSI exactly 50
    v_macro[9] = 0.1    # 1 low-noise tweet
    v_macro[10] = 1.0
    v_macro[11] = 0.0
    v_macro[12] = 0.35
    v_macro[13] = 0.50
    v_macro[15] = 0.008 # low intraday spread
    regimes["4. Quiet Sideways / Macro Anchored Market"] = v_macro

    results = []
    print(f"\n{'Market Regime':<40} | {'Tech %':<8} | {'Sent %':<8} | {'Human %':<8} | {'Dominant Focus':<20}")
    print("-" * 92)

    for name, vec in regimes.items():
        res = engine.route_single(vec)
        w = res["weights"]
        dom = res["dominant_expert"]
        results.append({
            "regime": name,
            "weights": w,
            "dominant": dom,
            "predicted_dir": res["predicted_direction"],
            "prob": res["ensemble_prob"]
        })
        print(f"{name:<40} | {w['technical']*100:<7.1f}% | {w['sentiment']*100:<7.1f}% | {w['human']*100:<7.1f}% | {dom:<20}")

    print("=" * 92)

    # Automated assertions proving responsiveness
    r_shock = results[0]["weights"]
    r_sent = results[1]["weights"]
    r_notext = results[2]["weights"]
    r_macro = results[3]["weights"]

    # 1. Proven Regime Diversity: Dominant expert changes across regimes
    dominant_experts = set(r["dominant"] for r in results)
    print(f"\n[VERIFICATION PASS 1/3]: Dynamic Dominance: {len(dominant_experts)} distinct experts selected as dominant ({dominant_experts}).")
    assert len(dominant_experts) >= 2, f"FAIL: Expected at least 2 different dominant experts, got {dominant_experts}"

    # 2. Weights across regimes must shift significantly (proves dynamic response)
    tech_shift = max(r["weights"]["technical"] for r in results) - min(r["weights"]["technical"] for r in results)
    human_shift = max(r["weights"]["human"] for r in results) - min(r["weights"]["human"] for r in results)
    sent_shift = max(r["weights"]["sentiment"] for r in results) - min(r["weights"]["sentiment"] for r in results)
    max_shift = max(tech_shift, human_shift, sent_shift)

    print(f"[VERIFICATION PASS 2/3]: Significant dynamic weight shifts across regimes:")
    print(f"  * Technical Expert Weight Range : {min(r['weights']['technical'] for r in results)*100:.1f}% to {max(r['weights']['technical'] for r in results)*100:.1f}% (Delta = {tech_shift*100:.1f}%)")
    print(f"  * Human Proxy Weight Range      : {min(r['weights']['human'] for r in results)*100:.1f}% to {max(r['weights']['human'] for r in results)*100:.1f}% (Delta = {human_shift*100:.1f}%)")
    print(f"  * Sentiment Expert Weight Range : {min(r['weights']['sentiment'] for r in results)*100:.1f}% to {max(r['weights']['sentiment'] for r in results)*100:.1f}% (Delta = {sent_shift*100:.1f}%)")
    assert max_shift >= 0.05, f"FAIL: Max shift across regimes ({max_shift:.4f}) is less than 5%"

    # 3. All allocations sum to 100%
    for r in results:
        total_w = sum(r["weights"].values())
        assert abs(total_w - 1.0) < 1e-4, f"FAIL: Weights do not sum to 1.0 ({total_w})"
    print("[VERIFICATION PASS 3/3]: Simplex constraint strictly satisfied (all weights sum to exactly 100%).")

    print("\n>>> RESULT FOR REQUIREMENT 7: [VERIFIED] (Dynamic Responsiveness Proven) <<<\n")
    return results


if __name__ == "__main__":
    run_responsiveness_verification()
