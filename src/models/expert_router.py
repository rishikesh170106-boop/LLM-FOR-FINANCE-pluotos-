"""
src/models/expert_router.py
Adaptive Expert Router for Ploutos+:
Implements a Context-Gated Neural Network (MLP) that dynamically weights domain experts
(Technical, Sentiment, Human Proxy) based on a 16-dimensional market context & regime vector.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERT_NAMES = ["technical", "sentiment", "human"]
ROUTER_FEATURE_DIM = 16

ROUTER_FEATURE_NAMES = [
    "roll_return_std_5d",       # 1. 5-day rolling return volatility
    "roll_momentum_5d",         # 2. 5-day cumulative momentum
    "roll_return_mean_5d",       # 3. 5-day mean return
    "lagged_rsi_14_norm",       # 4. 14-day RSI normalized: (RSI - 50) / 50
    "lag_1_return",             # 5. 1-day lagged return
    "lag_2_return",             # 6. 2-day lagged return
    "lag_3_return",             # 7. 3-day lagged return
    "lag_1_log_volume_norm",    # 8. 1-day lagged log volume normalized
    "vol_surge_ratio",          # 9. Volume surge vs lookback average
    "tweet_count_norm",         # 10. Tweet count / 10.0 [0, 1]
    "has_tweets",               # 11. Binary tweet availability flag
    "sentiment_polarity",       # 12. Net tweet sentiment polarity [-1, 1]
    "sentiment_confidence",     # 13. Sentiment expert confidence [0, 1]
    "tech_confidence",          # 14. Technical expert confidence [0, 1]
    "tech_signal_dir",          # 15. Technical signal direction: +1 (Bull), -1 (Bear), 0 (Neut)
    "intraday_dispersion_5d"    # 16. Intraday spread (high_return - low_return)
]


class ContextGatedRouter(nn.Module):
    """
    Context-Gated Neural Network that maps 16-dimensional market context
    to gating weights across 3 domain experts (Technical, Sentiment, Human Proxy).
    """
    def __init__(
        self,
        input_dim: int = ROUTER_FEATURE_DIM,
        hidden_dim: int = 64,
        expert_count: int = 3,
        temperature: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.expert_count = expert_count
        self.temperature = temperature

        # Context Gating Network
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, expert_count)
        )

        # Expert Prediction Heads (each maps domain signals to P(UP))
        self.tech_head = nn.Linear(3, 1)    # momentum, rsi, tech_signal_dir
        self.sent_head = nn.Linear(3, 1)    # polarity, confidence, has_tweets
        self.human_head = nn.Linear(2, 1)   # mean return, market dispersion

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch_size, input_dim)
        Returns:
            weights: Softmax gating weights of shape (batch_size, expert_count)
            ensemble_prob: Combined P(UP) of shape (batch_size, 1)
        """
        # 1. Compute gating logits & temperature-scaled softmax weights
        logits = self.gating_network(x)
        weights = F.softmax(logits / self.temperature, dim=-1)

        # 2. Compute individual expert predictions
        # Tech features: roll_momentum_5d (col 1), lagged_rsi_14_norm (col 3), tech_signal_dir (col 14)
        tech_inputs = torch.stack([x[:, 1], x[:, 3], x[:, 14]], dim=1)
        p_tech = torch.sigmoid(self.tech_head(tech_inputs))

        # Sent features: sentiment_polarity (col 11), sentiment_confidence (col 12), has_tweets (col 10)
        sent_inputs = torch.stack([x[:, 11], x[:, 12], x[:, 10]], dim=1)
        p_sent = torch.sigmoid(self.sent_head(sent_inputs))

        # Human/Macro features: roll_return_mean_5d (col 2), intraday_dispersion_5d (col 15)
        human_inputs = torch.stack([x[:, 2], x[:, 15]], dim=1)
        p_human = torch.sigmoid(self.human_head(human_inputs))

        # 3. Dynamic Mixture of Experts combination
        expert_probs = torch.cat([p_tech, p_sent, p_human], dim=1)  # (batch_size, 3)
        ensemble_prob = torch.sum(weights * expert_probs, dim=1, keepdim=True)  # (batch_size, 1)

        return weights, ensemble_prob


def extract_context_features_matrix(
    df: pd.DataFrame,
    features_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs the 16-dimensional strictly lagged context matrix X and target array y.
    Guarantees zero target leakage and robust imputation of missing values.
    """
    merged = df.merge(features_df, on=["ticker", "date"], how="left")

    n = len(merged)
    X = np.zeros((n, ROUTER_FEATURE_DIM), dtype=np.float32)

    # 1. roll_return_std_5d
    X[:, 0] = merged["roll_return_std_5d"].fillna(0.01).values
    # 2. roll_momentum_5d
    X[:, 1] = merged["roll_momentum_5d"].fillna(0.0).values
    # 3. roll_return_mean_5d
    X[:, 2] = merged["roll_return_mean_5d"].fillna(0.0).values
    # 4. lagged_rsi_14_norm
    rsi = merged["lagged_rsi_14"].fillna(50.0).values
    X[:, 3] = np.clip((rsi - 50.0) / 50.0, -1.0, 1.0)
    # 5. lag_1_return
    X[:, 4] = merged["lag_1_return"].fillna(0.0).values
    # 6. lag_2_return
    X[:, 5] = merged["lag_2_return"].fillna(0.0).values
    # 7. lag_3_return
    X[:, 6] = merged["lag_3_return"].fillna(0.0).values
    # 8. lag_1_log_volume_norm
    log_vol = merged["lag_1_log_volume"].fillna(10.0).values
    X[:, 7] = (log_vol - 10.0) / 5.0
    # 9. vol_surge_ratio
    vol_mean = (merged["lag_1_log_volume"] + merged["lag_2_log_volume"] + merged["lag_3_log_volume"]) / 3.0
    X[:, 8] = np.clip((log_vol - vol_mean.fillna(10.0).values) / 2.0, -2.0, 2.0)
    # 10. tweet_count_norm
    tc = merged["tweet_count"].fillna(0).values
    X[:, 9] = np.clip(tc / 10.0, 0.0, 1.0)
    # 11. has_tweets
    X[:, 10] = (tc > 0).astype(float)

    # 12-13. Sentiment polarity & confidence heuristic from tweet text
    pos_words = {"bull", "call", "gain", "buy", "surge", "higher", "record", "profit", "win"}
    neg_words = {"bear", "put", "loss", "drop", "sell", "fall", "lower", "warning", "crash"}

    sent_polarities = np.zeros(n, dtype=np.float32)
    sent_confs = np.zeros(n, dtype=np.float32)
    for i, text in enumerate(merged["text_evidence"].fillna("").values):
        if not text or text == "[No textual evidence available for this trading day]":
            sent_polarities[i] = 0.0
            sent_confs[i] = 0.0
            continue
        lower_txt = text.lower()
        pos_cnt = sum(lower_txt.count(w) for w in pos_words)
        neg_cnt = sum(lower_txt.count(w) for w in neg_words)
        total_cues = pos_cnt + neg_cnt
        if total_cues > 0:
            sent_polarities[i] = (pos_cnt - neg_cnt) / total_cues
            sent_confs[i] = min(0.9, 0.5 + 0.1 * total_cues)
        else:
            sent_polarities[i] = 0.0
            sent_confs[i] = 0.4

    X[:, 11] = sent_polarities
    X[:, 12] = sent_confs

    # 14. tech_confidence
    # Strong momentum or RSI extremes give higher confidence
    mom = np.abs(X[:, 1])
    rsi_ext = np.abs(X[:, 3])
    X[:, 13] = np.clip(0.5 + mom * 2.0 + rsi_ext * 0.2, 0.5, 0.95)

    # 15. tech_signal_dir (+1 Bullish, -1 Bearish, 0 Neutral)
    tech_dir = np.zeros(n, dtype=np.float32)
    tech_dir[(X[:, 1] > 0.01) | (X[:, 3] < -0.3)] = 1.0
    tech_dir[(X[:, 1] < -0.01) | (X[:, 3] > 0.3)] = -1.0
    X[:, 14] = tech_dir

    # 16. intraday_dispersion_5d
    high_ret = merged["lag_1_high_ret"].fillna(0.01).values
    low_ret = merged["lag_1_low_ret"].fillna(-0.01).values
    X[:, 15] = np.clip(high_ret - low_ret, 0.0, 0.2)

    y = merged["label"].values.astype(np.int64)
    return X, y


class AdaptiveRouterEngine:
    """
    High-level engine for running routing inference, generating routing rationales,
    and injecting dynamic weights into prompt contexts.
    """
    def __init__(self, model_path: Optional[Path] = None, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = ContextGatedRouter().to(self.device)
        self.model_path = model_path

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.is_trained = True
        else:
            self.is_trained = False

    def route_single(self, context_vector: np.ndarray) -> Dict[str, Any]:
        """
        Routes a single 16-d observation and returns structured weights and dominant expert.
        """
        self.model.eval()
        x_tensor = torch.tensor(context_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            weights, ensemble_prob = self.model(x_tensor)

        w = weights.cpu().numpy()[0]
        prob = float(ensemble_prob.cpu().numpy()[0, 0])

        w_dict = {
            "technical": round(float(w[0]), 4),
            "sentiment": round(float(w[1]), 4),
            "human": round(float(w[2]), 4)
        }

        # Determine dominant expert
        dominant_idx = int(np.argmax(w))
        dominant_name = ["Technical Expert", "Sentiment Expert", "Human/Sector Proxy"][dominant_idx]

        # Generate routing explanation
        if dominant_idx == 0:
            reason = f"Technical Expert dominant ({w[0]*100:.1f}%) due to active momentum and RSI technical boundaries."
        elif dominant_idx == 1:
            reason = f"Sentiment Expert prioritized ({w[1]*100:.1f}%) due to prominent social media polarity signals."
        else:
            reason = f"Fundamental/Sector Proxy prioritized ({w[2]*100:.1f}%) providing macro anchoring amid low noise."

        return {
            "weights": w_dict,
            "weights_array": w,
            "dominant_expert": dominant_name,
            "dominant_index": dominant_idx,
            "ensemble_prob": round(prob, 4),
            "predicted_label": 1 if prob >= 0.5 else 0,
            "predicted_direction": "UP" if prob >= 0.5 else "DOWN",
            "routing_reason": reason
        }

    def format_router_prompt_header(self, routing_res: Dict[str, Any]) -> str:
        """Constructs the adaptive routing prompt header to inject into PloutosGen prompts."""
        w = routing_res["weights"]
        header_lines = [
            "--- [ Ploutos+ Adaptive Expert Router Allocation ] ---",
            f"Dynamic Expert Weights: Technical: {w['technical']*100:.1f}% | Sentiment: {w['sentiment']*100:.1f}% | Fundamental: {w['human']*100:.1f}%",
            f"Primary Focus: {routing_res['dominant_expert']} ({routing_res['routing_reason']})",
            ""
        ]
        return "\n".join(header_lines)
