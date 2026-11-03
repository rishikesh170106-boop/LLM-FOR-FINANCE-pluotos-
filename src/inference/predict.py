"""
src/inference/predict.py
Interactive CLI inference script for Ploutos+ fine-tuned model:
Loads base model + LoRA adapter and generates stock movement predictions with natural language rationales.
Usage:
  python src/inference/predict.py --sample
  python src/inference/predict.py --ticker AAPL --date 2015-10-15
"""
import os
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
import json
import argparse
import torch
import pandas as pd

from src.experts.prompt_builder import PloutosPromptBuilder
from src.experts.visual_expert import VisualFinancialExpert
from src.features.technical_features import extract_all_technical_features
from src.models.expert_router import AdaptiveRouterEngine, ROUTER_FEATURE_DIM

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
CHECKPOINT_DIR = Path("checkpoints/ploutos_lora")
ROUTER_CHECKPOINT = Path("checkpoints/expert_router/best_router.pt")
TEST_PROMPTS_PATH = Path("data/processed/ploutos_test_prompts.jsonl")

def load_inference_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("[Inference] Loading model and tokenizer...")
    tokenizer_path = str(CHECKPOINT_DIR) if CHECKPOINT_DIR.exists() else MODEL_ID
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=True
        )
    except Exception:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

    if CHECKPOINT_DIR.exists():
        print(f"[Inference] Attaching trained LoRA adapter from {CHECKPOINT_DIR}...")
        model = PeftModel.from_pretrained(base_model, str(CHECKPOINT_DIR))
    else:
        print("[Inference] Notice: Trained adapter not found. Using base instruction model.")
        model = base_model

    model.eval()
    return model, tokenizer, device

def parse_prediction_and_rationale(output_text: str):
    """Parses direction and explanation from the model generation."""
    pred_match = re.search(r'Prediction:\s*(UP|DOWN)', output_text, re.IGNORECASE)
    direction = pred_match.group(1).upper() if pred_match else "UNKNOWN"

    rationale_match = re.search(r'Rationale:\s*(.*)', output_text, re.IGNORECASE | re.DOTALL)
    rationale = rationale_match.group(1).strip() if rationale_match else output_text.strip()
    return direction, rationale

def run_prediction(ticker: str = None, date: str = None, use_sample: bool = True):
    # 1. Load test sample
    target_sample = None
    if use_sample or (not ticker or not date):
        if TEST_PROMPTS_PATH.exists():
            with open(TEST_PROMPTS_PATH, "r", encoding="utf-8") as f:
                # Grab a sample from the test split
                for line in f:
                    item = json.loads(line)
                    if item["ticker"] == (ticker or "AAPL"):
                        target_sample = item
                        break
                if not target_sample:
                    f.seek(0)
                    target_sample = json.loads(f.readline())

    if not target_sample:
        print("ERROR: Test prompts not found or sample could not be resolved.")
        return

    sample_ticker = target_sample["ticker"]
    sample_date = target_sample["date"]
    ground_truth = target_sample["label"]
    true_direction = "UP" if ground_truth == 1 else "DOWN"

    print("\n" + "=" * 70)
    print(f"PLOUTOS+ LIVE FINANCIAL PREDICTION DEMO")
    print("=" * 70)
    print(f"Ticker: ${sample_ticker}  |  Target Date: {sample_date}  |  Ground Truth Label: {true_direction}")
    print("-" * 70)
    print("EVIDENCE PASSED TO LLM:")
    print(target_sample["input"])
    print("-" * 70)

    # 1b. Compute Adaptive Router Allocation (Phase 8 Extension)
    routing_info = None
    router_header = ""
    guided_instruction = target_sample["instruction"]
    if ROUTER_CHECKPOINT.exists():
        router_engine = AdaptiveRouterEngine(model_path=ROUTER_CHECKPOINT, device="cpu")
        
        # Extract authentic context vector for this ticker/date if available
        context_vector = np.zeros(ROUTER_FEATURE_DIM, dtype=np.float32)
        try:
            test_csv_path = Path("data/processed/test_with_text.csv")
            if test_csv_path.exists():
                test_df = pd.read_csv(test_csv_path)
                match = test_df[(test_df["ticker"] == sample_ticker) & (test_df["date"].astype(str) == str(sample_date))]
                if not match.empty:
                    feats = extract_all_technical_features([sample_ticker])
                    from src.models.expert_router import extract_context_features_matrix
                    X_ctx, _ = extract_context_features_matrix(match, feats)
                    if len(X_ctx) > 0:
                        context_vector = X_ctx[0]
        except Exception as e:
            pass

        routing_info = router_engine.route_single(context_vector)
        router_header = router_engine.format_router_prompt_header(routing_info)
        w = routing_info["weights"]

        print("--- [ Ploutos+ Phase 8 Adaptive Router Allocation ] ---")
        print(f"Dynamic Weights: Tech: {w['technical']*100:.1f}% | Sent: {w['sentiment']*100:.1f}% | Human: {w['human']*100:.1f}%")
        print(f"Dominant Focus : {routing_info['dominant_expert']}")
        print(f"Router Rationale: {routing_info['routing_reason']}")
        print("-" * 70)

        guided_instruction += (
            f"\n\n[ADAPTIVE ROUTER GUIDANCE]:\n"
            f"The Adaptive Router has evaluated prevailing market conditions and allocated the following expert weights:\n"
            f"- Technical Expert: {w['technical']*100:.1f}%\n"
            f"- Sentiment Expert: {w['sentiment']*100:.1f}%\n"
            f"- Human/Fundamental Proxy: {w['human']*100:.1f}%\n"
            f"Dominant Priority: {routing_info['dominant_expert']} ({routing_info['routing_reason']}).\n"
            f"You MUST condition your directional movement prediction and explanation on these dynamic weights, "
            f"giving primary consideration to {routing_info['dominant_expert']}."
        )

    # 1c. Query Visual Financial Expert (Phase 9 Multimodal Extension)
    visual_expert = VisualFinancialExpert()
    visual_res = visual_expert.analyze(sample_ticker, sample_date, generate_plot=True)
    visual_evidence_block = (
        f"\n\n--- [4. Visual Chart Expert Evidence] ---\n"
        f"Signal: {visual_res['signal']} (Confidence: {visual_res['confidence']:.2f})\n"
        f"Evidence: {visual_res['evidence']}"
    )
    if visual_res["chart_path"]:
        print("--- [ Phase 9 Visual Financial Expert ] ---")
        print(f"Chart Signal   : {visual_res['signal']} (Confidence: {visual_res['confidence']:.2f})")
        print(f"Chart Patterns : {', '.join(visual_res['detected_patterns'])}")
        print(f"Saved PNG Chart: {visual_res['chart_path']}")
        print("-" * 70)

        guided_instruction += (
            f"\n\n[VISUAL FINANCIAL EXPERT GUIDANCE]:\n"
            f"The Visual Financial Expert analyzed the 15-day candlestick chart and identified:\n"
            f"- Candlestick Structure Signal: {visual_res['signal']} (Confidence: {visual_res['confidence']:.2f})\n"
            f"- Detected Formations: {', '.join(visual_res['detected_patterns'])}\n"
            f"You MUST synthesize this visual chart evidence into your rationale, explaining how candlestick patterns "
            f"and support/resistance levels confirm or challenge the other signals."
        )

    # 2. Load model
    model, tokenizer, device = load_inference_model()

    # 3. Format prompt for generation with active router and visual chart influence
    prompt_payload = f"{guided_instruction}\n\n{router_header}{target_sample['input']}{visual_evidence_block}"
    messages = [
        {"role": "system", "content": target_sample["system"]},
        {"role": "user", "content": prompt_payload}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    # 4. Generate
    print("\n[Generating Financial Prediction & Rationale]...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only new tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    pred_dir, rationale = parse_prediction_and_rationale(response_text)
    is_correct = (pred_dir == true_direction)

    print("\n" + "=" * 70)
    print(f">>> MODEL PREDICTION: {pred_dir} <<<")
    print(f"Ground Truth Outcome: {true_direction}  |  Status: {'[CORRECT]' if is_correct else '[INCORRECT]'}")
    print(f"\nGenerated Rationale:")
    print(f"\"{rationale}\"")
    print("=" * 70 + "\n")

    return {
        "ticker": sample_ticker,
        "date": sample_date,
        "prediction": pred_dir,
        "ground_truth": true_direction,
        "is_correct": is_correct,
        "rationale": rationale
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ploutos+ Live Inference CLI")
    parser.add_argument("--ticker", type=str, default=None, help="Stock ticker (e.g. AAPL)")
    parser.add_argument("--date", type=str, default=None, help="Prediction date (YYYY-MM-DD)")
    parser.add_argument("--sample", action="store_true", help="Run on a sample test observation")
    args = parser.parse_args()

    run_prediction(ticker=args.ticker, date=args.date, use_sample=args.sample)
