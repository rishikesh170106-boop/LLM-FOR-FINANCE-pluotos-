"""
tests/test_llm_inference.py
Unit tests verifying Phase 7 fine-tuning artifacts and inference functionality.
"""
import pytest
from pathlib import Path
import json

CHECKPOINT_DIR = Path("checkpoints/ploutos_lora")
LOSS_PLOT_PATH = Path("results/figures/training_loss.png")
HISTORY_PATH = Path("results/metrics/training_history.json")

def test_checkpoint_files_exist():
    assert CHECKPOINT_DIR.exists(), "Checkpoint directory does not exist!"
    adapter_file = CHECKPOINT_DIR / "adapter_model.safetensors"
    config_file = CHECKPOINT_DIR / "adapter_config.json"
    assert adapter_file.exists(), "adapter_model.safetensors is missing!"
    assert config_file.exists(), "adapter_config.json is missing!"

def test_loss_plot_generated():
    assert LOSS_PLOT_PATH.exists(), "training_loss.png plot is missing!"
    assert LOSS_PLOT_PATH.stat().st_size > 10000, "Loss plot file is suspiciously small!"

def test_training_loss_converged():
    assert HISTORY_PATH.exists(), "training_history.json is missing!"
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)

    assert "loss" in history and len(history["loss"]) > 0
    initial_loss = history["loss"][0]
    final_loss = history["loss"][-1]
    # Verify training loss decreased
    assert final_loss < initial_loss, f"Loss did not decrease! Initial: {initial_loss}, Final: {final_loss}"

def test_inference_parsing():
    from src.inference.predict import parse_prediction_and_rationale
    sample_output = "Prediction: UP\nRationale: Strong momentum and positive social sentiment."
    direction, rationale = parse_prediction_and_rationale(sample_output)
    assert direction == "UP"
    assert "Strong momentum" in rationale
