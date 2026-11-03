"""
src/training/train_llm.py
Fine-tunes Qwen2.5-1.5B-Instruct using LoRA on Ploutos expert financial instruction data.
Optimized for NVIDIA RTX 4050 (6GB VRAM) on Windows:
- FP16 mixed precision
- LoRA on attention projection layers (r=16, alpha=32)
- Proper label masking (-100 on pad tokens)
- Real-time loss logging and plot generation
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import torch
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_PROMPTS_PATH = Path("data/processed/ploutos_train_prompts.jsonl")
VAL_PROMPTS_PATH = Path("data/processed/ploutos_val_prompts.jsonl")
CHECKPOINT_DIR = Path("checkpoints/ploutos_lora")
FIGURES_DIR = Path("results/figures")
METRICS_DIR = Path("results/metrics")

def load_prompt_dataset(jsonl_path: Path, max_samples: int = 600) -> List[Dict[str, Any]]:
    """Loads and samples from the pre-generated Ploutos prompt JSONL."""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                samples.append(json.loads(line_str))
                if len(samples) >= max_samples:
                    break
    return samples

def train_ploutos_lora(
    max_train_samples: int = 500,
    max_val_samples: int = 100,
    epochs: int = 2,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_seq_len: int = 448
):
    print("=" * 70)
    print("PLOUTOS+ PHASE 7: LLM SUPERVISED FINE-TUNING (LoRA)")
    print("=" * 70)

    # 1. Device check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] Training Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU: {gpu_name} ({vram_gb:.1f} GB VRAM detected)")
        torch.cuda.empty_cache()

    # 2. Imports from transformers & peft
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    # 3. Load Tokenizer & Base Model
    print(f"\n[Model] Loading base model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    # Enable gradient checkpointing to conserve VRAM
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 4. Configure LoRA
    print("[LoRA] Configuring Low-Rank Adaptation (r=16, alpha=32)...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)
    trainable_params, all_params = model.get_nb_trainable_parameters()
    print(f"  Trainable Parameters: {trainable_params:,} / {all_params:,} ({trainable_params/all_params*100:.2f}%)")

    # 5. Load and format datasets
    print(f"\n[Data] Loading instruction prompt samples...")
    train_raw = load_prompt_dataset(TRAIN_PROMPTS_PATH, max_samples=max_train_samples)
    val_raw = load_prompt_dataset(VAL_PROMPTS_PATH, max_samples=max_val_samples)
    print(f"  Loaded {len(train_raw):,} training samples, {len(val_raw):,} validation samples.")

    def format_sample(item: Dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": item["system"]},
            {"role": "user", "content": f"{item['instruction']}\n\n{item['input']}"},
            {"role": "assistant", "content": item["output"]}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    train_texts = [format_sample(x) for x in train_raw]
    val_texts = [format_sample(x) for x in val_raw]

    # Tokenize
    print("[Data] Tokenizing sequences...")
    train_encodings = tokenizer(train_texts, max_length=max_seq_len, truncation=True, padding=True, return_tensors="pt")
    val_encodings = tokenizer(val_texts, max_length=max_seq_len, truncation=True, padding=True, return_tensors="pt")

    # Custom Torch Dataset with pad token label masking (-100)
    class PromptDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, pad_id):
            self.input_ids = encodings["input_ids"]
            self.attention_mask = encodings["attention_mask"]
            self.labels = encodings["input_ids"].clone()
            self.labels[self.labels == pad_id] = -100

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx]
            }

    train_loader = torch.utils.data.DataLoader(
        PromptDataset(train_encodings, tokenizer.pad_token_id),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        PromptDataset(val_encodings, tokenizer.pad_token_id),
        batch_size=batch_size,
        shuffle=False
    )

    # 6. Training Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    print(f"\n[Training] Beginning fine-tuning: {epochs} Epochs, {total_steps} effective optimization steps...")
    history = {"step": [], "loss": [], "epoch": [], "val_loss": []}
    global_step = 0
    start_time = time.time()

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            epoch_loss += outputs.loss.item()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = outputs.loss.item()
                history["step"].append(global_step)
                history["loss"].append(round(avg_loss, 4))
                history["epoch"].append(epoch)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step})

        # Epoch Validation Loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for v_batch in val_loader:
                v_in = v_batch["input_ids"].to(device)
                v_mask = v_batch["attention_mask"].to(device)
                v_lab = v_batch["labels"].to(device)
                v_out = model(input_ids=v_in, attention_mask=v_mask, labels=v_lab)
                val_losses.append(v_out.loss.item())

        mean_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        history["val_loss"].append({"epoch": epoch, "val_loss": round(mean_val_loss, 4)})
        print(f"\n  Epoch {epoch} Finished | Mean Train Loss: {epoch_loss / len(train_loader):.4f} | Val Loss: {mean_val_loss:.4f}")
        model.train()

    elapsed_min = (time.time() - start_time) / 60
    print(f"\n[Completed] Fine-tuning completed in {elapsed_min:.2f} minutes.")

    # 7. Save Model & Checkpoints
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Saving] Saving LoRA adapter and tokenizer to: {CHECKPOINT_DIR}...")
    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)

    # Save metrics history
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # 8. Plot Training Loss Curve
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = FIGURES_DIR / "training_loss.png"
    plt.figure(figsize=(10, 5), dpi=200)
    plt.plot(history["step"], history["loss"], label="Training Loss", color="#1f77b4", lw=2)
    plt.title("Ploutos+ LoRA Fine-Tuning Loss Curve (Qwen2.5-1.5B)", fontsize=13, fontweight="bold")
    plt.xlabel("Optimization Steps", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[Figures] Saved training loss plot to: {plot_path}")

    print("=" * 70)
    print("PHASE 7 FINE-TUNING EXECUTION: SUCCESS")
    print("=" * 70)
    return history

if __name__ == "__main__":
    train_ploutos_lora()
