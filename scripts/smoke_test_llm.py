"""
scripts/smoke_test_llm.py
Quick smoke test to verify model loading, LoRA wrapping, forward pass, and VRAM footprint on RTX 4050.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def smoke_test():
    print("=" * 60)
    print("RUNNING LLM & GPU SMOKE TEST")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("1. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("2. Loading Base Model in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    print("3. Applying LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"   Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    print("4. Testing Forward + Backward pass...")
    dummy_input = tokenizer("Prediction: UP because momentum is positive", return_tensors="pt").to(device)
    dummy_input["labels"] = dummy_input["input_ids"].clone()
    
    model.train()
    out = model(**dummy_input)
    loss = out.loss
    loss.backward()
    print(f"   Forward loss: {loss.item():.4f}")

    if device == "cuda":
        mem_used = torch.cuda.memory_allocated(0) / (1024**3)
        mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"   GPU Memory Allocated: {mem_used:.2f} GB | Reserved: {mem_reserved:.2f} GB")

    print("=" * 60)
    print("SMOKE TEST: SUCCESSFUL")
    print("=" * 60)

if __name__ == "__main__":
    smoke_test()
