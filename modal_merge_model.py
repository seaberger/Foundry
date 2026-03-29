"""Merge LoRA adapter into base model and save as 16-bit for vLLM serving.

Uses Unsloth to load (matches training quantization), then saves a merged
16-bit model that can be served by vLLM without Unsloth.

Usage:
    modal run modal_merge_model.py --adapter madison-orpo-v3b-lr2e5
"""

from __future__ import annotations

import modal

MINUTES = 60

app = modal.App("foundry-merge-model")

model_cache_vol = modal.Volume.from_name("foundry-model-cache", create_if_missing=True)
adapter_vol = modal.Volume.from_name("foundry-adapters", create_if_missing=True)

# No Unsloth — vanilla transformers >= 4.54.1 (fixes Gemma 3 save + cache bugs)
merge_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate>=1.9.0",
        "bitsandbytes>=0.45",
        "huggingface_hub>=0.34.2",
        "peft>=0.16.0",
        "transformers>=4.54.1",
        "torch>=2.6",
    )
    .env({"HF_HOME": "/model_cache"})
)


@app.function(
    image=merge_image,
    gpu="A100-80GB",  # Need 80GB for dequantizing 27B from 4-bit to 16-bit
    timeout=60 * MINUTES,
    volumes={
        "/model_cache": model_cache_vol,
        "/adapters": adapter_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def merge_adapter(
    adapter_name: str = "madison-orpo-v3b-lr2e5",
    base_model: str = "google/gemma-3-27b-it",
):
    """Load base + adapter, dequantize, merge, save 16-bit to volume."""
    from pathlib import Path

    adapter_path = f"/adapters/experiments/{adapter_name}"
    output_path = f"/adapters/merged/{adapter_name}-16bit"

    # Check if already merged
    if Path(output_path).exists() and any(Path(output_path).glob("*.safetensors")):
        print(f"Merged model already exists at {output_path}")
        return output_path

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Step 1: Load base model in 4-bit (vanilla transformers, no Unsloth)
    print(f"Loading {base_model} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Step 2: Load LoRA adapter
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    print(f"  PeftModel type: {type(model).__name__}")

    # Step 3: Merge LoRA into base, then dequantize to saveable 16-bit
    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    print(f"  Merged type: {type(model).__name__}")
    print("Dequantizing 4-bit → 16-bit on GPU...")
    model = model.dequantize()
    print(f"  Merged type: {type(model).__name__}")

    # Step 4: Save merged model as standard HF format
    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving merged 16-bit model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    adapter_vol.commit()

    # Verify
    saved_files = list(Path(output_path).iterdir())
    total_size = sum(f.stat().st_size for f in saved_files if f.is_file()) / 1024 / 1024 / 1024
    print(f"\nMerged model saved: {output_path}")
    print(f"  {len(saved_files)} files, {total_size:.1f} GB total")
    for f in sorted(saved_files):
        size_mb = f.stat().st_size / 1024 / 1024
        if size_mb > 1:
            print(f"  {f.name}: {size_mb:.0f} MB")

    return output_path


@app.local_entrypoint()
def main(
    adapter: str = "madison-orpo-v3b-lr2e5",
    base: str = "google/gemma-3-27b-it",
):
    """Merge adapter and save 16-bit model to volume."""
    output = merge_adapter.remote(adapter_name=adapter, base_model=base)
    print(f"\nMerged model at: {output}")
    print("Ready for vLLM serving.")
