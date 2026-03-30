"""Merge Qwen 3-32B LoRA adapter into base model and save as 16-bit.

Uses Unsloth for model loading (matches training environment) to avoid
GLIBC/bitsandbytes compatibility issues with vanilla transformers.

Usage:
    modal run modal_merge_model_qwen.py --adapter madison-qwen3-v2
"""

from __future__ import annotations

import modal

MINUTES = 60

app = modal.App("foundry-qwen3-merge")

model_cache_vol = modal.Volume.from_name("foundry-model-cache", create_if_missing=True)
adapter_vol = modal.Volume.from_name("foundry-adapters", create_if_missing=True)

# Use the same Unsloth training image — guaranteed compatible CUDA/GLIBC
merge_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.9.0",
        "huggingface_hub==0.34.2",
        "peft==0.16.0",
        "transformers==4.54.0",
        "trl==0.19.1",
        "unsloth[cu128-torch270]==2025.7.8",
        "unsloth_zoo==2025.7.10",
    )
    .env({"HF_HOME": "/model_cache"})
)

with merge_image.imports():
    import unsloth  # noqa: F401,I001
    from unsloth import FastLanguageModel


@app.function(
    image=merge_image,
    gpu="A100-80GB",
    timeout=60 * MINUTES,
    volumes={
        "/model_cache": model_cache_vol,
        "/adapters": adapter_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def merge_adapter(
    adapter_name: str = "madison-qwen3-v2",
    base_model: str = "Qwen/Qwen3-32B",
    max_seq_length: int = 2048,
):
    """Load base + adapter via Unsloth, merge, save 16-bit to volume."""
    from pathlib import Path

    from peft import PeftModel

    adapter_path = f"/adapters/experiments/{adapter_name}"
    output_path = f"/adapters/merged/{adapter_name}-16bit"

    if Path(output_path).exists() and any(Path(output_path).glob("*.safetensors")):
        print(f"Merged model already exists at {output_path}")
        return output_path

    print(f"Loading {base_model} via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model_cache_vol.commit()

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    print(f"  PeftModel type: {type(model).__name__}")

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    print(f"  Merged type: {type(model).__name__}")

    print("Dequantizing 4-bit → 16-bit on GPU...")
    model = model.dequantize()
    print(f"  Dequantized type: {type(model).__name__}")

    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving merged 16-bit model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    adapter_vol.commit()

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
    adapter: str = "madison-qwen3-v2",
    base: str = "Qwen/Qwen3-32B",
):
    """Merge adapter and save 16-bit model to volume."""
    output = merge_adapter.remote(adapter_name=adapter, base_model=base)
    print(f"\nMerged model at: {output}")
    print("Ready for SFT or GGUF conversion.")
