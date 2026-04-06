"""Merge a LoRA adapter into the base model and save to Modal volume.

Usage:
    modal run scripts/modal/merge_lora.py

Merges Qwen3-32B + madison-qwen3-r2-v1 LoRA adapter into a single model
and saves it to the foundry-merged-models volume. This eliminates the need
for --enable-lora at serving time, which fixes vLLM streaming detokenization
spacing artifacts (see Foundry issue #3).
"""

import modal

MINUTES = 60

merge_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "torch>=2.4",
        "transformers>=4.45",
        "peft>=0.13",
        "huggingface-hub>=0.25",
        "accelerate>=0.34",
        "safetensors>=0.4",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App("foundry-merge-lora")

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
adapter_vol = modal.Volume.from_name("foundry-adapters", create_if_missing=True)
merged_vol = modal.Volume.from_name("foundry-merged-models", create_if_missing=True)

BASE_MODEL = "Qwen/Qwen3-32B"
ADAPTER_NAME = "madison-qwen3-r2-v1"
ADAPTER_PATH = f"/adapters/experiments/{ADAPTER_NAME}"
OUTPUT_PATH = f"/merged/{ADAPTER_NAME}-merged"


@app.function(
    image=merge_image,
    gpu="A100-80GB:1",
    timeout=30 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/adapters": adapter_vol,
        "/merged": merged_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def merge():
    import os
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {ADAPTER_PATH}")
    # List adapter contents for verification
    adapter_files = os.listdir(ADAPTER_PATH)
    print(f"Adapter files: {adapter_files}")

    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {OUTPUT_PATH}")
    model.save_pretrained(OUTPUT_PATH, safe_serialization=True)

    # Save tokenizer alongside the model
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.save_pretrained(OUTPUT_PATH)

    # Commit the volume so the merged model persists
    merged_vol.commit()

    # Verify output
    output_files = os.listdir(OUTPUT_PATH)
    total_size = sum(
        os.path.getsize(os.path.join(OUTPUT_PATH, f))
        for f in output_files
        if os.path.isfile(os.path.join(OUTPUT_PATH, f))
    )
    print(f"\nMerged model saved: {len(output_files)} files, {total_size / 1e9:.1f} GB")
    for f in sorted(output_files):
        fpath = os.path.join(OUTPUT_PATH, f)
        if os.path.isfile(fpath):
            print(f"  {f}: {os.path.getsize(fpath) / 1e6:.1f} MB")

    print("\nDone! Update serve_chamber.py to use the merged model path.")
