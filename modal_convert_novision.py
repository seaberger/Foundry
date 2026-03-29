"""Convert merged Gemma 3 VLM model to text-only Gemma3ForCausalLM format.

The merged fine-tuned model has `language_model.` prefixed weight keys, which
forces vLLM to load Gemma3ForConditionalGeneration (VLM class). This triggers
Gemma3Processor initialization which crashes on `image_token_id`.

This script strips the VLM wrapper:
  1. Renames weight keys: strip `language_model.` prefix
  2. Drops vision weights (`vision_model.*`, `multi_modal_projector.*`)
  3. Flattens `text_config` into top-level config.json
  4. Sets architecture to Gemma3ForCausalLM
  5. Copies only tokenizer files (not processor files)

Based on gghfez/gemma-3-27b-novision approach.

Usage:
    modal run modal_convert_novision.py
    modal run modal_convert_novision.py --input /adapters/merged/madison-orpo-v4-16bit
    modal run modal_convert_novision.py --output-name madison-orpo-v4-novision
"""

from __future__ import annotations

import modal

MINUTES = 60

app = modal.App("foundry-novision-convert")

adapter_vol = modal.Volume.from_name("foundry-adapters", create_if_missing=True)
model_cache_vol = modal.Volume.from_name("foundry-model-cache", create_if_missing=True)

convert_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "safetensors",
        "torch",
        "transformers>=4.54.0",
        "huggingface_hub",
        "sentencepiece",
    )
    .env({"HF_HOME": "/model_cache"})
)


@app.function(
    image=convert_image,
    volumes={
        "/adapters": adapter_vol,
        "/model_cache": model_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=65536,  # 64GB RAM for loading safetensors
    timeout=30 * MINUTES,
    cpu=8,
)
def convert_to_novision(
    input_path: str = "/adapters/merged/madison-orpo-v4-16bit",
    output_name: str = "madison-orpo-v4-novision",
):
    """Convert VLM-format merged model to text-only Gemma3ForCausalLM."""
    import glob
    import json
    import os
    import shutil
    from pathlib import Path

    from safetensors.torch import load_file, save_file

    input_dir = Path(input_path)
    output_dir = Path(f"/adapters/merged/{output_name}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input model not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {input_dir} → {output_dir}")
    print(f"Input files:")
    for f in sorted(input_dir.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name} ({size_mb:.1f} MB)")

    # Step 1: Convert weight keys in safetensors
    shard_files = sorted(glob.glob(str(input_dir / "*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors files in {input_dir}")

    total_renamed = 0
    total_dropped = 0
    weight_map = {}  # for index.json

    for shard_path in shard_files:
        shard_name = os.path.basename(shard_path)
        print(f"\nProcessing {shard_name}...")

        tensors = load_file(shard_path)
        new_tensors = {}

        for key, value in tensors.items():
            if any(key.startswith(p) for p in (
                "vision_model.", "multi_modal_projector.", "vision_tower.",
                "visual_projection.", "image_projection.",
            )):
                total_dropped += 1
                continue
            elif key.startswith("language_model."):
                new_key = key.replace("language_model.", "", 1)
                new_tensors[new_key] = value
                weight_map[new_key] = shard_name
                total_renamed += 1
            else:
                new_tensors[key] = value
                weight_map[key] = shard_name

        out_shard = output_dir / shard_name
        save_file(new_tensors, str(out_shard))
        print(f"  Saved {len(new_tensors)} tensors ({total_renamed} renamed, {total_dropped} vision dropped)")

    # Step 2: Write model.safetensors.index.json
    index = {
        "metadata": {"total_size": sum(
            os.path.getsize(str(output_dir / f)) for f in os.listdir(output_dir)
            if f.endswith(".safetensors")
        )},
        "weight_map": weight_map,
    }
    index_path = output_dir / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index, indent=2))
    print(f"\nWrote weight index: {len(weight_map)} keys")

    # Step 3: Patch config.json — flatten text_config, set CausalLM architecture
    config_path = input_dir / "config.json"
    config = json.loads(config_path.read_text())

    text_config = config.get("text_config", {})

    # Start from text_config, override with CausalLM specifics
    new_config = dict(text_config)
    new_config["architectures"] = ["Gemma3ForCausalLM"]
    new_config["model_type"] = "gemma3_text"

    # Remove vision-related keys
    for key in ["vision_config", "mm_tokens_per_image", "image_token_index",
                "boi_token_index", "eoi_token_index"]:
        new_config.pop(key, None)
        config.pop(key, None)

    # Ensure rope_scaling is present (not rope_parameters)
    if "rope_parameters" in new_config and "rope_scaling" not in new_config:
        new_config["rope_scaling"] = {"rope_type": "linear", "factor": 8.0}
        del new_config["rope_parameters"]

    out_config = output_dir / "config.json"
    out_config.write_text(json.dumps(new_config, indent=2))
    print(f"Wrote config.json (architecture: Gemma3ForCausalLM)")

    # Step 4: Copy tokenizer files only (NOT processor files)
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "tokenizer.model",
    ]

    for fname in tokenizer_files:
        src = input_dir / fname
        if src.exists():
            shutil.copy(str(src), str(output_dir / fname))
            print(f"Copied {fname}")
        else:
            print(f"Missing {fname} — will try downloading from google/gemma-3-27b-it")

    # Download any missing tokenizer files from HF
    for fname in tokenizer_files:
        if not (output_dir / fname).exists():
            try:
                from huggingface_hub import hf_hub_download
                local = hf_hub_download("google/gemma-3-27b-it", fname)
                shutil.copy(local, str(output_dir / fname))
                print(f"Downloaded {fname} from HF")
            except Exception as e:
                print(f"WARNING: Could not get {fname}: {e}")

    # Step 5: Write generation_config.json if present
    gen_config = input_dir / "generation_config.json"
    if gen_config.exists():
        shutil.copy(str(gen_config), str(output_dir / "generation_config.json"))
        print("Copied generation_config.json")

    # Commit to volume
    adapter_vol.commit()

    # Summary
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Input:   {input_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Renamed: {total_renamed} weight keys (stripped language_model. prefix)")
    print(f"  Dropped: {total_dropped} vision weights")
    print(f"  Config:  Gemma3ForCausalLM (text_config flattened)")
    print(f"{'='*60}")

    # List output files
    print(f"\nOutput files:")
    total_size = 0
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        total_size += size_mb
        print(f"  {f.name} ({size_mb:.1f} MB)")
    print(f"  Total: {total_size / 1024:.1f} GB")

    return {"output_path": str(output_dir), "renamed": total_renamed, "dropped": total_dropped}


@app.local_entrypoint()
def main(
    input: str = "/adapters/merged/madison-orpo-v4-16bit",
    output_name: str = "madison-orpo-v4-novision",
):
    """Convert merged Gemma 3 model to text-only format for vLLM."""
    print(f"Converting {input} → {output_name}")
    result = convert_to_novision.remote(input_path=input, output_name=output_name)
    print(f"\nDone: {result}")
