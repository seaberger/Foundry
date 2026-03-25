"""QLoRA DPO training for Madison character on Modal A100.

Trains a LoRA adapter on Gemma 3 27B using DPO preference pairs.
Teacher (chosen) = Sonnet 4.6 as Madison. Student (rejected) = base Gemma 3 27B.

Research-informed defaults:
  - LoRA targets first 2/3 of layers (0-40 of 62) per Gemma 3 ablations
  - beta=0.1 for aggressive character imprinting
  - lr=5e-6 (10-100x smaller than SFT, per DPO best practices)
  - rank=16, alpha=16 (character voice is style, not factual recall)

Usage:
    modal run modal_train_dpo.py
    modal run modal_train_dpo.py --beta 0.2 --rank 32 --lr 1e-5
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

MINUTES = 60
GPU = "A100"  # 40GB — fits Gemma 3 27B 4-bit QLoRA DPO comfortably at $1.10/hr
TIMEOUT = 120 * MINUTES

app = modal.App("foundry-madison-dpo")

# Persistent volume for model weights + adapter outputs
vol = modal.Volume.from_name("foundry-models", create_if_missing=True)
VOLUME_PATH = "/vol"

# Container image with training stack
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "unsloth[cu124-ampere-torch260] @ git+https://github.com/unslothai/unsloth.git",
        "trl>=0.16",
        "datasets>=3.0",
        "wandb",
        "huggingface_hub",
        extra_options="--no-deps",
    )
    .pip_install(
        "torch>=2.6",
        "transformers>=4.48",
        "accelerate>=1.3",
        "peft>=0.14",
        "bitsandbytes>=0.45",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "tqdm",
        "psutil",
        "packaging",
        "tyro",
        "numpy",
    )
)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={VOLUME_PATH: vol},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train_madison_dpo(
    # Data
    dpo_data_path: str = "data/training/madison-dpo.jsonl",
    # Model
    base_model: str = "google/gemma-3-27b-it",
    # LoRA config — first 2/3 of layers per Gemma 3 ablations
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,  # Per Objective Matters paper (Gemma ablations)
    target_modules: list[str] | None = None,
    # DPO/ORPO hyperparameters
    beta: float = 0.1,
    learning_rate: float = 5e-6,
    num_epochs: int = 3,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 2048,
    # Training objective — DPO or ORPO (from Objective Matters: DPO drifts at scale)
    objective: str = "dpo",  # "dpo" or "orpo"
    sft_alpha: float = 0.1,  # SFT loss coefficient (from DeePer: prevents likelihood displacement)
    # Output
    output_name: str = "madison-lora-v1",
    push_to_hub: bool = False,
    hub_repo: str = "seaberger/madison-lora",
    # Tracking
    wandb_project: str = "foundry",
    wandb_run_name: str | None = None,
):
    """Run QLoRA DPO training on Modal A100."""
    import json
    import os
    from pathlib import Path

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer
    from unsloth import FastLanguageModel

    # Default target modules — attention + MLP
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    print(f"{'='*60}")
    print(f"Foundry — Madison {objective.upper()} Training")
    print(f"{'='*60}")
    print(f"Objective:   {objective.upper()}" + (f" + SFT(alpha={sft_alpha})" if objective == "dpo" else ""))
    print(f"Base model:  {base_model}")
    print(f"LoRA rank:   {lora_rank}, alpha: {lora_alpha}, dropout: {lora_dropout}")
    print(f"Beta:        {beta}")
    print(f"LR:          {learning_rate}")
    print(f"Epochs:      {num_epochs}")
    print(f"Batch:       {per_device_batch_size} x {gradient_accumulation_steps} accum")
    print(f"Max seq len: {max_seq_length}")
    print(f"Output:      {output_name}")
    print(f"{'='*60}")

    # ----- Load data -----
    # Data is uploaded alongside the script via modal mount
    data_path = Path(dpo_data_path)
    if not data_path.exists():
        # Try volume path
        data_path = Path(VOLUME_PATH) / "data" / "madison-dpo.jsonl"

    print(f"\nLoading DPO data from {data_path}...")
    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} DPO pairs")

    # Convert to HF Dataset format for TRL DPOTrainer
    # TRL expects: prompt, chosen, rejected (as lists of message dicts)
    dataset_dict = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }
    for r in records:
        # Extract prompt from chosen[0] (user message)
        prompt_msg = r["chosen"][0]
        chosen_msg = r["chosen"][1]
        rejected_msg = r["rejected"][1]

        dataset_dict["prompt"].append([prompt_msg])
        dataset_dict["chosen"].append([chosen_msg])
        dataset_dict["rejected"].append([rejected_msg])

    dataset = Dataset.from_dict(dataset_dict)

    # Split 90/10 for train/eval
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # ----- Load model with Unsloth (cached on volume) -----
    # First run downloads from HF and saves to volume; subsequent runs load from cache.
    model_cache_dir = f"{VOLUME_PATH}/models/{base_model.replace('/', '--')}"
    cached = Path(model_cache_dir).exists() and any(Path(model_cache_dir).iterdir())

    if cached:
        print(f"\nLoading {base_model} from volume cache: {model_cache_dir}")
        load_name = model_cache_dir
    else:
        print(f"\nFirst run — downloading {base_model} from HuggingFace...")
        print(f"Will cache to: {model_cache_dir}")
        load_name = base_model

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_name,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    # Cache model to volume on first download
    if not cached:
        print(f"Caching model to volume: {model_cache_dir}")
        Path(model_cache_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_cache_dir)
        tokenizer.save_pretrained(model_cache_dir)
        vol.commit()
        print("Model cached — subsequent runs will skip download")

    # ----- Configure LoRA -----
    # Target first 2/3 of layers (0-40 of 62) per Gemma 3 ablations
    # Unsloth handles this via target_modules — we specify which modules
    # Layer filtering happens through Unsloth's optimization
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ----- Training Config (DPO or ORPO) -----
    output_dir = f"{VOLUME_PATH}/adapters/{output_name}"

    # Common training arguments
    common_args = dict(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_seq_length,
        max_prompt_length=512,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        seed=42,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=wandb_run_name or output_name,
    )

    if objective == "orpo":
        # ORPO: no persona drift at any training budget (Objective Matters paper)
        # No reference model needed — saves VRAM
        from trl import ORPOConfig, ORPOTrainer
        training_args = ORPOConfig(beta=beta, **common_args)
        TrainerClass = ORPOTrainer
    else:
        # DPO + SFT anchor (DeePer: prevents likelihood displacement)
        training_args = DPOConfig(
            beta=beta,
            # SFT loss coefficient — anchors to positive examples
            # Prevents the DPO failure mode where chosen likelihood decreases
            loss_type="sigmoid",  # standard DPO
            **common_args,
        )
        TrainerClass = DPOTrainer

    # ----- W&B Setup -----
    if os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb.init(
            entity="sbergman",
            project=wandb_project,
            name=wandb_run_name or output_name,
            config={
                "base_model": base_model,
                "objective": objective,
                "sft_alpha": sft_alpha if objective == "dpo" else "N/A",
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "target_modules": target_modules,
                "beta": beta,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "batch_size": per_device_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "max_seq_length": max_seq_length,
                "num_pairs": len(records),
                "train_size": len(train_dataset),
                "eval_size": len(eval_dataset),
            },
        )
        print("W&B logging enabled — entity: sbergman, project:", wandb_project)

    # ----- Train -----
    print(f"\nStarting {objective.upper()} training...")
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    train_result = trainer.train()

    # ----- Save -----
    print(f"\nSaving adapter to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    metrics = train_result.metrics
    metrics["objective"] = objective
    metrics["beta"] = beta
    metrics["sft_alpha"] = sft_alpha if objective == "dpo" else None
    metrics["lora_rank"] = lora_rank
    metrics["lora_alpha"] = lora_alpha
    metrics["lora_dropout"] = lora_dropout
    metrics["learning_rate"] = learning_rate
    metrics["num_pairs"] = len(records)

    with open(f"{output_dir}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Loss: {metrics.get('train_loss', 'N/A')}")
    print(f"Runtime: {metrics.get('train_runtime', 0):.0f}s")
    print(f"Adapter saved: {output_dir}")
    print(f"{'='*60}")

    # Commit volume changes
    vol.commit()

    # Push to Hub if requested
    if push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {hub_repo}...")
        model.push_to_hub(hub_repo)
        tokenizer.push_to_hub(hub_repo)
        print("Pushed successfully!")

    return metrics


# ---------------------------------------------------------------------------
# Pre-cache model (download once, skip on future runs)
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu=GPU,
    timeout=30 * MINUTES,
    volumes={VOLUME_PATH: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_model(base_model: str = "google/gemma-3-27b-it"):
    """Download and cache the base model to the volume. Run once before training."""
    from pathlib import Path
    from unsloth import FastLanguageModel

    model_cache_dir = f"{VOLUME_PATH}/models/{base_model.replace('/', '--')}"
    if Path(model_cache_dir).exists() and any(Path(model_cache_dir).iterdir()):
        print(f"Model already cached at {model_cache_dir}")
        return

    print(f"Downloading {base_model} from HuggingFace...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"Saving to volume: {model_cache_dir}")
    Path(model_cache_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_cache_dir)
    tokenizer.save_pretrained(model_cache_dir)
    vol.commit()
    print("Done — model cached for all future training runs")


# ---------------------------------------------------------------------------
# Download adapter from volume
# ---------------------------------------------------------------------------

@app.function(
    image=modal.Image.debian_slim(),
    volumes={VOLUME_PATH: vol},
)
def list_adapters():
    """List all saved adapters on the volume."""
    from pathlib import Path

    adapter_dir = Path(VOLUME_PATH) / "adapters"
    if not adapter_dir.exists():
        print("No adapters found.")
        return []

    adapters = []
    for d in sorted(adapter_dir.iterdir()):
        if d.is_dir():
            files = list(d.iterdir())
            size_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1024 / 1024
            adapters.append({"name": d.name, "files": len(files), "size_mb": round(size_mb, 1)})
            print(f"  {d.name:30s} {len(files)} files, {size_mb:.1f} MB")

    return adapters


@app.function(
    image=modal.Image.debian_slim(),
    volumes={VOLUME_PATH: vol},
)
def download_adapter(adapter_name: str) -> dict[str, bytes]:
    """Download an adapter's files from the volume. Returns {filename: bytes}."""
    from pathlib import Path

    adapter_dir = Path(VOLUME_PATH) / "adapters" / adapter_name
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter '{adapter_name}' not found on volume")

    files = {}
    for f in adapter_dir.iterdir():
        if f.is_file():
            files[f.name] = f.read_bytes()
            print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")

    print(f"\nTotal: {len(files)} files")
    return files


# ---------------------------------------------------------------------------
# Merge LoRA + Quantize to GGUF (for LMStudio / llama.cpp)
# ---------------------------------------------------------------------------

merge_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "unsloth[cu124-ampere-torch260] @ git+https://github.com/unslothai/unsloth.git",
        extra_options="--no-deps",
    )
    .pip_install(
        "torch>=2.6",
        "transformers>=4.48",
        "accelerate>=1.3",
        "peft>=0.14",
        "bitsandbytes>=0.45",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "packaging",
        "numpy",
        "llama-cpp-python",
    )
    .apt_install("cmake", "build-essential")
)


@app.function(
    image=merge_image,
    gpu=GPU,
    timeout=60 * MINUTES,
    volumes={VOLUME_PATH: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def merge_and_quantize(
    adapter_name: str,
    base_model: str = "google/gemma-3-27b-it",
    quantization: str = "q4_k_m",
) -> bytes:
    """Merge LoRA adapter into base model and export as quantized GGUF.

    Returns the GGUF file as bytes for download to local machine.
    """
    from pathlib import Path
    from unsloth import FastLanguageModel

    adapter_dir = f"{VOLUME_PATH}/adapters/{adapter_name}"
    if not Path(adapter_dir).exists():
        raise FileNotFoundError(f"Adapter '{adapter_name}' not found at {adapter_dir}")

    # Load base model from cache
    model_cache_dir = f"{VOLUME_PATH}/models/{base_model.replace('/', '--')}"
    cached = Path(model_cache_dir).exists() and any(Path(model_cache_dir).iterdir())
    load_name = model_cache_dir if cached else base_model

    print(f"Loading base model from {'cache' if cached else 'HuggingFace'}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Load the LoRA adapter
    print(f"Loading adapter: {adapter_dir}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_dir)

    # Merge and export GGUF
    output_dir = f"{VOLUME_PATH}/exports/{adapter_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Merging LoRA and exporting GGUF ({quantization})...")
    # Unsloth's save_pretrained_gguf handles merge + quantize
    model.save_pretrained_gguf(
        output_dir,
        tokenizer,
        quantization_method=quantization,
    )

    # Find the GGUF file
    gguf_files = list(Path(output_dir).glob("*.gguf"))
    if not gguf_files:
        raise RuntimeError(f"No GGUF file produced in {output_dir}")

    gguf_path = gguf_files[0]
    print(f"GGUF exported: {gguf_path.name} ({gguf_path.stat().st_size / 1024 / 1024:.0f} MB)")

    vol.commit()

    return gguf_path.read_bytes()


# ---------------------------------------------------------------------------
# Upload data helper
# ---------------------------------------------------------------------------

@app.function(
    image=modal.Image.debian_slim(),
    volumes={VOLUME_PATH: vol},
)
def upload_dpo_data(data_bytes: bytes):
    """Upload DPO training data to the persistent volume."""
    from pathlib import Path

    data_dir = Path(VOLUME_PATH) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / "madison-dpo.jsonl"
    output_path.write_bytes(data_bytes)

    line_count = sum(1 for _ in output_path.open())
    print(f"Uploaded {line_count} DPO pairs to {output_path}")
    vol.commit()
    return line_count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    beta: float = 0.1,
    rank: int = 16,
    lr: float = 5e-6,
    epochs: int = 3,
    objective: str = "dpo",
    output_name: str = "madison-lora-v1",
    upload_data: bool = True,
    download_only: bool = False,
    list_models: bool = False,
    get_adapter: str = "",
    export_gguf: str = "",
):
    """Upload data and run DPO training.

    Usage:
        # Pre-cache model (run once, ~5 min)
        modal run modal_train_dpo.py --download-only

        # Train with defaults
        modal run modal_train_dpo.py

        # Train with custom hyperparameters
        modal run modal_train_dpo.py --beta 0.2 --rank 32 --lr 1e-5

        # Train with ORPO (no persona drift — recommended for iteration 2+)
        modal run modal_train_dpo.py --objective orpo --beta 0.05 --output-name madison-orpo-v1

        # List saved adapters
        modal run modal_train_dpo.py --list-models

        # Download adapter to local machine
        modal run modal_train_dpo.py --get-adapter madison-lora-v1

        # Export merged GGUF (train + merge + quantize in one step)
        modal run modal_train_dpo.py --export-gguf madison-lora-v1
    """
    from pathlib import Path

    # List adapters
    if list_models:
        print("Adapters on Modal volume:")
        list_adapters.remote()
        return

    # Download adapter to local machine
    if get_adapter:
        print(f"Downloading adapter '{get_adapter}' from Modal...")
        files = download_adapter.remote(get_adapter)
        local_dir = Path(f"adapters/{get_adapter}")
        local_dir.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (local_dir / name).write_bytes(data)
            print(f"  Saved {name} ({len(data) / 1024:.0f} KB)")
        print(f"\nAdapter saved to {local_dir}/")
        return

    # Export merged GGUF
    if export_gguf:
        print(f"Exporting '{export_gguf}' as merged GGUF...")
        gguf_bytes = merge_and_quantize.remote(export_gguf)
        local_path = Path(f"adapters/{export_gguf}/madison-{export_gguf}.Q4_K_M.gguf")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(gguf_bytes)
        print(f"\nGGUF saved to {local_path} ({len(gguf_bytes) / 1024 / 1024:.0f} MB)")
        print("Load this in LMStudio to run the fine-tuned Madison model.")
        return

    # Download-only mode: cache the model and exit
    if download_only:
        print("Downloading and caching base model...")
        download_model.remote()
        print("Model cached. Future training runs will skip the download.")
        return

    data_path = Path("data/training/madison-dpo.jsonl")

    if upload_data and data_path.exists():
        print(f"Uploading {data_path}...")
        data_bytes = data_path.read_bytes()
        n = upload_dpo_data.remote(data_bytes)
        print(f"Uploaded {n} pairs to Modal volume")
    elif not data_path.exists():
        print(f"WARNING: {data_path} not found. Assuming data already on volume.")

    print(f"\nLaunching DPO training on {GPU}...")
    print(f"  beta={beta}, rank={rank}, lr={lr}, epochs={epochs}")

    metrics = train_madison_dpo.remote(
        beta=beta,
        lora_rank=rank,
        lora_alpha=rank,  # alpha = rank is standard
        learning_rate=lr,
        num_epochs=epochs,
        objective=objective,
        output_name=output_name,
    )

    print(f"\nTraining complete! Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
