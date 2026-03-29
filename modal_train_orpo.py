"""QLoRA ORPO training for Madison character on Modal A100.

ORPO (Odds Ratio Preference Optimization) — reference-model-free preference
training that integrates SFT and preference learning in a single objective.

Chosen over DPO for v2 based on:
  - "Objective Matters" paper: ORPO shows zero persona drift at our data scale
  - v1 DPO collapsed (loss→0, margins→15) by epoch 0.6 on 475 pairs
  - ORPO's SFT component prevents likelihood displacement
  - No reference model = ~50% less VRAM (no OOM risk on A100-40GB)

Usage:
    modal run modal_train_orpo.py --output-name madison-orpo-v1
    modal run modal_train_orpo.py --lr 8e-6 --output-name madison-orpo-v2
    modal run modal_train_orpo.py --list-models
    modal run modal_train_orpo.py --get-adapter madison-orpo-v1
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

MINUTES = 60
GPU = "A100"
TIMEOUT = 240 * MINUTES

app = modal.App("foundry-madison-orpo")

# Persistent volumes — reuse model cache from DPO runs
model_cache_vol = modal.Volume.from_name("foundry-model-cache", create_if_missing=True)
adapter_vol = modal.Volume.from_name("foundry-adapters", create_if_missing=True)

# Container image — pinned versions from Modal's official Unsloth example
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.9.0",
        "datasets==3.6.0",
        "huggingface_hub==0.34.2",
        "peft==0.16.0",
        "transformers==4.54.0",
        "trl==0.19.1",
        "unsloth[cu128-torch270]==2025.7.8",
        "unsloth_zoo==2025.7.10",
        "wandb==0.21.0",
    )
    .env({"HF_HOME": "/model_cache"})
)

# Pre-import with correct ordering — unsloth MUST be first (applies monkey patches)
with training_image.imports():
    import unsloth  # noqa: F401,I001
    from unsloth import FastLanguageModel, PatchDPOTrainer

    PatchDPOTrainer()  # Patches both DPO and ORPO trainers

    import datasets
    import torch
    import wandb
    from transformers import AutoTokenizer
    from trl import ORPOConfig, ORPOTrainer
    from unsloth import is_bfloat16_supported

# LoRA target modules — attention + MLP (all linear layers)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={
        "/model_cache": model_cache_vol,
        "/adapters": adapter_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train_madison_orpo(
    # Data
    dpo_data_path: str = "/adapters/data/madison-dpo.jsonl",
    # Model
    base_model: str = "google/gemma-3-27b-it",
    max_seq_length: int = 2048,
    # LoRA config
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,  # Unsloth recommends 0 for fast patching
    # ORPO hyperparameters
    beta: float = 0.1,  # Odds ratio weight (ORPO paper default)
    learning_rate: float = 5e-6,
    num_epochs: int = 3,
    per_device_batch_size: int = 1,  # Conservative for A100-40GB (no ref model but still 27B)
    gradient_accumulation_steps: int = 4,  # Effective batch = 4
    max_grad_norm: float = 1.0,  # Clip gradient spikes (v1 DPO had norms 14-708)
    # Output
    output_name: str = "madison-orpo-v1",
    # Resume
    resume_from_checkpoint: str = "",  # e.g. "/adapters/experiments/madison-orpo-v4/checkpoint-650"
    # Tracking
    wandb_project: str = "foundry",
    wandb_run_name: str | None = None,
):
    """Run QLoRA ORPO training on Modal A100."""
    import json
    import os
    from pathlib import Path

    print(f"{'='*60}")
    print(f"Foundry — Madison ORPO Training")
    print(f"{'='*60}")
    print(f"Base model:  {base_model}")
    print(f"LoRA:        rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"Beta:        {beta}, LR: {learning_rate}")
    print(f"Epochs:      {num_epochs}")
    print(f"Batch:       {per_device_batch_size} x {gradient_accumulation_steps} accum = {per_device_batch_size * gradient_accumulation_steps} effective")
    print(f"Grad clip:   {max_grad_norm}")
    print(f"Output:      {output_name}")
    print(f"{'='*60}")

    # ----- Load data -----
    data_path = Path(dpo_data_path)
    print(f"\nLoading preference data from {data_path}...")
    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} preference pairs")

    # Plain string format — avoids Arrow serialization issues with Gemma 3 tokenizer
    dataset_dict = {"prompt": [], "chosen": [], "rejected": []}
    for r in records:
        dataset_dict["prompt"].append(r["chosen"][0]["content"])
        dataset_dict["chosen"].append(r["chosen"][1]["content"])
        dataset_dict["rejected"].append(r["rejected"][1]["content"])

    dataset = datasets.Dataset.from_dict(dataset_dict)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # With effective batch=4 and 427 train examples: ~107 steps/epoch, ~321 steps total
    steps_per_epoch = len(train_dataset) // (per_device_batch_size * gradient_accumulation_steps)
    total_steps = steps_per_epoch * num_epochs
    print(f"Estimated: ~{steps_per_epoch} steps/epoch, ~{total_steps} steps total")

    # ----- Load model -----
    print(f"\nLoading {base_model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model_cache_vol.commit()

    # AutoTokenizer for ORPO — Unsloth's tokenizer causes Arrow serialization
    # issues with Gemma 3 (unslothai/unsloth#2310, #2214)
    orpo_tokenizer = AutoTokenizer.from_pretrained(base_model)

    # ----- Configure LoRA -----
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=max_seq_length,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ----- W&B Setup -----
    run_name = wandb_run_name or output_name
    if os.environ.get("WANDB_API_KEY"):
        wandb.init(
            entity="sbergman",
            project=wandb_project,
            name=run_name,
            config={
                "base_model": base_model,
                "objective": "orpo",
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "beta": beta,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "batch_size": per_device_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": per_device_batch_size * gradient_accumulation_steps,
                "max_grad_norm": max_grad_norm,
                "max_seq_length": max_seq_length,
                "num_pairs": len(records),
                "train_size": len(train_dataset),
                "eval_size": len(eval_dataset),
            },
        )

    # ----- Training Config -----
    output_dir = f"/adapters/experiments/{output_name}"

    training_args = ORPOConfig(
        beta=beta,
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_seq_length,
        max_prompt_length=512,
        max_grad_norm=max_grad_norm,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        seed=42,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=run_name,
        dataset_num_proc=1,
    )

    # ----- Train -----
    # ORPO is reference-model-free — no ref_model needed
    checkpoint = resume_from_checkpoint if resume_from_checkpoint else None
    if checkpoint:
        print(f"\nResuming ORPO training from {checkpoint}...")
    else:
        print(f"\nStarting ORPO training...")
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=orpo_tokenizer,
    )

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # ----- Save -----
    print(f"\nSaving adapter to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    metrics["objective"] = "orpo"
    metrics["beta"] = beta
    metrics["lora_rank"] = lora_rank
    metrics["lora_dropout"] = lora_dropout
    metrics["learning_rate"] = learning_rate
    metrics["max_grad_norm"] = max_grad_norm
    metrics["num_pairs"] = len(records)

    with open(f"{output_dir}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Loss: {metrics.get('train_loss', 'N/A')}")
    print(f"Runtime: {metrics.get('train_runtime', 0):.0f}s")
    print(f"Adapter saved: {output_dir}")
    print(f"{'='*60}")

    adapter_vol.commit()

    if os.environ.get("WANDB_API_KEY"):
        wandb.finish()

    return metrics


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/adapters": adapter_vol},
)
def list_adapters():
    """List all saved adapters on the volume."""
    from pathlib import Path

    exp_dir = Path("/adapters/experiments")
    if not exp_dir.exists():
        print("No adapters found.")
        return []

    adapters = []
    for d in sorted(exp_dir.iterdir()):
        if d.is_dir():
            files = list(d.iterdir())
            size_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1024 / 1024
            adapters.append({"name": d.name, "files": len(files), "size_mb": round(size_mb, 1)})
            print(f"  {d.name:30s} {len(files)} files, {size_mb:.1f} MB")

    return adapters


@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/adapters": adapter_vol},
)
def inspect_experiment(experiment_name: str):
    """List all files and subdirectories in an experiment directory."""
    from pathlib import Path

    exp_dir = Path("/adapters/experiments") / experiment_name
    if not exp_dir.exists():
        print(f"Experiment '{experiment_name}' not found")
        return

    def walk(path, indent=0):
        for item in sorted(path.iterdir()):
            if item.is_dir():
                sub_files = list(item.rglob("*"))
                sub_size = sum(f.stat().st_size for f in sub_files if f.is_file()) / 1024 / 1024
                print(f"{'  ' * indent}📁 {item.name}/ ({len(sub_files)} files, {sub_size:.1f} MB)")
                walk(item, indent + 1)
            else:
                size_kb = item.stat().st_size / 1024
                print(f"{'  ' * indent}  {item.name} ({size_kb:.0f} KB)")

    walk(exp_dir)


@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/adapters": adapter_vol},
)
def download_adapter(adapter_name: str) -> dict[str, bytes]:
    """Download an adapter's files from the volume."""
    from pathlib import Path

    adapter_dir = Path("/adapters/experiments") / adapter_name
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter '{adapter_name}' not found")

    files = {}
    for f in adapter_dir.iterdir():
        if f.is_file():
            files[f.name] = f.read_bytes()
            print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")

    print(f"\nTotal: {len(files)} files")
    return files


@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/adapters": adapter_vol},
)
def upload_data(data_bytes: bytes):
    """Upload preference training data to the persistent volume."""
    from pathlib import Path

    data_dir = Path("/adapters/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / "madison-dpo.jsonl"
    output_path.write_bytes(data_bytes)

    line_count = sum(1 for _ in output_path.open())
    print(f"Uploaded {line_count} preference pairs to {output_path}")
    adapter_vol.commit()
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
    output_name: str = "madison-orpo-v1",
    dataset: str = "data/training/madison-dpo.jsonl",
    upload_data_flag: bool = True,
    list_models: bool = False,
    get_adapter: str = "",
    inspect: str = "",
    resume_from: str = "",
):
    """Run Madison ORPO training on Modal.

    Usage:
        modal run modal_train_orpo.py --output-name madison-orpo-v1
        modal run modal_train_orpo.py --lr 8e-6 --output-name madison-orpo-v2
        modal run modal_train_orpo.py --list-models
        modal run modal_train_orpo.py --get-adapter madison-orpo-v1
        modal run modal_train_orpo.py --inspect madison-orpo-v4
        modal run modal_train_orpo.py --resume-from madison-orpo-v4/checkpoint-650 --output-name madison-orpo-v4
    """
    from pathlib import Path

    if list_models:
        print("Adapters on Modal volume:")
        list_adapters.remote()
        return

    if inspect:
        print(f"Inspecting experiment '{inspect}':")
        inspect_experiment.remote(inspect)
        return

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

    data_path = Path(dataset)

    if upload_data_flag and data_path.exists():
        print(f"Uploading {data_path}...")
        data_bytes = data_path.read_bytes()
        n = upload_data.remote(data_bytes)
        print(f"Uploaded {n} pairs to Modal volume")
    elif not data_path.exists():
        print(f"WARNING: {data_path} not found. Assuming data already on volume.")

    print(f"\nLaunching ORPO training on {GPU}...")
    print(f"  beta={beta}, rank={rank}, lr={lr}, epochs={epochs}")

    metrics = train_madison_orpo.remote(
        beta=beta,
        lora_rank=rank,
        lora_alpha=rank,
        learning_rate=lr,
        num_epochs=epochs,
        output_name=output_name,
        resume_from_checkpoint=f"/adapters/experiments/{resume_from}" if resume_from else "",
    )

    print(f"\nTraining complete! Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
