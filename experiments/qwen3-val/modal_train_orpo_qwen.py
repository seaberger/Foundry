"""QLoRA ORPO training for Madison character on Qwen 3-32B — VALIDATION EXPERIMENT.

Small rank-8 LoRA to test whether Qwen 3-32B takes character imprinting.
De-risks the Gemma 3 → Qwen 3 base model switch before v5 ORPO.

Usage:
    modal run modal_train_orpo_qwen.py --output-name madison-qwen3-val-v1
    modal run modal_train_orpo_qwen.py --list-models
    modal run modal_train_orpo_qwen.py --get-adapter madison-qwen3-val-v1
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

MINUTES = 60
GPU = "A100-80GB"  # 32B model at rank 64 needs headroom
TIMEOUT = 240 * MINUTES

app = modal.App("foundry-qwen3-val")

model_cache_vol = modal.Volume.from_name("foundry-model-cache", create_if_missing=True)
adapter_vol = modal.Volume.from_name("foundry-adapters", create_if_missing=True)

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

with training_image.imports():
    import unsloth  # noqa: F401,I001
    from unsloth import FastLanguageModel, PatchDPOTrainer

    PatchDPOTrainer()

    import datasets
    import torch
    import wandb
    from trl import ORPOConfig, ORPOTrainer
    from unsloth import is_bfloat16_supported

# LoRA target modules — same for Qwen 3 as Gemma 3
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
def train_qwen3_orpo(
    # Data
    dpo_data_path: str = "/adapters/data/madison-orpo-qwen3-val.jsonl",
    # Model — Qwen 3-32B instruct (pure ForCausalLM, no VLM issues)
    base_model: str = "Qwen/Qwen3-32B",
    max_seq_length: int = 2048,
    # LoRA config — rank 64 to survive GGUF Q4_K_M (rank 16 proven fragile)
    lora_rank: int = 64,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    # ORPO hyperparameters
    beta: float = 0.1,
    learning_rate: float = 5e-6,
    num_epochs: int = 3,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
    # Output
    output_name: str = "madison-qwen3-val-v1",
    # Resume
    resume_from_checkpoint: str = "",
    # Tracking
    wandb_project: str = "foundry",
    wandb_run_name: str | None = None,
):
    """Run QLoRA ORPO training on Qwen 3-32B — validation experiment."""
    import json
    import os
    from pathlib import Path

    print(f"{'='*60}")
    print(f"Foundry — Qwen 3-32B Validation ORPO Training")
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
                "experiment": "qwen3-validation",
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
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        seed=42,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=run_name,
        dataset_num_proc=1,
    )

    # ----- Train -----
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
        tokenizer=tokenizer,
    )

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # ----- Save -----
    print(f"\nSaving adapter to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    metrics["objective"] = "orpo"
    metrics["experiment"] = "qwen3-validation"
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
# Utility functions (reused from modal_train_orpo.py)
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
def upload_data(data_bytes: bytes, filename: str = "madison-orpo-qwen3-val.jsonl"):
    """Upload preference training data to the persistent volume."""
    from pathlib import Path

    data_dir = Path("/adapters/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / filename
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
    rank: int = 64,
    lr: float = 5e-6,
    epochs: int = 3,
    output_name: str = "madison-qwen3-val-v1",
    dataset: str = "../../data/training/madison-orpo-qwen3-val.jsonl",
    upload_data_flag: bool = True,
    list_models: bool = False,
    get_adapter: str = "",
):
    """Run Qwen 3-32B validation ORPO training on Modal.

    Usage:
        modal run modal_train_orpo_qwen.py --output-name madison-qwen3-val-v1
        modal run modal_train_orpo_qwen.py --list-models
        modal run modal_train_orpo_qwen.py --get-adapter madison-qwen3-val-v1
    """
    from pathlib import Path

    if list_models:
        print("Adapters on Modal volume:")
        list_adapters.remote()
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

    print(f"\nLaunching Qwen 3-32B validation ORPO training on {GPU}...")
    print(f"  beta={beta}, rank={rank}, lr={lr}, epochs={epochs}")

    metrics = train_qwen3_orpo.remote(
        beta=beta,
        lora_rank=rank,
        lora_alpha=rank,
        learning_rate=lr,
        num_epochs=epochs,
        output_name=output_name,
    )

    print(f"\nTraining complete! Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
