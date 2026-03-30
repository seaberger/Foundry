"""QLoRA SFT training for Madison character introspection on Modal A100.

Supervised Fine-Tuning (SFT) — second stage after ORPO preference alignment.
Trains on introspection data (self-reflective, metacognitive responses) starting
from the merged ORPO v4 model as the base.

Training pipeline:
  1. ORPO v4 (preference alignment on chosen/rejected pairs)
  2. SFT v1 (introspection — this script)

Loads the merged ORPO v4 16-bit model from the Modal volume, applies fresh
LoRA rank 16 on top, and fine-tunes on messages-format JSONL data.

Usage:
    modal run scripts/modal/train_sft.py --output-name madison-sft-v1
    modal run scripts/modal/train_sft.py --lr 1e-5 --output-name madison-sft-v2
    modal run scripts/modal/train_sft.py --list-models
    modal run scripts/modal/train_sft.py --get-adapter madison-sft-v1
    modal run scripts/modal/train_sft.py --dataset data/training/my-data.jsonl
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

MINUTES = 60
GPU = "A100"
TIMEOUT = 240 * MINUTES

app = modal.App("foundry-madison-sft")

# Persistent volumes — same as ORPO
model_cache_vol = modal.Volume.from_name("foundry-model-cache", create_if_missing=True)
adapter_vol = modal.Volume.from_name("foundry-adapters", create_if_missing=True)

# Container image — pinned versions matching ORPO script
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

    PatchDPOTrainer()  # Patches trainers including SFT

    import datasets
    import torch
    import wandb
    from trl import SFTConfig, SFTTrainer
    from unsloth import is_bfloat16_supported

# LoRA target modules — attention + MLP (all linear layers)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Default merged model path on the Modal volume
DEFAULT_BASE_MODEL = "/adapters/merged/madison-orpo-v4-16bit"


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
def train_madison_sft(
    # Data
    sft_data_path: str = "/adapters/data/madison-introspection-sft.jsonl",
    # Model
    base_model: str = DEFAULT_BASE_MODEL,
    max_seq_length: int = 2048,
    # LoRA config
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,  # Unsloth recommends 0 for fast patching
    # SFT hyperparameters
    learning_rate: float = 2e-5,
    num_epochs: int = 1,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,  # Effective batch = 4
    max_grad_norm: float = 1.0,
    # Output
    output_name: str = "madison-sft-v1",
    # Resume
    resume_from_checkpoint: str = "",
    # Tracking
    wandb_project: str = "foundry",
    wandb_run_name: str | None = None,
):
    """Run QLoRA SFT introspection training on Modal A100."""
    import json
    import os
    from pathlib import Path

    print(f"{'='*60}")
    print(f"Foundry — Madison SFT Introspection Training")
    print(f"{'='*60}")
    print(f"Base model:  {base_model}")
    print(f"LoRA:        rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"LR:          {learning_rate}")
    print(f"Epochs:      {num_epochs}")
    print(f"Batch:       {per_device_batch_size} x {gradient_accumulation_steps} accum = {per_device_batch_size * gradient_accumulation_steps} effective")
    print(f"Grad clip:   {max_grad_norm}")
    print(f"Max seq len: {max_seq_length}")
    print(f"Output:      {output_name}")
    print(f"{'='*60}")

    # ----- Load data -----
    data_path = Path(sft_data_path)
    print(f"\nLoading SFT data from {data_path}...")
    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} training examples")

    # Validate format — each record must have "messages" key
    for i, r in enumerate(records):
        if "messages" not in r:
            raise ValueError(f"Record {i} missing 'messages' key: {list(r.keys())}")
        if not isinstance(r["messages"], list) or len(r["messages"]) < 2:
            raise ValueError(f"Record {i} 'messages' must be a list with at least 2 turns")

    # Build HuggingFace dataset — SFTTrainer consumes the messages column directly
    dataset = datasets.Dataset.from_dict({
        "messages": [r["messages"] for r in records],
    })
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    steps_per_epoch = len(train_dataset) // (per_device_batch_size * gradient_accumulation_steps)
    total_steps = steps_per_epoch * num_epochs
    print(f"Estimated: ~{steps_per_epoch} steps/epoch, ~{total_steps} steps total")

    # ----- Load model -----
    # Load merged ORPO v4 model (16-bit on volume) with 4-bit quantization for training
    print(f"\nLoading {base_model} (4-bit quantized for training)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model_cache_vol.commit()

    # ----- Configure LoRA -----
    # Fresh LoRA adapter on top of the merged ORPO model
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
                "objective": "sft",
                "stage": "introspection",
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "batch_size": per_device_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": per_device_batch_size * gradient_accumulation_steps,
                "max_grad_norm": max_grad_norm,
                "max_seq_length": max_seq_length,
                "num_examples": len(records),
                "train_size": len(train_dataset),
                "eval_size": len(eval_dataset),
            },
        )

    # ----- Training Config -----
    output_dir = f"/adapters/experiments/{output_name}"

    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_seq_length=max_seq_length,
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
        packing=False,  # No packing — each example is a complete conversation
    )

    # ----- Train -----
    checkpoint = resume_from_checkpoint if resume_from_checkpoint else None
    if checkpoint:
        print(f"\nResuming SFT training from {checkpoint}...")
    else:
        print(f"\nStarting SFT introspection training...")

    # Unsloth requires a formatting_func to convert messages → text.
    # Use manual Gemma 3 template since novision tokenizer lacks chat_template.
    # Unsloth calls this both per-example (test) and batched (training).
    def formatting_func(examples):
        messages_list = examples["messages"]
        # Handle single example (test call) vs batched
        if isinstance(messages_list[0], dict):
            messages_list = [messages_list]
        texts = []
        for msgs in messages_list:
            text = ""
            for msg in msgs:
                role = "model" if msg["role"] == "assistant" else msg["role"]
                text += f"<start_of_turn>{role}\n{msg['content']}<end_of_turn>\n"
            texts.append(text)
        return texts

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # ----- Save -----
    print(f"\nSaving adapter to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    metrics["objective"] = "sft"
    metrics["stage"] = "introspection"
    metrics["lora_rank"] = lora_rank
    metrics["lora_dropout"] = lora_dropout
    metrics["learning_rate"] = learning_rate
    metrics["max_grad_norm"] = max_grad_norm
    metrics["num_examples"] = len(records)

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
                print(f"{'  ' * indent}  {item.name}/ ({len(sub_files)} files, {sub_size:.1f} MB)")
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
def upload_data(data_bytes: bytes, filename: str = "madison-introspection-sft.jsonl"):
    """Upload SFT training data to the persistent volume."""
    from pathlib import Path

    data_dir = Path("/adapters/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / filename
    output_path.write_bytes(data_bytes)

    line_count = sum(1 for line in output_path.open() if line.strip())
    print(f"Uploaded {line_count} examples to {output_path}")
    adapter_vol.commit()
    return line_count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    rank: int = 16,
    lr: float = 2e-5,
    epochs: int = 1,
    output_name: str = "madison-sft-v1",
    dataset: str = "data/training/madison-introspection-sft.jsonl",
    base_model: str = DEFAULT_BASE_MODEL,
    upload_data_flag: bool = True,
    list_models: bool = False,
    get_adapter: str = "",
    inspect: str = "",
    resume_from: str = "",
):
    """Run Madison SFT introspection training on Modal.

    Usage:
        modal run scripts/modal/train_sft.py --output-name madison-sft-v1
        modal run scripts/modal/train_sft.py --lr 1e-5 --output-name madison-sft-v2
        modal run scripts/modal/train_sft.py --list-models
        modal run scripts/modal/train_sft.py --get-adapter madison-sft-v1
        modal run scripts/modal/train_sft.py --inspect madison-sft-v1
        modal run scripts/modal/train_sft.py --resume-from madison-sft-v1/checkpoint-50 --output-name madison-sft-v1
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
    data_filename = data_path.name

    if upload_data_flag and data_path.exists():
        print(f"Uploading {data_path}...")
        data_bytes = data_path.read_bytes()
        n = upload_data.remote(data_bytes, filename=data_filename)
        print(f"Uploaded {n} examples to Modal volume")
    elif not data_path.exists():
        print(f"WARNING: {data_path} not found. Assuming data already on volume.")

    # Resolve the remote data path on the Modal volume
    remote_data_path = f"/adapters/data/{data_filename}"

    print(f"\nLaunching SFT introspection training on {GPU}...")
    print(f"  base_model={base_model}")
    print(f"  rank={rank}, lr={lr}, epochs={epochs}")

    metrics = train_madison_sft.remote(
        sft_data_path=remote_data_path,
        base_model=base_model,
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
