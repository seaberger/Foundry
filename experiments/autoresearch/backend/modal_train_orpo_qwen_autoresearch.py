"""Modal backend for step-capped ORPO autoresearch on Qwen 3-32B.

This is a starting point adapted from Foundry's existing Qwen ORPO training script.
It adds the knobs the wrapper needs for short probe runs:

- max_steps
- warmup_ratio
- batch/accum settings
- explicit alpha/dropout
- dataset path passthrough

Copy this into the Foundry repo as:
  autoresearch_qwen/backend/modal_train_orpo_qwen_autoresearch.py

Then run it through the wrapper in `autoresearch_qwen/train.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

MINUTES = 60
GPU = "A100-80GB"
TIMEOUT = 240 * MINUTES

app = modal.App("foundry-qwen3-autoresearch")
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
    import datasets
    import os
    import wandb
    from trl import ORPOConfig, ORPOTrainer
    from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported

    PatchDPOTrainer()

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@app.function(
    image=training_image,
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={"/model_cache": model_cache_vol, "/adapters": adapter_vol},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train_qwen3_orpo(
    dpo_data_path: str = "/adapters/data/madison-dpo.jsonl",
    base_model: str = "Qwen/Qwen3-32B",
    max_seq_length: int = 2048,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    beta: float = 0.1,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    max_steps: int = 150,
    warmup_ratio: float = 0.10,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
    output_name: str = "madison-qwen3-autoresearch",
    resume_from_checkpoint: str = "",
    save_steps: int = 0,
    save_total_limit: int = 3,
    wandb_project: str = "foundry",
    wandb_run_name: str | None = None,
):
    print(f"{'='*60}")
    print("Foundry — Qwen 3-32B ORPO Autoresearch")
    print(f"{'='*60}")
    print(f"Base model: {base_model}")
    print(f"LoRA: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"Beta: {beta}, LR: {learning_rate}, warmup_ratio={warmup_ratio}")
    print(f"Epochs: {num_epochs}, max_steps={max_steps}")
    print(f"Batch: {per_device_batch_size} x {gradient_accumulation_steps} accum")
    print(f"Output: {output_name}")
    print(f"{'='*60}")

    data_path = Path(dpo_data_path)
    records = []
    with data_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
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
    steps_per_epoch = max(1, len(train_dataset) // (per_device_batch_size * gradient_accumulation_steps))
    total_steps_est = steps_per_epoch * num_epochs
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    print(f"Estimated steps/epoch: ~{steps_per_epoch}, uncapped total: ~{total_steps_est}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model_cache_vol.commit()

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

    run_name = wandb_run_name or output_name
    if os.environ.get("WANDB_API_KEY"):
        wandb.init(
            entity="sbergman",
            project=wandb_project,
            name=run_name,
            config={
                "base_model": base_model,
                "objective": "orpo",
                "experiment": "qwen3-autoresearch",
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "beta": beta,
                "learning_rate": learning_rate,
                "warmup_ratio": warmup_ratio,
                "num_epochs": num_epochs,
                "max_steps": max_steps,
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

    output_dir = f"/adapters/experiments/{output_name}"
    training_args = ORPOConfig(
        beta=beta,
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_seq_length,
        max_prompt_length=512,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=save_steps if save_steps > 0 else max(10, min(50, max_steps // 3 if max_steps > 0 else 50)),
        save_strategy="steps",
        save_steps=save_steps if save_steps > 0 else max(10, min(50, max_steps // 3 if max_steps > 0 else 50)),
        save_total_limit=save_total_limit,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        seed=42,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=run_name,
        dataset_num_proc=1,
    )

    checkpoint = resume_from_checkpoint if resume_from_checkpoint else None
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics = train_result.metrics
    metrics.update(
        {
            "objective": "orpo",
            "experiment": "qwen3-autoresearch",
            "beta": beta,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "learning_rate": learning_rate,
            "warmup_ratio": warmup_ratio,
            "max_steps": max_steps,
            "max_grad_norm": max_grad_norm,
            "num_pairs": len(records),
        }
    )
    with open(f"{output_dir}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"{'='*60}")
    print("Training complete!")
    print(f"Loss: {metrics.get('train_loss', 'N/A')}")
    print(f"Runtime: {metrics.get('train_runtime', 0):.0f}s")
    print(f"Adapter saved: {output_dir}")
    print(f"{'='*60}")

    adapter_vol.commit()
    if os.environ.get("WANDB_API_KEY"):
        wandb.finish()
    return metrics


@app.function(image=modal.Image.debian_slim(), volumes={"/adapters": adapter_vol})
def download_adapter(adapter_name: str) -> dict[str, bytes]:
    adapter_dir = Path("/adapters/experiments") / adapter_name
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter '{adapter_name}' not found")
    files = {}
    for f in adapter_dir.iterdir():
        if f.is_file():
            files[f.name] = f.read_bytes()
    return files


@app.function(image=modal.Image.debian_slim(), volumes={"/adapters": adapter_vol})
def upload_data(data_bytes: bytes, filename: str = "madison-dpo.jsonl"):
    data_dir = Path("/adapters/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / filename
    output_path.write_bytes(data_bytes)
    line_count = sum(1 for _ in output_path.open())
    adapter_vol.commit()
    return {"path": str(output_path), "line_count": line_count}


@app.local_entrypoint()
def main(
    beta: float = 0.1,
    rank: int = 64,
    alpha: int = 64,
    dropout: float = 0.0,
    lr: float = 2e-5,
    epochs: int = 3,
    max_steps: int = 150,
    warmup_ratio: float = 0.10,
    batch_size: int = 1,
    grad_accum: int = 4,
    max_seq_length: int = 2048,
    output_name: str = "madison-qwen3-autoresearch",
    save_steps: int = 0,
    save_total_limit: int = 3,
    dataset: str = "autoresearch_qwen/runs/latest/train.jsonl",
    upload_data_flag: bool = True,
    resume_from: str = "",
):
    data_path = Path(dataset)
    remote_dataset_path = "/adapters/data/madison-dpo-autoresearch.jsonl"
    if upload_data_flag and data_path.exists():
        payload = upload_data.remote(data_path.read_bytes(), filename=Path(remote_dataset_path).name)
        print(f"Uploaded {payload['line_count']} pairs to {payload['path']}")
        remote_dataset_path = payload["path"]
    elif not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    metrics = train_qwen3_orpo.remote(
        dpo_data_path=remote_dataset_path,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        beta=beta,
        learning_rate=lr,
        num_epochs=epochs,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        per_device_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_seq_length=max_seq_length,
        output_name=output_name,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        resume_from_checkpoint=f"/adapters/experiments/{resume_from}" if resume_from else "",
    )
    print(json.dumps(metrics, indent=2))
