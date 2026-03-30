"""Generate evaluation responses from fine-tuned Madison model on Modal.

Two modes:
  1. vLLM (fast, ~10-30s/prompt) — requires merged 16-bit model on volume
     First run: modal run scripts/modal/merge_model.py --adapter madison-orpo-v3b-lr2e5
     Then: modal run scripts/modal/generate_eval.py --tag orpo-v3b

  2. Unsloth fallback (slow, ~160-200s/prompt) — uses adapter directly
     modal run scripts/modal/generate_eval.py --tag orpo-v3b --use-unsloth

Supports checkpointing and resume for both modes.

Usage:
    # Fast mode (vLLM with merged model)
    modal run scripts/modal/generate_eval.py --tag orpo-v3b

    # Resume interrupted run
    modal run scripts/modal/generate_eval.py --tag orpo-v3b

    # Slow fallback (Unsloth, no merge needed)
    modal run scripts/modal/generate_eval.py --tag orpo-v3b --use-unsloth

    # Force regenerate
    modal run scripts/modal/generate_eval.py --tag orpo-v3b --fresh
"""

from __future__ import annotations

import modal

MINUTES = 60
GPU = "A100"

app = modal.App("foundry-madison-eval")

model_cache_vol = modal.Volume.from_name("foundry-model-cache", create_if_missing=True)
adapter_vol = modal.Volume.from_name("foundry-adapters", create_if_missing=True)

# vLLM image — no Unsloth, native Gemma 3 cache support
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "vllm>=0.12.0",
    )
    .env({"HF_HOME": "/model_cache"})
)

# Unsloth fallback image — for when merged model doesn't exist yet
unsloth_image = (
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
    )
    .env({"HF_HOME": "/model_cache"})
)

with unsloth_image.imports():
    import unsloth  # noqa: F401,I001
    from unsloth import FastLanguageModel


# ---------------------------------------------------------------------------
# vLLM generation (fast path)
# ---------------------------------------------------------------------------

@app.function(
    image=vllm_image,
    gpu="A100-80GB",  # 51GB BF16 model needs >40GB for model + KV cache
    timeout=30 * MINUTES,
    volumes={
        "/model_cache": model_cache_vol,
        "/adapters": adapter_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_vllm(
    eval_prompts: list[dict],
    tag: str = "orpo-v3b",
    merged_model_path: str = "/adapters/merged/madison-orpo-v3b-lr2e5-16bit",
    max_tokens: int = 1024,
    fresh: bool = False,
) -> list[dict]:
    """Generate responses using vLLM with merged 16-bit model."""
    import json
    import time
    from pathlib import Path

    from vllm import LLM, SamplingParams

    # ----- Checkpoint setup -----
    checkpoint_dir = Path(f"/adapters/eval-checkpoints/{tag}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    completed = {}
    if not fresh:
        for f in checkpoint_dir.glob("*.json"):
            try:
                r = json.loads(f.read_text())
                completed[r["id"]] = r
            except (json.JSONDecodeError, KeyError):
                pass
        if completed:
            print(f"Resuming: {len(completed)}/{len(eval_prompts)} already completed")

    remaining = [p for p in eval_prompts if p["id"] not in completed]
    if not remaining:
        print("All prompts already completed.")
        return list(completed.values())

    print(f"{len(remaining)} prompts to generate ({len(completed)} cached)")

    # ----- Load merged model with vLLM -----
    print(f"Loading merged model from {merged_model_path}...")
    llm = LLM(
        model=merged_model_path,
        tokenizer="google/gemma-3-27b-it",  # Original tokenizer has proper image_token vocab
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        dtype="auto",
        # Use native ForConditionalGeneration — weights have language_model. prefix
        # Text-only inputs work fine; vision stack just costs ~2-3GB extra VRAM
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_k=64,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    # ----- Generate with checkpointing -----
    for i, p in enumerate(remaining):
        prompt_text = p["prompt"]
        print(f"[{len(completed) + i + 1}/{len(eval_prompts)}] {p['id']}: {prompt_text[:60]}...")

        # Gemma 3 chat format
        conversation = f"<start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"

        start = time.time()
        outputs = llm.generate([conversation], sampling_params)
        response_text = outputs[0].outputs[0].text
        elapsed = time.time() - start

        num_tokens = len(outputs[0].outputs[0].token_ids)

        result = {
            "id": p["id"],
            "category": p["category"],
            "difficulty": p.get("difficulty", "medium"),
            "prompt": prompt_text,
            "ground_truth_signal": p.get("ground_truth_signal", ""),
            "response": response_text,
            "generation_time": round(elapsed, 1),
            "model": "madison-orpo-v3b-lr2e5",
            "prompt_tokens": 0,
            "completion_tokens": num_tokens,
        }

        checkpoint_path = checkpoint_dir / f"{p['id']}.json"
        checkpoint_path.write_text(json.dumps(result, indent=2))
        adapter_vol.commit()

        completed[p["id"]] = result
        print(f"  {num_tokens} tokens in {elapsed:.1f}s [checkpointed]")

    all_results = [completed[p["id"]] for p in eval_prompts if p["id"] in completed]
    print(f"\nDone. {len(all_results)} total responses.")
    return all_results


# ---------------------------------------------------------------------------
# Unsloth generation (slow fallback)
# ---------------------------------------------------------------------------

@app.function(
    image=unsloth_image,
    gpu=GPU,
    timeout=240 * MINUTES,
    volumes={
        "/model_cache": model_cache_vol,
        "/adapters": adapter_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_unsloth(
    eval_prompts: list[dict],
    adapter_name: str = "madison-orpo-v3b-lr2e5",
    tag: str = "orpo-v3b",
    base_model: str = "google/gemma-3-27b-it",
    max_seq_length: int = 2048,
    max_new_tokens: int = 1024,
    fresh: bool = False,
) -> list[dict]:
    """Generate responses using Unsloth (slow, use_cache=False)."""
    import json
    import time
    from pathlib import Path

    import torch
    from peft import PeftModel

    # ----- Checkpoint setup -----
    checkpoint_dir = Path(f"/adapters/eval-checkpoints/{tag}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    completed = {}
    if not fresh:
        for f in checkpoint_dir.glob("*.json"):
            try:
                r = json.loads(f.read_text())
                completed[r["id"]] = r
            except (json.JSONDecodeError, KeyError):
                pass
        if completed:
            print(f"Resuming: {len(completed)}/{len(eval_prompts)} already completed")

    remaining = [p for p in eval_prompts if p["id"] not in completed]
    if not remaining:
        print("All prompts already completed.")
        return list(completed.values())

    print(f"{len(remaining)} prompts to generate ({len(completed)} cached)")

    # ----- Load model -----
    print(f"Loading {base_model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    adapter_path = f"/adapters/experiments/{adapter_name}"
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    FastLanguageModel.for_inference(model)
    print(f"Model ready. Generating {len(remaining)} responses...")

    # ----- Generate with checkpointing -----
    for i, p in enumerate(remaining):
        prompt_text = p["prompt"]
        print(f"[{len(completed) + i + 1}/{len(eval_prompts)}] {p['id']}: {prompt_text[:60]}...")

        conversation = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]

        inputs = tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=False,
            )

        new_tokens = outputs[0][inputs.shape[1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        elapsed = time.time() - start

        result = {
            "id": p["id"],
            "category": p["category"],
            "difficulty": p.get("difficulty", "medium"),
            "prompt": prompt_text,
            "ground_truth_signal": p.get("ground_truth_signal", ""),
            "response": response_text,
            "generation_time": round(elapsed, 1),
            "model": adapter_name,
            "prompt_tokens": inputs.shape[1],
            "completion_tokens": len(new_tokens),
        }

        checkpoint_path = checkpoint_dir / f"{p['id']}.json"
        checkpoint_path.write_text(json.dumps(result, indent=2))
        adapter_vol.commit()

        completed[p["id"]] = result
        print(f"  {len(new_tokens)} tokens in {elapsed:.1f}s [checkpointed]")

    all_results = [completed[p["id"]] for p in eval_prompts if p["id"] in completed]
    print(f"\nDone. {len(all_results)} total responses.")
    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    adapter: str = "madison-orpo-v3b-lr2e5",
    tag: str = "orpo-v3b",
    eval_prompts: str = "data/eval/eval-prompts.jsonl",
    fresh: bool = False,
    use_unsloth: bool = False,
):
    """Generate eval responses on Modal, save locally."""
    import json
    from pathlib import Path

    prompts = []
    with open(eval_prompts) as f:
        for line in f:
            prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} eval prompts")

    if use_unsloth:
        print(f"Using Unsloth fallback (slow, ~160-200s/prompt)...")
        responses = generate_unsloth.remote(
            eval_prompts=prompts,
            adapter_name=adapter,
            tag=tag,
            fresh=fresh,
        )
    else:
        merged_path = f"/adapters/merged/{adapter}-16bit"
        print(f"Using vLLM with merged model at {merged_path}...")
        print("(Run scripts/modal/merge_model.py first if this fails)")
        responses = generate_vllm.remote(
            eval_prompts=prompts,
            tag=tag,
            merged_model_path=merged_path,
            fresh=fresh,
        )

    # Save responses locally
    output_dir = Path("data/eval/responses")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"responses-{tag}.jsonl"

    with open(output_path, "w") as f:
        for r in responses:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved {len(responses)} responses to {output_path}")
    print(f"\nSample response:")
    print(f"  Prompt: {responses[0]['prompt'][:80]}...")
    print(f"  Response: {responses[0]['response'][:200]}...")

    total_tokens = sum(r["completion_tokens"] for r in responses)
    total_time = sum(r["generation_time"] for r in responses)
    if total_time > 0:
        print(f"\nTotal: {total_tokens} tokens in {total_time:.0f}s ({total_tokens/total_time:.1f} tok/s)")
