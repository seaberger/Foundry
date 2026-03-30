"""Generate evaluation responses from Qwen 3-32B + LoRA adapter on Modal.

Uses vLLM with LoRA serving (adapter-on-base, no merge needed).
This is the production serving path — validates vLLM + LoRA works cleanly
for Qwen 3 with zero workarounds.

Usage:
    modal run modal_generate_eval_qwen.py --tag qwen3-val-v1
    modal run modal_generate_eval_qwen.py --tag qwen3-val-v1 --fresh
    modal run modal_generate_eval_qwen.py --tag qwen3-val-v1 --num-prompts 36
"""

from __future__ import annotations

import modal

MINUTES = 60

app = modal.App("foundry-qwen3-eval")

model_cache_vol = modal.Volume.from_name("foundry-model-cache", create_if_missing=True)
adapter_vol = modal.Volume.from_name("foundry-adapters", create_if_missing=True)

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "vllm>=0.12.0",
    )
    .env({"HF_HOME": "/model_cache"})
)


# ---------------------------------------------------------------------------
# vLLM LoRA serving generation
# ---------------------------------------------------------------------------

@app.function(
    image=vllm_image,
    gpu="A100-80GB",
    timeout=30 * MINUTES,
    volumes={
        "/model_cache": model_cache_vol,
        "/adapters": adapter_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_vllm_lora(
    eval_prompts: list[dict],
    tag: str = "qwen3-val-v1",
    base_model: str = "Qwen/Qwen3-32B",
    adapter_name: str = "madison-qwen3-val-v1",
    max_tokens: int = 1024,
    fresh: bool = False,
) -> list[dict]:
    """Generate responses using vLLM with LoRA serving on Qwen 3-32B."""
    import json
    import time
    from pathlib import Path

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

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

    # ----- Load base model with vLLM + LoRA support -----
    adapter_path = f"/adapters/experiments/{adapter_name}"
    print(f"Loading {base_model} with LoRA serving enabled...")
    print(f"Adapter path: {adapter_path}")

    # Qwen 3-32B is pure ForCausalLM — no multimodal workarounds needed
    llm = LLM(
        model=base_model,
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        dtype="auto",
        enable_lora=True,
        max_lora_rank=64,  # Match training rank
    )

    lora_request = LoRARequest("madison", 1, adapter_path)

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

        # Qwen 3 ChatML format — /no_think suppresses <think> reasoning traces
        conversation = (
            f"<|im_start|>system\n/no_think<|im_end|>\n"
            f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        start = time.time()
        outputs = llm.generate([conversation], sampling_params, lora_request=lora_request)
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
            "model": f"qwen3-32b+{adapter_name}",
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
# Baseline generation (no adapter — measures the delta)
# ---------------------------------------------------------------------------

@app.function(
    image=vllm_image,
    gpu="A100-80GB",
    timeout=30 * MINUTES,
    volumes={
        "/model_cache": model_cache_vol,
        "/adapters": adapter_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_baseline(
    eval_prompts: list[dict],
    tag: str = "qwen3-baseline",
    base_model: str = "Qwen/Qwen3-32B",
    max_tokens: int = 1024,
    fresh: bool = False,
) -> list[dict]:
    """Generate responses from base Qwen 3-32B with NO adapter — the control."""
    import json
    import time
    from pathlib import Path

    from vllm import LLM, SamplingParams

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

    print(f"{len(remaining)} baseline prompts to generate ({len(completed)} cached)")

    llm = LLM(
        model=base_model,
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        dtype="auto",
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_k=64,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    for i, p in enumerate(remaining):
        prompt_text = p["prompt"]
        print(f"[{len(completed) + i + 1}/{len(eval_prompts)}] {p['id']}: {prompt_text[:60]}...")

        conversation = (
            f"<|im_start|>system\n/no_think<|im_end|>\n"
            f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

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
            "model": "qwen3-32b-baseline",
            "prompt_tokens": 0,
            "completion_tokens": num_tokens,
        }

        checkpoint_path = checkpoint_dir / f"{p['id']}.json"
        checkpoint_path.write_text(json.dumps(result, indent=2))
        adapter_vol.commit()

        completed[p["id"]] = result
        print(f"  {num_tokens} tokens in {elapsed:.1f}s [checkpointed]")

    all_results = [completed[p["id"]] for p in eval_prompts if p["id"] in completed]
    print(f"\nDone. {len(all_results)} baseline responses.")
    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    adapter: str = "madison-qwen3-val-v1",
    tag: str = "qwen3-val-v1",
    eval_prompts: str = "../../data/eval/eval-prompts.jsonl",
    num_prompts: int = 12,
    fresh: bool = False,
    baseline: bool = False,
    base_model: str = "Qwen/Qwen3-32B",
):
    """Generate eval responses from Qwen 3-32B + LoRA on Modal, save locally.

    Usage:
        modal run modal_generate_eval_qwen.py --tag qwen3-val-v1
        modal run modal_generate_eval_qwen.py --tag qwen3-val-v1 --baseline
        modal run modal_generate_eval_qwen.py --tag qwen3-val-v1 --num-prompts 36
    """
    import json
    from pathlib import Path

    all_prompts = []
    with open(eval_prompts) as f:
        for line in f:
            all_prompts.append(json.loads(line))

    # Select a subset: 2 from each of the 6 categories for balanced coverage
    if num_prompts < len(all_prompts):
        from collections import defaultdict
        by_cat = defaultdict(list)
        for p in all_prompts:
            by_cat[p["category"]].append(p)
        prompts = []
        per_cat = max(1, num_prompts // len(by_cat))
        for cat, cat_prompts in sorted(by_cat.items()):
            prompts.extend(cat_prompts[:per_cat])
        # Fill remaining slots
        remaining = [p for p in all_prompts if p not in prompts]
        while len(prompts) < num_prompts and remaining:
            prompts.append(remaining.pop(0))
        prompts = prompts[:num_prompts]
    else:
        prompts = all_prompts

    print(f"Using {len(prompts)} eval prompts from {len(all_prompts)} total")
    cats = {}
    for p in prompts:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    for cat, n in sorted(cats.items()):
        print(f"  {cat}: {n}")

    output_dir = Path("../../data/eval/responses")
    output_dir.mkdir(parents=True, exist_ok=True)

    if baseline:
        # Generate baseline (no adapter) for comparison
        print(f"\nGenerating BASELINE responses (no adapter)...")
        baseline_tag = f"{tag}-baseline"
        baseline_responses = generate_baseline.remote(
            eval_prompts=prompts,
            tag=baseline_tag,
            fresh=fresh,
        )
        baseline_path = output_dir / f"responses-{baseline_tag}.jsonl"
        with open(baseline_path, "w") as f:
            for r in baseline_responses:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(baseline_responses)} baseline responses to {baseline_path}")
        if baseline_responses:
            print(f"  Sample: {baseline_responses[0]['response'][:200]}...")
    else:
        # Generate fine-tuned responses via LoRA serving
        print(f"\nUsing vLLM LoRA serving (no merge needed)...")
        responses = generate_vllm_lora.remote(
            eval_prompts=prompts,
            tag=tag,
            base_model=base_model,
            adapter_name=adapter,
            fresh=fresh,
        )
        output_path = output_dir / f"responses-{tag}.jsonl"
        with open(output_path, "w") as f:
            for r in responses:
                f.write(json.dumps(r) + "\n")
        print(f"\nSaved {len(responses)} responses to {output_path}")
        if responses:
            print(f"\nSample response:")
            print(f"  Prompt: {responses[0]['prompt'][:80]}...")
            print(f"  Response: {responses[0]['response'][:300]}...")

            total_tokens = sum(r["completion_tokens"] for r in responses)
            total_time = sum(r["generation_time"] for r in responses)
            if total_time > 0:
                print(f"\nTotal: {total_tokens} tokens in {total_time:.0f}s ({total_tokens/total_time:.1f} tok/s)")
