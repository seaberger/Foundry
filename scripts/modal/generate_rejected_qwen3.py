"""Generate rejected responses from base Qwen 3-32B on Modal.

Runs all voice prompts through base Qwen 3-32B (no adapter) to produce
on-policy rejected responses for ORPO v5 training. These replace the
Gemma 3 rejected data from v4, capturing Qwen 3's actual failure modes
(Chinese character leaks, different hallucination patterns).

Usage:
    cd ~/Repositories/Foundry
    modal run scripts/modal/generate_rejected_qwen3.py
    modal run scripts/modal/generate_rejected_qwen3.py --fresh
"""

from __future__ import annotations

import modal

MINUTES = 60

app = modal.App("foundry-qwen3-rejected")

model_cache_vol = modal.Volume.from_name("foundry-model-cache", create_if_missing=True)
adapter_vol = modal.Volume.from_name("foundry-adapters", create_if_missing=True)

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("vllm>=0.12.0")
    .env({"HF_HOME": "/model_cache"})
)


@app.function(
    image=vllm_image,
    gpu="A100-80GB",
    timeout=360 * MINUTES,
    volumes={
        "/model_cache": model_cache_vol,
        "/adapters": adapter_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_rejected(
    prompts: list[dict],
    base_model: str = "Qwen/Qwen3-32B",
    max_tokens: int = 1024,
    fresh: bool = False,
) -> list[dict]:
    """Generate responses from base Qwen 3-32B with no adapter."""
    import json
    import time
    from pathlib import Path

    from vllm import LLM, SamplingParams

    checkpoint_dir = Path("/adapters/rejected-checkpoints/qwen3-base")
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
            print(f"Resuming: {len(completed)}/{len(prompts)} already completed")

    remaining = [p for p in prompts if p["id"] not in completed]
    if not remaining:
        print("All prompts already completed.")
        return list(completed.values())

    print(f"{len(remaining)} prompts to generate ({len(completed)} cached)")

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
        print(f"[{len(completed) + i + 1}/{len(prompts)}] {p['id']}: {prompt_text[:60]}...")

        conversation = (
            f"<|im_start|>system\n/no_think<|im_end|>\n"
            f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        start = time.time()
        outputs = llm.generate([conversation], sampling_params)
        response_text = outputs[0].outputs[0].text
        elapsed = time.time() - start

        # Strip empty think tags
        import re
        response_text = re.sub(r"<think>\s*</think>\s*", "", response_text).strip()

        num_tokens = len(outputs[0].outputs[0].token_ids)

        result = {
            "id": p["id"],
            "category": p.get("category", "voice"),
            "prompt": prompt_text,
            "response": response_text,
            "generation_time": round(elapsed, 1),
            "model": "qwen3-32b-base",
            "completion_tokens": num_tokens,
        }

        checkpoint_path = checkpoint_dir / f"{p['id']}.json"
        checkpoint_path.write_text(json.dumps(result, indent=2))

        if (i + 1) % 20 == 0:
            adapter_vol.commit()

        completed[p["id"]] = result
        print(f"  {num_tokens} tokens in {elapsed:.1f}s")

    adapter_vol.commit()

    all_results = [completed[p["id"]] for p in prompts if p["id"] in completed]
    print(f"\nDone. {len(all_results)} total responses.")
    return all_results


@app.local_entrypoint()
def main(
    prompts_file: str = "data/training/voice-prompts.jsonl",
    output_file: str = "data/training/rejected-qwen3-base.jsonl",
    fresh: bool = False,
):
    """Generate rejected responses from base Qwen 3-32B on Modal."""
    import json
    from pathlib import Path

    prompts = []
    with open(prompts_file) as f:
        for i, line in enumerate(f):
            p = json.loads(line)
            if "id" not in p:
                p["id"] = f"v5-{i:03d}"
            prompts.append(p)
    print(f"Loaded {len(prompts)} prompts from {prompts_file}")

    responses = generate_rejected.remote(prompts=prompts, fresh=fresh)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in responses:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved {len(responses)} rejected responses to {output_path}")
    if responses:
        total_tokens = sum(r["completion_tokens"] for r in responses)
        total_time = sum(r["generation_time"] for r in responses)
        if total_time > 0:
            print(f"Total: {total_tokens} tokens in {total_time:.0f}s ({total_tokens/total_time:.1f} tok/s)")
