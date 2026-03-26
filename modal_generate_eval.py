"""Generate evaluation responses from fine-tuned Madison model on Modal.

Loads the ORPO adapter, runs all eval prompts, saves responses locally.
Then run the judge separately with evaluate.py.

Usage:
    # Generate responses from fine-tuned model
    modal run modal_generate_eval.py --adapter madison-orpo-v3b-lr2e5 --tag orpo-v3b

    # Then judge locally (uses Anthropic API for Sonnet judge)
    .venv/bin/python -m foundry.press.evaluate_judge --responses data/eval/responses/orpo-v3b.jsonl --tag orpo-v3b
"""

from __future__ import annotations

import modal

MINUTES = 60
GPU = "A100"

app = modal.App("foundry-madison-eval")

model_cache_vol = modal.Volume.from_name("foundry-model-cache", create_if_missing=True)
adapter_vol = modal.Volume.from_name("foundry-adapters", create_if_missing=True)

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.9.0",
        "huggingface_hub==0.34.2",
        "peft==0.16.0",
        "transformers==4.54.0",
        "trl==0.19.1",
        "unsloth[cu128-torch270]==2025.7.8",
        "unsloth_zoo==2025.7.10",
    )
    .env({"HF_HOME": "/model_cache"})
)

with eval_image.imports():
    import unsloth  # noqa: F401,I001
    from unsloth import FastLanguageModel


@app.function(
    image=eval_image,
    gpu=GPU,
    timeout=60 * MINUTES,
    volumes={
        "/model_cache": model_cache_vol,
        "/adapters": adapter_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_responses(
    eval_prompts: list[dict],
    adapter_name: str = "madison-orpo-v3b-lr2e5",
    base_model: str = "google/gemma-3-27b-it",
    max_seq_length: int = 2048,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
) -> list[dict]:
    """Load model + adapter, generate responses for all eval prompts."""
    import time

    import torch
    from peft import PeftModel

    # Load base model
    print(f"Loading {base_model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Load LoRA adapter
    adapter_path = f"/adapters/experiments/{adapter_name}"
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    FastLanguageModel.for_inference(model)
    print(f"Model ready. Generating {len(eval_prompts)} responses...")

    results = []
    for i, p in enumerate(eval_prompts):
        prompt_text = p["prompt"]
        print(f"[{i+1}/{len(eval_prompts)}] {p['id']}: {prompt_text[:60]}...")

        # Build conversation — Gemma 3 expects structured content for multimodal template
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
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                use_cache=False,
            )

        new_tokens = outputs[0][inputs.shape[1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        elapsed = time.time() - start

        results.append({
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
        })

        print(f"  {len(new_tokens)} tokens in {elapsed:.1f}s")

    print(f"\nDone. Generated {len(results)} responses.")
    return results


@app.local_entrypoint()
def main(
    adapter: str = "madison-orpo-v3b-lr2e5",
    tag: str = "orpo-v3b",
    eval_prompts: str = "data/eval/eval-prompts.jsonl",
):
    """Generate eval responses on Modal, save locally, then run judge."""
    import json
    import time
    from pathlib import Path

    # Load eval prompts
    prompts = []
    with open(eval_prompts) as f:
        for line in f:
            prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} eval prompts")

    # Generate responses on Modal
    print(f"Generating responses with adapter '{adapter}' on A100...")
    responses = generate_responses.remote(
        eval_prompts=prompts,
        adapter_name=adapter,
    )

    # Save responses locally
    output_dir = Path("data/eval/responses")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"responses-{tag}-{timestamp}.jsonl"

    with open(output_path, "w") as f:
        for r in responses:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved {len(responses)} responses to {output_path}")
    print(f"\nSample response (first prompt):")
    print(f"  Prompt: {responses[0]['prompt'][:80]}...")
    print(f"  Response: {responses[0]['response'][:200]}...")

    # Print summary
    total_tokens = sum(r["completion_tokens"] for r in responses)
    total_time = sum(r["generation_time"] for r in responses)
    print(f"\nTotal: {total_tokens} tokens in {total_time:.0f}s ({total_tokens/total_time:.1f} tok/s)")
    print(f"\nNext step: judge these responses with the eval harness")
    print(f"  .venv/bin/python -m foundry.press.evaluate --responses {output_path} --tag {tag}")
