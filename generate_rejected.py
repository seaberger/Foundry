"""Generate rejected responses by running voice prompts through LM Studio.

Sends all prompts from voice-prompts.jsonl through the model loaded in
LM Studio (either base gemma-3-27b-it or fine-tuned madison-orpo-v3b).
Saves responses with checkpointing for resume support.

Usage:
    cd ~/Repositories/Foundry
    python generate_rejected.py --model-tag base
    python generate_rejected.py --model-tag v3b
    python generate_rejected.py --model-tag v3b --resume  # resume interrupted run
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import httpx

log = logging.getLogger("foundry.gen_rejected")

LMSTUDIO_ENDPOINT = "http://100.81.70.30:1234/v1/chat/completions"


def generate_one(prompt: str, endpoint: str, model: str, max_tokens: int = 1024) -> tuple[str, float]:
    """Send a prompt to LM Studio and return (response_text, elapsed_seconds)."""
    messages = [{"role": "user", "content": prompt}]

    start = time.time()
    response = httpx.post(
        endpoint,
        json={
            "model": model,
            "messages": messages,
            "temperature": 1.0,
            "top_k": 64,
            "top_p": 0.95,
            "max_tokens": max_tokens,
        },
        timeout=300,
    )
    response.raise_for_status()
    elapsed = time.time() - start

    data = response.json()
    text = data["choices"][0]["message"]["content"]
    return text, elapsed


def main():
    parser = argparse.ArgumentParser(description="Generate rejected responses via LM Studio")
    parser.add_argument("--model-tag", required=True, help="Tag for this run (e.g., 'base' or 'v3b')")
    parser.add_argument("--model-id", default=None,
                        help="LM Studio model ID (default: auto-detect from /v1/models)")
    parser.add_argument("--prompts", default="data/training/voice-prompts.jsonl")
    parser.add_argument("--endpoint", default=LMSTUDIO_ENDPOINT)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    # Auto-detect model ID from LM Studio
    model_id = args.model_id
    if not model_id:
        base_url = args.endpoint.rsplit("/v1/", 1)[0]
        models_resp = httpx.get(f"{base_url}/v1/models", timeout=10)
        models = models_resp.json().get("data", [])
        if not models:
            log.error("No models loaded in LM Studio. Load a model first.")
            return
        model_id = models[0]["id"]
        log.info("Auto-detected model: %s", model_id)

    # Load prompts
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            prompts.append(json.loads(line))
    log.info("Loaded %d prompts from %s", len(prompts), args.prompts)

    # Checkpoint directory
    checkpoint_dir = Path(f"data/training/rejected-checkpoints/{args.model_tag}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load existing checkpoints
    completed = {}
    if args.resume:
        for f in sorted(checkpoint_dir.glob("*.json")):
            try:
                r = json.loads(f.read_text())
                completed[r["id"]] = r
            except (json.JSONDecodeError, KeyError):
                pass
        if completed:
            log.info("Resuming: %d/%d already completed", len(completed), len(prompts))

    remaining = [p for p in prompts if p["id"] not in completed]
    if not remaining:
        log.info("All prompts already completed.")
    else:
        log.info("%d prompts to generate (%d cached)", len(remaining), len(completed))

    # Generate responses
    for i, p in enumerate(remaining):
        pid = p["id"]
        prompt_text = p["prompt"]

        log.info("[%d/%d] %s: %s...", len(completed) + 1, len(prompts), pid, prompt_text[:60])

        try:
            response_text, elapsed = generate_one(
                prompt_text, args.endpoint, model_id, args.max_tokens,
            )
        except Exception as e:
            log.error("  Failed: %s", e)
            continue

        result = {
            "id": pid,
            "category": p.get("category", ""),
            "difficulty": p.get("difficulty", "medium"),
            "prompt": prompt_text,
            "ground_truth_signal": p.get("ground_truth_signal", ""),
            "response": response_text,
            "generation_time": round(elapsed, 1),
            "model": model_id,
            "model_tag": args.model_tag,
        }

        # Checkpoint
        (checkpoint_dir / f"{pid}.json").write_text(json.dumps(result, indent=2))
        completed[pid] = result

        tokens_approx = len(response_text.split())
        log.info("  ~%d words in %.1fs [checkpointed]", tokens_approx, elapsed)

    # Assemble into JSONL
    all_results = [completed.get(p["id"]) for p in prompts if p["id"] in completed]
    output_path = Path(f"data/training/rejected-{args.model_tag}.jsonl")
    with open(output_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    log.info("\nSaved %d responses to %s", len(all_results), output_path)

    # Summary
    total_time = sum(r["generation_time"] for r in all_results)
    print(f"\n{'='*60}")
    print(f"Rejected Response Generation — {args.model_tag}")
    print(f"{'='*60}")
    print(f"Model:     {model_id}")
    print(f"Prompts:   {len(all_results)}/{len(prompts)}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    if all_results:
        print(f"Avg time:  {total_time/len(all_results):.1f}s per prompt")
    print(f"Output:    {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
