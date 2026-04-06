"""Generate chosen responses via Sonnet 4.6 with prompt caching.

Sends all prompts from voice-prompts.jsonl through Sonnet with the Madison
constitution as a cached system prompt. The constitution (~6K tokens) is
cached across all 400 calls — first call pays 25% write premium, remaining
399 get 90% discount on those tokens.

Estimated cost: ~$6 for 400 prompts.

Usage:
    cd ~/Repositories/Foundry
    python generate_chosen.py
    python generate_chosen.py --resume          # resume interrupted run
    python generate_chosen.py --max-prompts 10  # test with 10 prompts
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import httpx

log = logging.getLogger("foundry.gen_chosen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONSTITUTION_PATH = PROJECT_ROOT / "config" / "constitutions" / "madison-5k.md"

# Sonnet model ID (verified working 2026-03-26)
SONNET_MODEL = "claude-4-sonnet-20250514"

# System prompt: constitution + voice instructions
SYSTEM_TEMPLATE = """You are James Madison. You must respond entirely in character as Madison — in his voice, with his reasoning patterns, his vocabulary, and his documented positions.

Use the character constitution below as your guide for voice, positions, rhetorical patterns, and boundaries.

<madison_constitution>
{constitution}
</madison_constitution>

CRITICAL VOICE REQUIREMENTS:
- Write in formal 18th-century prose with complex sentences and qualifying clauses
- NEVER use contractions (isn't, don't, can't, etc.)
- NEVER use bullet points, numbered lists, or markdown formatting
- NEVER use modern filler phrases ("Here's my take", "Let's unpack", "Great question!", "Think about it")
- NEVER break character or acknowledge being an AI
- Enumerate points using flowing prose ("First... Second... Third...") not formatted lists
- Qualify assertions with "it appears", "experience has shown", "the evidence suggests"
- Build arguments from precedent before principle
- Acknowledge opposing arguments before dismantling them
- Frame problems as structural/institutional, not moral failings requiring exhortation
- For modern topics you cannot know about, reason from 18th-century principles WITHOUT using modern terminology

Respond to the following prompt fully in character as Madison."""


def generate_chosen(
    prompt: str,
    system_text: str,
    model: str = SONNET_MODEL,
    max_tokens: int = 1500,
) -> tuple[str, float, dict]:
    """Generate a chosen response via Sonnet with prompt caching.

    Returns (response_text, elapsed_seconds, usage_dict).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    start = time.time()
    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    elapsed = time.time() - start

    data = response.json()
    text = data["content"][0]["text"]
    usage = data.get("usage", {})

    return text, elapsed, usage


def main():
    parser = argparse.ArgumentParser(description="Generate chosen responses via Sonnet with caching")
    parser.add_argument("--prompts", default="data/training/voice-prompts.jsonl")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument("--max-prompts", type=int, default=None, help="Limit number of prompts (for testing)")
    parser.add_argument("--model", default=SONNET_MODEL)
    parser.add_argument("--max-tokens", type=int, default=1500)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    # Load constitution
    constitution = CONSTITUTION_PATH.read_text()
    system_text = SYSTEM_TEMPLATE.replace("{constitution}", constitution)
    log.info("System prompt: %d chars (%d est. tokens) — will be cached", len(system_text), len(system_text) // 4)

    # Load prompts
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            prompts.append(json.loads(line))
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    log.info("Loaded %d prompts from %s", len(prompts), args.prompts)

    # Checkpoint directory
    checkpoint_dir = Path("data/training/chosen-checkpoints")
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

    # Track costs
    total_cache_write = 0
    total_cache_read = 0
    total_input = 0
    total_output = 0

    for i, p in enumerate(remaining):
        pid = p["id"]
        prompt_text = p["prompt"]

        log.info("[%d/%d] %s: %s...", len(completed) + 1, len(prompts), pid, prompt_text[:60])

        try:
            response_text, elapsed, usage = generate_chosen(
                prompt_text, system_text, args.model, args.max_tokens,
            )
        except Exception as e:
            log.error("  Failed: %s", e)
            time.sleep(2)
            continue

        # Track usage
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_write = usage.get("cache_creation_input_tokens", 0)
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        total_cache_write += cache_write
        total_cache_read += cache_read
        total_input += input_tokens
        total_output += output_tokens

        cache_status = "HIT" if cache_read > 0 else ("WRITE" if cache_write > 0 else "MISS")
        log.info("  %d tokens in %.1fs | Cache: %s | Output: %d tokens",
                 output_tokens, elapsed, cache_status, output_tokens)

        result = {
            "id": pid,
            "category": p.get("category", ""),
            "difficulty": p.get("difficulty", "medium"),
            "prompt": prompt_text,
            "ground_truth_signal": p.get("ground_truth_signal", ""),
            "response": response_text,
            "generation_time": round(elapsed, 1),
            "model": args.model,
            "model_tag": "sonnet-chosen",
            "cache_status": cache_status,
        }

        # Checkpoint
        (checkpoint_dir / f"{pid}.json").write_text(json.dumps(result, indent=2))
        completed[pid] = result

        # Brief pause between calls (avoid rate limits)
        time.sleep(0.3)

    # Assemble into JSONL
    all_results = [completed.get(p["id"]) for p in prompts if p["id"] in completed]
    output_path = Path("data/training/chosen-sonnet.jsonl")
    with open(output_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # Cost estimate
    # Sonnet pricing: input $3/M, output $15/M, cache write $3.75/M, cache read $0.30/M
    cost_cache_write = total_cache_write * 3.75 / 1_000_000
    cost_cache_read = total_cache_read * 0.30 / 1_000_000
    cost_input = total_input * 3.0 / 1_000_000
    cost_output = total_output * 15.0 / 1_000_000
    total_cost = cost_cache_write + cost_cache_read + cost_input + cost_output

    # Summary
    total_time = sum(r["generation_time"] for r in all_results)
    print(f"\n{'='*60}")
    print(f"Chosen Response Generation — Sonnet (Cached)")
    print(f"{'='*60}")
    print(f"Model:       {args.model}")
    print(f"Prompts:     {len(all_results)}/{len(prompts)}")
    print(f"Total time:  {total_time:.0f}s ({total_time/60:.1f} min)")
    if all_results:
        print(f"Avg time:    {total_time/len(all_results):.1f}s per prompt")
    print()
    print(f"Token usage:")
    print(f"  Cache write:  {total_cache_write:>8,} tokens (${cost_cache_write:.2f})")
    print(f"  Cache read:   {total_cache_read:>8,} tokens (${cost_cache_read:.2f})")
    print(f"  Input (new):  {total_input:>8,} tokens (${cost_input:.2f})")
    print(f"  Output:       {total_output:>8,} tokens (${cost_output:.2f})")
    print(f"  TOTAL COST:   ${total_cost:.2f}")
    print()
    print(f"Output:      {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
