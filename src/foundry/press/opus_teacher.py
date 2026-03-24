"""Generate teacher responses using Opus 4.6 via the Anthropic API.

Uses the same Anthropic API key that Claude Code runs on. Reads prompts,
sends each to Claude with the Madison constitution as system prompt,
and saves responses to JSONL.

Usage:
    python -m foundry.press.opus_teacher [--start N] [--limit N] [--batch-size N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import httpx

log = logging.getLogger("foundry.press.opus_teacher")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONSTITUTION_PATH = PROJECT_ROOT / "config" / "constitutions" / "madison-5k.md"
PROMPTS_PATH = PROJECT_ROOT / "data" / "training" / "prompts.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "training" / "teacher-responses.jsonl"

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-opus-4-6"


def load_constitution() -> str:
    """Load the Madison constitution, keeping only character content."""
    text = CONSTITUTION_PATH.read_text()
    lines = text.split("\n")
    content_lines = []
    in_content = False
    for line in lines:
        if line.startswith("## 1."):
            in_content = True
        if in_content:
            content_lines.append(line)
    return "\n".join(content_lines)


def load_prompts(path: Path | None = None) -> list[dict]:
    path = path or PROMPTS_PATH
    prompts = []
    with open(path) as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def count_existing(output_path: Path) -> int:
    """Count how many responses already exist (for resuming)."""
    if not output_path.exists():
        return 0
    with open(output_path) as f:
        return sum(1 for _ in f)


def generate_response(
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
) -> str:
    """Generate a response using the Anthropic Messages API."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 2048,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.85,
        "top_p": 0.92,
    }
    response = httpx.post(ANTHROPIC_API_URL, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["content"][0]["text"]


def main():
    parser = argparse.ArgumentParser(description="Generate teacher responses via Anthropic API (Opus)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompts", default=str(PROMPTS_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--start", type=int, default=-1, help="Start from prompt N (-1 = auto-resume)")
    parser.add_argument("--limit", type=int, default=0, help="Process only N prompts (0=all)")
    parser.add_argument("--batch-size", type=int, default=50, help="Log progress every N prompts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set. Export it or set in .env")
        return

    constitution = load_constitution()
    log.info("Constitution: %d chars", len(constitution))

    prompts = load_prompts(Path(args.prompts))
    log.info("Total prompts: %d", len(prompts))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-resume: skip prompts we've already generated
    if args.start == -1:
        existing = count_existing(output_path)
        if existing > 0:
            log.info("Auto-resuming: %d responses already exist, starting from %d", existing, existing)
            args.start = existing
        else:
            args.start = 0

    prompts = prompts[args.start:]
    if args.limit > 0:
        prompts = prompts[:args.limit]

    log.info("Processing %d prompts (start=%d, model=%s)", len(prompts), args.start, args.model)

    mode = "a" if args.start > 0 else "w"
    completed = 0
    failed = 0
    total_time = 0.0

    with open(output_path, mode) as f:
        for i, prompt_data in enumerate(prompts):
            prompt_text = prompt_data["prompt"]
            start = time.time()

            try:
                response = generate_response(prompt_text, constitution, args.model, api_key)
                elapsed = time.time() - start
                total_time += elapsed

                record = {
                    "prompt": prompt_text,
                    "response": response,
                    "theme": prompt_data.get("theme", "unknown"),
                    "register": prompt_data.get("register", "unknown"),
                    "model": args.model,
                    "backend": "anthropic",
                    "generation_time": round(elapsed, 1),
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                completed += 1

                if completed % args.batch_size == 0 or completed == len(prompts):
                    avg = total_time / completed
                    remaining = (len(prompts) - completed) * avg
                    log.info(
                        "[%d/%d] avg %.1fs/prompt, ~%.0f min remaining",
                        args.start + completed,
                        args.start + len(prompts),
                        avg,
                        remaining / 60,
                    )

            except Exception as e:
                elapsed = time.time() - start
                failed += 1
                log.error("[%d] Failed: %s (%.1fs)", args.start + i, e, elapsed)
                # Rate limit handling
                if "rate" in str(e).lower() or "429" in str(e):
                    log.info("Rate limited — waiting 60s")
                    time.sleep(60)

    log.info(
        "Done: %d completed, %d failed, %.1f min total (avg %.1fs/prompt)",
        completed,
        failed,
        total_time / 60,
        total_time / max(completed, 1),
    )


if __name__ == "__main__":
    main()
