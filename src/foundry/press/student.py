"""Generate student (rejected) responses for DPO training.

Sends each prompt to the BASE model (Gemma 3 27B) with NO constitution or
persona instruction. These plain responses become the "what Madison wouldn't
say" examples — the rejected side of DPO pairs.

Usage:
    python -m foundry.press.student [--model MODEL] [--endpoint URL]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import httpx

log = logging.getLogger("foundry.press.student")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROMPTS_PATH = PROJECT_ROOT / "data" / "training" / "prompts.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "training" / "student-responses.jsonl"

DEFAULT_ENDPOINT = "http://192.168.4.28:1234/v1"
DEFAULT_MODEL = "google/gemma-3-27b"  # The actual fine-tuning target — no persona


def load_prompts(path: Path | None = None) -> list[dict]:
    """Load prompts from JSONL."""
    path = path or PROMPTS_PATH
    prompts = []
    with open(path) as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def generate_response(
    prompt: str,
    endpoint: str,
    model: str,
) -> str:
    """Generate a plain response — NO system prompt, NO persona."""
    url = f"{endpoint.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            # No system prompt — the student responds as a generic assistant
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.85,
        "top_p": 0.92,
        "max_tokens": 1024,
    }
    response = httpx.post(url, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="Generate student (rejected) responses for DPO")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="LLM API endpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name (should be fine-tune target)")
    parser.add_argument("--prompts", default=str(PROMPTS_PATH), help="Input prompts JSONL")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output JSONL")
    parser.add_argument("--start", type=int, default=0, help="Start from prompt N (for resuming)")
    parser.add_argument("--limit", type=int, default=0, help="Process only N prompts (0=all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    prompts = load_prompts(Path(args.prompts))
    log.info("Loaded %d prompts (model: %s — NO persona)", len(prompts), args.model)

    if args.start > 0:
        prompts = prompts[args.start:]
    if args.limit > 0:
        prompts = prompts[:args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.start > 0 else "w"
    completed = 0
    failed = 0
    total_time = 0.0

    with open(output_path, mode) as f:
        for i, prompt_data in enumerate(prompts):
            prompt_text = prompt_data["prompt"]
            start = time.time()

            try:
                response = generate_response(prompt_text, args.endpoint, args.model)
                elapsed = time.time() - start
                total_time += elapsed

                record = {
                    "prompt": prompt_text,
                    "response": response,
                    "theme": prompt_data.get("theme", "unknown"),
                    "register": prompt_data.get("register", "unknown"),
                    "model": args.model,
                    "generation_time": round(elapsed, 1),
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                completed += 1

                if completed % 10 == 0:
                    avg = total_time / completed
                    remaining = (len(prompts) - completed) * avg
                    log.info(
                        "  [%d/%d] avg %.1fs/prompt, ~%.0f min remaining",
                        completed,
                        len(prompts),
                        avg,
                        remaining / 60,
                    )

            except Exception as e:
                elapsed = time.time() - start
                failed += 1
                log.error("  [%d] Failed on '%s...': %s", i, prompt_text[:50], e)

    log.info(
        "Done: %d completed, %d failed, %.1f min total",
        completed,
        failed,
        total_time / 60,
    )


if __name__ == "__main__":
    main()
