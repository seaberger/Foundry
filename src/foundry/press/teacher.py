"""Generate teacher (chosen) responses for DPO training.

Sends each prompt to a teacher model with the Madison constitution as the
system prompt. The teacher generates in-character Madison responses that
become the "chosen" side of DPO pairs.

Supports multiple backends:
  - local: OpenAI-compatible API (LM Studio, llama.cpp, Ollama)
  - gemini: Google Gemini API
  - anthropic: Anthropic Messages API (Claude Sonnet/Opus)

Usage:
    python -m foundry.press.teacher [--backend local] [--model MODEL] [--endpoint URL]
    python -m foundry.press.teacher --backend gemini --model gemini-2.5-flash
    python -m foundry.press.teacher --backend anthropic --model claude-opus-4-6
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import httpx

from .utils import (
    PROJECT_ROOT,
    CONSTITUTION_PATH,
    ANTHROPIC_API_URL,
    ANTHROPIC_VERSION,
    load_constitution,
    load_jsonl,
)

log = logging.getLogger("foundry.press.teacher")

PROMPTS_PATH = PROJECT_ROOT / "data" / "training" / "prompts.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "training" / "teacher-responses.jsonl"

DEFAULT_LOCAL_ENDPOINT = "http://192.168.4.28:1234/v1"
DEFAULT_LOCAL_MODEL = "qwen3-32b-mlx"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-6"

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai"


def load_prompts(path: Path | None = None) -> list[dict]:
    """Load prompts from JSONL."""
    return load_jsonl(path or PROMPTS_PATH)


def generate_response_local(
    prompt: str,
    system_prompt: str,
    endpoint: str,
    model: str,
) -> str:
    """Generate a response using a local OpenAI-compatible API."""
    url = f"{endpoint.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.85,
        "top_p": 0.92,
        "max_tokens": 1024,
    }
    response = httpx.post(url, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]

    # Strip thinking traces if present (Qwen thinking models)
    if "<think>" in content:
        # Remove everything between <think> and </think>
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    return content


def generate_response_gemini(
    prompt: str,
    system_prompt: str,
    model: str,
) -> str:
    """Generate a response using the Gemini API (OpenAI-compatible mode)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    url = f"{GEMINI_ENDPOINT}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.85,
        "top_p": 0.92,
        "max_tokens": 2048,
    }
    response = httpx.post(url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def generate_response_anthropic(
    prompt: str,
    system_prompt: str,
    model: str,
) -> str:
    """Generate a response using the Anthropic Messages API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 2048,
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.85,
        "top_p": 0.92,
    }
    response = httpx.post(ANTHROPIC_API_URL, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["content"][0]["text"]


def main():
    parser = argparse.ArgumentParser(description="Generate teacher responses for DPO training")
    parser.add_argument("--backend", choices=["local", "gemini", "anthropic"], default="local")
    parser.add_argument("--endpoint", default=DEFAULT_LOCAL_ENDPOINT, help="Local API endpoint")
    parser.add_argument("--model", default=None, help="Model name (defaults per backend)")
    parser.add_argument("--prompts", default=str(PROMPTS_PATH), help="Input prompts JSONL")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output JSONL")
    parser.add_argument("--start", type=int, default=0, help="Start from prompt N (for resuming)")
    parser.add_argument("--limit", type=int, default=0, help="Process only N prompts (0=all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    # Set default model per backend
    if args.model is None:
        if args.backend == "local":
            args.model = DEFAULT_LOCAL_MODEL
        elif args.backend == "gemini":
            args.model = "gemini-2.5-flash"
        elif args.backend == "anthropic":
            args.model = DEFAULT_ANTHROPIC_MODEL

    # Load constitution and prompts
    constitution = load_constitution()
    log.info("Loaded constitution: %d chars", len(constitution))

    prompts = load_prompts(Path(args.prompts))
    log.info("Loaded %d prompts", len(prompts))

    # Apply start/limit
    if args.start > 0:
        prompts = prompts[args.start:]
        log.info("Starting from prompt %d", args.start)
    if args.limit > 0:
        prompts = prompts[:args.limit]
        log.info("Limited to %d prompts", args.limit)

    # Generate responses
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Append mode for resuming
    mode = "a" if args.start > 0 else "w"
    completed = 0
    failed = 0
    total_time = 0.0

    with open(output_path, mode) as f:
        for i, prompt_data in enumerate(prompts):
            prompt_text = prompt_data["prompt"]
            start = time.time()

            try:
                if args.backend == "local":
                    response = generate_response_local(
                        prompt_text, constitution, args.endpoint, args.model
                    )
                elif args.backend == "gemini":
                    response = generate_response_gemini(
                        prompt_text, constitution, args.model
                    )
                elif args.backend == "anthropic":
                    response = generate_response_anthropic(
                        prompt_text, constitution, args.model
                    )

                elapsed = time.time() - start
                total_time += elapsed

                record = {
                    "prompt": prompt_text,
                    "response": response,
                    "theme": prompt_data.get("theme", "unknown"),
                    "register": prompt_data.get("register", "unknown"),
                    "model": args.model,
                    "backend": args.backend,
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
                log.error("  [%d] Failed on '%s...': %s (%.1fs)", i, prompt_text[:50], e, elapsed)
                if "rate" in str(e).lower() or "429" in str(e):
                    log.info("Rate limited — waiting 60s")
                    time.sleep(60)

    log.info(
        "Done: %d completed, %d failed, %.1f min total",
        completed,
        failed,
        total_time / 60,
    )


if __name__ == "__main__":
    main()
