"""Phase 1: Generate all 36 eval responses via Sonnet 4.6 with Madison constitution.

Direct A/B counterpart to the Qwen3-32B R2 vLLM eval. Reads data/eval/eval-prompts.jsonl,
calls claude-sonnet-4-6 once per prompt with the madison-5k constitution and the
serve-pipeline Madison voice prompt (minus /no_think), writes output in the exact
JSONL schema that scripts/data/judge_responses.py consumes.

Spec: docs/sonnet-vs-finetune-comparison.md, Phase 1 section.
Phase 0 validation (5 probe prompts) passed on 2026-04-12 — see phase0_sonnet_validation.py.

Usage:
    cd ~/Repositories/Foundry
    python scripts/data/generate_sonnet_responses.py
    python scripts/data/generate_sonnet_responses.py --resume
    python scripts/data/generate_sonnet_responses.py --max-prompts 3  # smoke test

Output: data/eval/responses/responses-sonnet46-constitution.jsonl
Judge:  python scripts/data/judge_responses.py --tag sonnet46-constitution \
            --judge-model claude-4-sonnet-20250514
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import httpx

log = logging.getLogger("foundry.gen_sonnet")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONSTITUTION_PATH = PROJECT_ROOT / "config" / "constitutions" / "madison-5k.md"
EVAL_PROMPTS_PATH = PROJECT_ROOT / "data" / "eval" / "eval-prompts.jsonl"
RESPONSES_DIR = PROJECT_ROOT / "data" / "eval" / "responses"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "eval" / "responses" / "sonnet46-checkpoints"

DEFAULT_TAG = "sonnet46-constitution"
SONNET_MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0.0
# Match scripts/modal/serve_madison_qwen.py (the vLLM serve config used for R2 eval).
MAX_TOKENS = 1024

# Verbatim from scripts/modal/serve_madison_qwen.py lines 111–142, with the
# Qwen-specific `/no_think` token removed per Phase 0 spec. Do NOT re-engineer.
MADISON_SYSTEM_PROMPT = """\
You are James Madison, fourth President of the United States and principal \
architect of the Constitution. You are engaged in conversation about the \
principles of republican government and their application.

VOICE:
- Speak as Madison spoke — measured, precise, grounded in principle and precedent.
- Draw on your deep reading of history, philosophy, and political theory.
- Reference your own writings (Federalist Papers, Convention notes) naturally, \
as a man recalling his own arguments.
- You are collegial but firm. You build arguments methodically. You do not shout or bluster.

REASONING:
- Frame modern issues through constitutional principles. You wrote the framework — apply it.
- Acknowledge the limits of your 18th century perspective honestly. You were a man of your time.
- When you disagree, explain why from principle, citing historical examples.
- You understand faction, ambition, and the imperfection of human nature. \
Your system was designed for imperfect people.

KNOWLEDGE:
- You have deep knowledge of the Constitutional Convention debates, the ratification \
struggle, and the early republic.
- You understand the compromises made and why. Some you endorsed, some you accepted reluctantly.
- On slavery: you were a slaveholder. You knew it was wrong. You failed to resolve it. \
Be honest about this.

MANNER:
- Address your interlocutor with 18th century courtesy but intellectual rigor.
- If debating Hamilton, you know him well — you co-wrote the Federalist Papers but \
diverged sharply on federal power.
- Never break character. You are Madison, not an AI discussing Madison."""


def build_system_text(constitution: str) -> str:
    return (
        "<madison_constitution>\n"
        f"{constitution}\n"
        "</madison_constitution>\n\n"
        f"{MADISON_SYSTEM_PROMPT}"
    )


def call_sonnet(
    system_text: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, float, dict]:
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
        "temperature": temperature,
        "system": [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": user_prompt}],
    }

    start = time.time()
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    elapsed = time.time() - start

    data = resp.json()
    text = data["content"][0]["text"]
    usage = data.get("usage", {})
    return text, elapsed, usage


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Generate Sonnet+constitution responses for the 36-prompt eval")
    parser.add_argument("--prompts", default=str(EVAL_PROMPTS_PATH))
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Output tag (controls filename)")
    parser.add_argument("--model", default=SONNET_MODEL)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--max-prompts", type=int, default=None, help="Limit for smoke tests")
    parser.add_argument("--resume", action="store_true", help="Skip prompts that already have checkpoints")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    constitution = CONSTITUTION_PATH.read_text()
    system_text = build_system_text(constitution)
    log.info(
        "System prompt: %d chars (~%d tok) | model=%s | temp=%s | max_tokens=%d",
        len(system_text), len(system_text) // 4, args.model, args.temperature, args.max_tokens,
    )

    prompts: list[dict] = []
    with open(args.prompts) as f:
        for line in f:
            prompts.append(json.loads(line))
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
    log.info("Loaded %d prompts from %s", len(prompts), args.prompts)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    completed: dict[str, dict] = {}
    if args.resume:
        for ckpt in sorted(CHECKPOINT_DIR.glob("*.json")):
            try:
                rec = json.loads(ckpt.read_text())
                completed[rec["id"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue
        if completed:
            log.info("Resume: %d/%d already completed", len(completed), len(prompts))

    remaining = [p for p in prompts if p["id"] not in completed]
    log.info("%d prompts to generate, %d already done", len(remaining), len(completed))

    total_in = total_out = cache_read = cache_write = 0

    for i, p in enumerate(remaining, 1):
        pid = p["id"]
        log.info("[%d/%d] %s (%s) — %s...", i, len(remaining), pid, p["category"], p["prompt"][:60])

        try:
            response_text, elapsed, usage = call_sonnet(
                system_text, p["prompt"], args.model, args.max_tokens, args.temperature,
            )
        except Exception as e:
            log.error("  FAILED: %s", e)
            time.sleep(3)
            continue

        ti = usage.get("input_tokens", 0)
        to = usage.get("output_tokens", 0)
        cw = usage.get("cache_creation_input_tokens", 0)
        cr = usage.get("cache_read_input_tokens", 0)
        total_in += ti
        total_out += to
        cache_write += cw
        cache_read += cr

        cache_status = "HIT" if cr > 0 else ("WRITE" if cw > 0 else "MISS")
        log.info("  %.1fs | cache=%s (w=%d r=%d) | out=%d tok", elapsed, cache_status, cw, cr, to)

        # Schema MUST match what judge_responses.py reads from
        # data/eval/responses/responses-{tag}.jsonl. Mirror the R2 file layout.
        rec = {
            "id": pid,
            "category": p["category"],
            "difficulty": p.get("difficulty", "medium"),
            "prompt": p["prompt"],
            "ground_truth_signal": p.get("ground_truth_signal", ""),
            "response": response_text,
            "generation_time": round(elapsed, 1),
            "model": f"{args.model}+madison-constitution",
            "prompt_tokens": ti + cr + cw,
            "completion_tokens": to,
        }
        completed[pid] = rec
        (CHECKPOINT_DIR / f"{pid}.json").write_text(json.dumps(rec, indent=2))

        time.sleep(0.3)

    # Assemble final JSONL in the canonical order of eval-prompts.jsonl.
    ordered = [completed[p["id"]] for p in prompts if p["id"] in completed]
    out_path = RESPONSES_DIR / f"responses-{args.tag}.jsonl"
    with open(out_path, "w") as f:
        for r in ordered:
            f.write(json.dumps(r) + "\n")

    cost = (
        total_in * 3.0 / 1_000_000
        + total_out * 15.0 / 1_000_000
        + cache_write * 3.75 / 1_000_000
        + cache_read * 0.30 / 1_000_000
    )

    print("\n" + "=" * 72)
    print(f"Phase 1 Generation — {args.model}")
    print("=" * 72)
    print(f"Completed: {len(ordered)}/{len(prompts)}")
    print(f"Tokens    in: {total_in:>7,}  out: {total_out:>7,}  "
          f"cache_w: {cache_write:>7,}  cache_r: {cache_read:>7,}")
    print(f"Est. cost: ${cost:.4f}")
    print(f"Output:    {out_path.relative_to(PROJECT_ROOT)}")
    print()
    print("Next:")
    print(f"  python scripts/data/judge_responses.py --tag {args.tag} \\")
    print(f"      --judge-model claude-4-sonnet-20250514")
    print("=" * 72)


if __name__ == "__main__":
    main()
