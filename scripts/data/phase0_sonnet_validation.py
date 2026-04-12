"""Phase 0: Ad hoc validation that Sonnet 4.6 stays in character as Madison.

Runs 5 hand-picked prompts across the highest-risk categories (ground_truth,
position_discrimination, anachronism_trap, character_consistency, private_voice)
through claude-sonnet-4-6 using the madison-5k constitution plus the Madison
system prompt from scripts/modal/serve_madison_qwen.py (with the Qwen-specific
/no_think token stripped).

Purpose: verify Sonnet doesn't break character, refuse, or drift to modern
register before committing to the full 36-prompt Phase 1 evaluation.

Spec: docs/sonnet-vs-finetune-comparison.md, Phase 0 section.

Usage:
    cd ~/Repositories/Foundry
    python scripts/data/phase0_sonnet_validation.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import httpx

log = logging.getLogger("foundry.phase0")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONSTITUTION_PATH = PROJECT_ROOT / "config" / "constitutions" / "madison-5k.md"
EVAL_PROMPTS_PATH = PROJECT_ROOT / "data" / "eval" / "eval-prompts.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "eval" / "phase0-sonnet-validation.json"

SONNET_MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0.0
MAX_TOKENS = 1500

# Madison voice prompt extracted verbatim from scripts/modal/serve_madison_qwen.py
# (lines 111–142) with the Qwen-specific `/no_think` token removed, per Phase 0
# spec. Do NOT re-engineer — this is a direct A/B of the same inputs to different
# models.
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

# Five prompts, one per high-risk category. IDs reference data/eval/eval-prompts.jsonl.
# Spec (docs/sonnet-vs-finetune-comparison.md, Phase 0) explicitly names four target
# categories: ground_truth, anachronism_trap, private_voice, character_consistency.
# Adding position_discrimination (pd-02) to also probe whether Sonnet softens
# Madison's disagreements with Jefferson — another plausible RLHF failure mode.
TARGET_IDS = ["gt-03", "pd-02", "at-01", "cc-02", "pv-04"]


def build_system_text(constitution: str) -> str:
    """Assemble system prompt: constitution wrapper + Madison voice prompt."""
    return (
        "<madison_constitution>\n"
        f"{constitution}\n"
        "</madison_constitution>\n\n"
        f"{MADISON_SYSTEM_PROMPT}"
    )


def call_sonnet(system_text: str, user_prompt: str) -> tuple[str, float, dict]:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": SONNET_MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
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
        timeout=120,
    )
    resp.raise_for_status()
    elapsed = time.time() - start

    data = resp.json()
    text = data["content"][0]["text"]
    usage = data.get("usage", {})
    return text, elapsed, usage


def load_selected_prompts() -> list[dict]:
    by_id: dict[str, dict] = {}
    with open(EVAL_PROMPTS_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec["id"] in TARGET_IDS:
                by_id[rec["id"]] = rec
    missing = [pid for pid in TARGET_IDS if pid not in by_id]
    if missing:
        raise RuntimeError(f"Prompts not found in eval-prompts.jsonl: {missing}")
    return [by_id[pid] for pid in TARGET_IDS]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    constitution = CONSTITUTION_PATH.read_text()
    system_text = build_system_text(constitution)
    log.info(
        "System prompt: %d chars (~%d tok) | model=%s | temp=%s",
        len(system_text), len(system_text) // 4, SONNET_MODEL, TEMPERATURE,
    )

    prompts = load_selected_prompts()
    log.info("Selected %d prompts: %s", len(prompts), ", ".join(p["id"] for p in prompts))

    results = []
    total_in = total_out = cache_read = cache_write = 0

    for i, p in enumerate(prompts, 1):
        pid = p["id"]
        category = p["category"]
        prompt_text = p["prompt"]
        log.info("[%d/%d] %s (%s)", i, len(prompts), pid, category)

        try:
            response_text, elapsed, usage = call_sonnet(system_text, prompt_text)
        except Exception as e:
            log.error("  FAILED: %s", e)
            continue

        cw = usage.get("cache_creation_input_tokens", 0)
        cr = usage.get("cache_read_input_tokens", 0)
        ti = usage.get("input_tokens", 0)
        to = usage.get("output_tokens", 0)
        total_in += ti
        total_out += to
        cache_write += cw
        cache_read += cr

        log.info(
            "  %.1fs | in=%d (cache w=%d r=%d) | out=%d",
            elapsed, ti, cw, cr, to,
        )

        results.append({
            "id": pid,
            "category": category,
            "difficulty": p.get("difficulty", ""),
            "prompt": prompt_text,
            "ground_truth_signal": p.get("ground_truth_signal", ""),
            "response": response_text,
            "model": SONNET_MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "generation_time": round(elapsed, 1),
            "usage": {
                "input_tokens": ti,
                "output_tokens": to,
                "cache_creation_input_tokens": cw,
                "cache_read_input_tokens": cr,
            },
        })

        time.sleep(0.3)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    # Cost (Sonnet 4.6 pricing: in $3/M, out $15/M, cache write $3.75/M, cache read $0.30/M)
    cost = (
        total_in * 3.0 / 1_000_000
        + total_out * 15.0 / 1_000_000
        + cache_write * 3.75 / 1_000_000
        + cache_read * 0.30 / 1_000_000
    )

    print("\n" + "=" * 78)
    print(f"PHASE 0 RESULTS — {len(results)}/{len(prompts)} prompts | {SONNET_MODEL}")
    print("=" * 78)
    for r in results:
        print()
        print(f"--- [{r['id']}] {r['category']} ---")
        print(f"PROMPT: {r['prompt']}")
        print()
        print("RESPONSE:")
        print(r["response"])
        print()
        print(f"(generated in {r['generation_time']}s, {r['usage']['output_tokens']} out tokens)")
        print("-" * 78)

    print()
    print(f"Tokens — in: {total_in} | out: {total_out} | "
          f"cache w: {cache_write} | cache r: {cache_read}")
    print(f"Estimated cost: ${cost:.4f}")
    print(f"Saved: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
