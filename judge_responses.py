"""Judge pre-generated eval responses using Sonnet 4.6 with prompt caching.

Reads responses from the JSONL file, sends each through a cached Sonnet judge,
and produces a full eval report. Uses Anthropic prompt caching to reduce costs
~50% by caching the system prompt (constitution + rubric) across all 36 calls.

This script does NOT modify evaluate.py — it reimplements the judge call with
caching support and imports only constants from evaluate.py.

Usage:
    cd ~/Repositories/Foundry
    python judge_responses.py --tag orpo-v3b
    python judge_responses.py --tag orpo-v3b --responses data/eval/responses/responses-orpo-v3b.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from src.foundry.press.evaluate import (
    CONSTITUTION_PATH,
    EVAL_OUTPUT_DIR,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT,
)

log = logging.getLogger("foundry.judge")


def judge_response_cached(
    prompt: str,
    response_text: str,
    ground_truth_signal: str,
    constitution: str,
    judge_model: str = "claude-4-sonnet-20250514",
) -> dict:
    """Score a response using the LLM judge with prompt caching.

    The system prompt (rubric + constitution) is marked with cache_control
    so it's cached across sequential calls. First call pays a 25% write premium,
    subsequent calls get a 90% discount on the cached portion.
    """
    import httpx

    # Use replace() not format() — JUDGE_SYSTEM_PROMPT has literal JSON braces
    # that .format() tries to interpolate as placeholders
    system_text = JUDGE_SYSTEM_PROMPT.replace("{constitution}", constitution)
    user_text = (JUDGE_USER_PROMPT
        .replace("{prompt}", prompt)
        .replace("{ground_truth_signal}", ground_truth_signal)
        .replace("{response}", response_text)
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # Use content block array for system prompt with cache_control
    payload = {
        "model": judge_model,
        "max_tokens": 2048,
        "system": [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": user_text}],
        "temperature": 0.0,
    }

    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    # Log cache usage
    usage = data.get("usage", {})
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_creation = usage.get("cache_creation_input_tokens", 0)
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    if cache_read > 0:
        log.info("    Cache HIT: %d cached, %d new input, %d output", cache_read, input_tokens, output_tokens)
    elif cache_creation > 0:
        log.info("    Cache WRITE: %d written, %d input, %d output", cache_creation, input_tokens, output_tokens)
    else:
        log.info("    No cache: %d input, %d output", input_tokens, output_tokens)

    # Extract JSON from response
    text = data["content"][0]["text"]
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        log.error("Failed to parse judge response: %s", text[:200])
        return {
            "voice_authenticity": {"score": 0, "justification": "Parse error"},
            "rhetorical_pattern": {"score": 0, "justification": "Parse error"},
            "historical_accuracy": {"score": 0, "justification": "Parse error"},
            "position_fidelity": {"score": 0, "justification": "Parse error"},
            "character_integrity": {"score": 0, "justification": "Parse error"},
            "overall_score": 0.0,
            "critical_failures": ["Failed to parse judge JSON"],
            "strongest_element": "N/A",
            "weakest_element": "N/A",
        }


def main():
    parser = argparse.ArgumentParser(description="Judge pre-generated eval responses with prompt caching")
    parser.add_argument("--tag", required=True, help="Eval tag")
    parser.add_argument("--responses", default=None,
                        help="Path to responses JSONL (default: data/eval/responses/responses-{tag}.jsonl)")
    parser.add_argument("--judge-model", default="claude-4-sonnet-20250514")
    parser.add_argument("--output-dir", default=str(EVAL_OUTPUT_DIR))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    # Load responses
    responses_path = Path(args.responses) if args.responses else Path(f"data/eval/responses/responses-{args.tag}.jsonl")
    if not responses_path.exists():
        log.error("Responses file not found: %s", responses_path)
        return

    responses = []
    with open(responses_path) as f:
        for line in f:
            responses.append(json.loads(line))
    log.info("Loaded %d responses from %s", len(responses), responses_path)

    # Load constitution
    constitution = CONSTITUTION_PATH.read_text()
    log.info("Loaded constitution (%d chars) — will be cached across all judge calls", len(constitution))

    # Judge each response
    results = []
    category_scores: dict[str, list[float]] = {}
    total_cache_read = 0
    total_cache_write = 0

    for i, r in enumerate(responses):
        prompt_id = r["id"]
        category = r["category"]
        prompt_text = r["prompt"]
        response_text = r["response"]
        ground_truth = r.get("ground_truth_signal", "")
        difficulty = r.get("difficulty", "medium")

        log.info("[%d/%d] Judging %s (%s, %s)...", i + 1, len(responses), prompt_id, category, difficulty)

        try:
            scores = judge_response_cached(
                prompt_text, response_text, ground_truth, constitution,
                judge_model=args.judge_model,
            )
        except Exception as e:
            log.error("  Judging failed for %s: %s", prompt_id, e)
            scores = {
                "voice_authenticity": {"score": 0, "justification": f"Judge error: {e}"},
                "rhetorical_pattern": {"score": 0, "justification": f"Judge error: {e}"},
                "historical_accuracy": {"score": 0, "justification": f"Judge error: {e}"},
                "position_fidelity": {"score": 0, "justification": f"Judge error: {e}"},
                "character_integrity": {"score": 0, "justification": f"Judge error: {e}"},
                "overall_score": 0.0,
                "critical_failures": [f"Judge error: {e}"],
                "strongest_element": "N/A",
                "weakest_element": "N/A",
            }

        overall = scores.get("overall_score")
        if overall is None:
            # Judge omitted overall_score — compute from component averages
            components = ["voice_authenticity", "rhetorical_pattern", "historical_accuracy",
                          "position_fidelity", "character_integrity"]
            comp_scores = [scores[c]["score"] for c in components
                           if c in scores and isinstance(scores[c], dict) and "score" in scores[c]]
            if comp_scores:
                overall = round(sum(comp_scores) / len(comp_scores), 1)
                log.warning("  Missing overall_score for %s — computed %.1f from %d components",
                            prompt_id, overall, len(comp_scores))
            else:
                overall = 0.0
                log.error("  Missing overall_score AND no valid components for %s", prompt_id)

        result = {
            "id": prompt_id,
            "category": category,
            "difficulty": difficulty,
            "prompt": prompt_text,
            "response": response_text,
            "generation_time": r.get("generation_time", 0),
            "scores": scores,
            "overall_score": overall,
            "model": r.get("model", "unknown"),
            "tag": args.tag,
        }
        results.append(result)

        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(overall)

        log.info("  Score: %.1f | %s", overall, prompt_id)

    # Compute summary
    all_scores = [r["overall_score"] for r in results]
    summary = {
        "tag": args.tag,
        "model": results[0]["model"] if results else "unknown",
        "judge_model": args.judge_model,
        "num_prompts": len(responses),
        "num_judged": len(results),
        "overall_mean": round(sum(all_scores) / len(all_scores), 2) if all_scores else 0,
        "overall_min": round(min(all_scores), 2) if all_scores else 0,
        "overall_max": round(max(all_scores), 2) if all_scores else 0,
        "by_category": {
            cat: round(sum(s) / len(s), 2)
            for cat, s in category_scores.items()
        },
        "by_difficulty": {},
        "critical_failure_count": sum(
            1 for r in results if r["scores"].get("critical_failures")
        ),
    }

    for diff in ["easy", "medium", "hard"]:
        diff_scores = [r["overall_score"] for r in results if r["difficulty"] == diff]
        if diff_scores:
            summary["by_difficulty"][diff] = round(sum(diff_scores) / len(diff_scores), 2)

    report = {"summary": summary, "results": results}

    # Save report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"eval-{args.tag}-judged-{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    s = summary
    print(f"\n{'='*60}")
    print(f"Madison Eval — {s['tag']} (Judge-Only, Cached)")
    print(f"{'='*60}")
    print(f"Model:    {s['model']}")
    print(f"Judge:    {s['judge_model']}")
    print(f"Judged:   {s['num_judged']}/{s['num_prompts']}")
    print()
    print(f"Overall:  {s['overall_mean']:.1f} (min {s['overall_min']:.1f}, max {s['overall_max']:.1f})")
    print()
    print("By category:")
    for cat, score in sorted(s["by_category"].items()):
        print(f"  {cat:30s} {score:.1f}")
    print()
    print("By difficulty:")
    for diff, score in sorted(s["by_difficulty"].items()):
        print(f"  {diff:30s} {score:.1f}")
    print()
    print(f"Critical failures: {s['critical_failure_count']}")
    print()
    print(f"Report saved: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
