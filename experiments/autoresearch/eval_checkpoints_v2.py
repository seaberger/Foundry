"""Evaluate checkpoints by generating responses on Modal + judging locally.

Bypasses the deploy/serve/poll cycle entirely. Instead:
1. Sends eval prompts to a Modal function that loads vLLM + adapter
2. Modal function generates all responses in one shot, returns them
3. Judges responses locally using the Anthropic API

Usage (from Foundry repo root):
    python3 experiments/autoresearch/eval_checkpoints_v2.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure foundry package is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from foundry.press.evaluate import (
    judge_response,
    compute_weighted_overall,
    extract_json,
    CONSTITUTION_PATH,
    COMPONENT_WEIGHTS,
)

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
PROBE_PROMPTS = HERE / "eval" / "probe-prompts.jsonl"
OUTPUT_NAME = "madison-qwen3-dryrun-20260404-202618"
EVAL_CHECKPOINTS = [150, 200, 300]

# Baseline from prepare.py
BASELINE_CATEGORIES = {
    "character_consistency": 9.41,
    "anachronism_trap": 9.39,
    "position_discrimination": 9.25,
    "ground_truth": 8.85,
    "private_voice": 8.75,
    "verified_response": 8.53,
}
BASELINE_OVERALL = 8.97


def load_prompts() -> list[dict]:
    prompts = []
    with open(PROBE_PROMPTS) as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def generate_responses_on_modal(adapter_path: str, prompts: list[dict]) -> list[dict]:
    """Send prompts to Modal, generate responses using vLLM + adapter, return results."""
    import subprocess

    # Write prompts to temp file for upload
    prompts_file = Path("/tmp/eval_prompts.jsonl")
    with open(prompts_file, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    # Use modal run to generate all responses
    script_path = HERE / "backend" / "generate_responses.py"
    cmd = [
        "modal", "run", str(script_path),
        "--adapter-path", adapter_path,
        "--prompts-file", str(prompts_file),
    ]

    print(f"    Generating {len(prompts)} responses on Modal...")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30 * 60)

    if proc.returncode != 0:
        print(f"    Generation failed: {proc.stderr[:500]}")
        return []

    # Parse responses from stdout (JSON per line after "RESPONSES_START" marker)
    responses = []
    capture = False
    for line in proc.stdout.split("\n"):
        if line.strip() == "RESPONSES_START":
            capture = True
            continue
        if line.strip() == "RESPONSES_END":
            break
        if capture and line.strip():
            try:
                responses.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    return responses


def judge_locally(prompts: list[dict], responses: list[dict], constitution: str) -> list[dict]:
    """Judge each response using the Anthropic API."""
    results = []
    for i, (prompt, resp) in enumerate(zip(prompts, responses)):
        prompt_text = prompt["prompt"]
        response_text = resp.get("response", "")
        ground_truth = prompt.get("ground_truth_signal", "")
        category = prompt["category"]

        print(f"    [{i+1}/{len(prompts)}] Judging {prompt['id']}...")

        try:
            scores = judge_response(
                prompt_text, response_text, ground_truth, constitution,
            )
        except Exception as e:
            print(f"      Judge error: {e}")
            scores = {"overall_score": 0.0, "critical_failures": [str(e)]}

        overall = scores.get("overall_score", 0.0)

        # Recompute weighted overall if possible
        computed = compute_weighted_overall(scores)
        if computed is not None:
            overall = computed

        results.append({
            "id": prompt["id"],
            "category": category,
            "prompt": prompt_text,
            "response": response_text[:200],
            "overall_score": overall,
            "scores": scores,
        })

    return results


def summarize_results(results: list[dict]) -> dict:
    """Compute category averages and deltas vs baseline."""
    by_cat: dict[str, list[float]] = {}
    for r in results:
        cat = r["category"]
        by_cat.setdefault(cat, []).append(r["overall_score"])

    cat_means = {cat: sum(s) / len(s) for cat, s in by_cat.items()}
    all_scores = [r["overall_score"] for r in results]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0

    gt = cat_means.get("ground_truth", 0)
    gt_delta = gt - BASELINE_CATEGORIES["ground_truth"]

    guard_deltas = {}
    for cat in ["character_consistency", "anachronism_trap", "position_discrimination", "private_voice", "verified_response"]:
        if cat in cat_means:
            guard_deltas[cat] = cat_means[cat] - BASELINE_CATEGORIES[cat]

    crit_failures = sum(1 for r in results if r["scores"].get("critical_failures"))

    return {
        "overall": round(overall, 2),
        "ground_truth": round(gt, 2),
        "gt_delta": round(gt_delta, 2),
        "guard_deltas": {k: round(v, 2) for k, v in guard_deltas.items()},
        "max_regression": round(min(guard_deltas.values()) if guard_deltas else 0, 2),
        "critical_failures": crit_failures,
        "by_category": {k: round(v, 2) for k, v in cat_means.items()},
    }


def main() -> int:
    run_tag = time.strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"eval-v2-{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts()
    constitution = CONSTITUTION_PATH.read_text()

    print(f"Evaluating {len(EVAL_CHECKPOINTS)} checkpoints with {len(prompts)} prompts")
    print(f"Output: {run_dir}")

    all_summaries = []

    for step in EVAL_CHECKPOINTS:
        print(f"\n{'='*60}")
        print(f"Checkpoint {step}")
        print(f"{'='*60}")

        if step == 300:
            adapter_path = f"/adapters/experiments/{OUTPUT_NAME}"
        else:
            adapter_path = f"/adapters/experiments/{OUTPUT_NAME}/checkpoint-{step}"

        # Generate responses on Modal
        responses = generate_responses_on_modal(adapter_path, prompts)
        if not responses:
            print(f"  No responses generated — skipping")
            all_summaries.append({"step": step, "status": "generation_failed"})
            continue

        print(f"  Got {len(responses)} responses")

        # Save raw responses
        resp_file = run_dir / f"responses-ckpt{step}.json"
        resp_file.write_text(json.dumps(responses, indent=2))

        # Judge locally
        results = judge_locally(prompts, responses, constitution)

        # Save judged results
        results_file = run_dir / f"judged-ckpt{step}.json"
        results_file.write_text(json.dumps(results, indent=2))

        # Summarize
        summary = summarize_results(results)
        summary["step"] = step
        summary["status"] = "ok"
        all_summaries.append(summary)

        print(f"  Overall: {summary['overall']}, GT: {summary['ground_truth']} (Δ{summary['gt_delta']:+.2f})")

    # Print comparison table
    print(f"\n{'='*80}")
    print("CHECKPOINT COMPARISON")
    print(f"{'='*80}")
    print(f"{'Step':>6} | {'Overall':>8} | {'GT':>8} | {'GT Δ':>8} | {'Max Regr':>9} | {'Crit':>5} | Status")
    print(f"{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*5}-+-------")

    for s in all_summaries:
        if s.get("status") == "ok":
            print(
                f"{s['step']:>6} | {s['overall']:>8.2f} | {s['ground_truth']:>8.2f} | "
                f"{s['gt_delta']:>+8.2f} | {s['max_regression']:>+9.2f} | "
                f"{s['critical_failures']:>5} | {s['status']}"
            )
        else:
            print(f"{s['step']:>6} | {'—':>8} | {'—':>8} | {'—':>8} | {'—':>9} | {'—':>5} | {s.get('status', '?')}")

    print(f"\nBaseline: overall={BASELINE_OVERALL}, GT={BASELINE_CATEGORIES['ground_truth']}")

    # Save summary
    summary_file = run_dir / "comparison.json"
    summary_file.write_text(json.dumps(all_summaries, indent=2))
    print(f"Saved: {summary_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
