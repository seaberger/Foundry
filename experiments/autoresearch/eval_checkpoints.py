"""Evaluate existing checkpoints from a dry run (skips training).

Usage (from Foundry repo root):
    python3 experiments/autoresearch/eval_checkpoints.py
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"

# The dry run output name — adapter already trained on Modal volume
OUTPUT_NAME = "madison-qwen3-dryrun-20260404-202618"
EVAL_CHECKPOINTS = [150, 200, 300]


def now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def run_command(command: str, *, cwd: Path, timeout: int, log_path: Path) -> subprocess.CompletedProcess[str]:
    args = shlex.split(command)
    proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {command}\n\nSTDOUT\n{'-' * 80}\n{proc.stdout}\n\nSTDERR\n{'-' * 80}\n{proc.stderr}"
    )
    return proc


def activate_candidate(adapter_name: str, adapter_path: str, log_path: Path) -> str | None:
    cmd = (
        f"python3 experiments/autoresearch/backend/activate_candidate.py "
        f"--adapter-name {adapter_name} "
        f"--adapter-path {adapter_path}"
    )
    print(f"  Activating {adapter_name}...")
    proc = run_command(cmd, cwd=ROOT, timeout=30 * 60, log_path=log_path)
    if proc.returncode != 0:
        print(f"  Activation failed! See {log_path}")
        print(f"  stderr: {proc.stderr[:500]}")
        return None

    url = proc.stdout.strip().split("\n")[-1].strip()
    if url.startswith("http"):
        return url
    print(f"  Activation did not return URL. Got: {url!r}")
    return None


def deactivate_candidate():
    try:
        subprocess.run(
            shlex.split("modal app stop foundry-autoresearch-candidate"),
            cwd=ROOT, capture_output=True, check=False, timeout=60,
        )
    except Exception:
        pass


def main() -> int:
    run_tag = now_tag()
    run_dir = RUNS_DIR / f"eval-ckpts-{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating checkpoints from {OUTPUT_NAME}")
    print(f"Checkpoints: {EVAL_CHECKPOINTS}")
    print(f"Output: {run_dir}")

    results = []

    for step in EVAL_CHECKPOINTS:
        print(f"\n{'='*60}")
        print(f"Checkpoint {step}")
        print(f"{'='*60}")

        if step == 300:
            adapter_path = f"/adapters/experiments/{OUTPUT_NAME}"
        else:
            adapter_path = f"/adapters/experiments/{OUTPUT_NAME}/checkpoint-{step}"

        adapter_name = f"{OUTPUT_NAME}-ckpt{step}"

        # Activate
        log_path = run_dir / f"activate-{step}.log"
        url = activate_candidate(adapter_name, adapter_path, log_path)
        if not url:
            results.append({"step": step, "status": "activation_failed"})
            continue

        prepare.EVAL_ENDPOINT = url
        print(f"  Endpoint: {url}")

        # Evaluate
        try:
            eval_tag = f"dryrun-ckpt{step}-{run_tag}"
            result = prepare.evaluate_model(tag=eval_tag, run_mode="probe", output_dir=run_dir)
            prepare.print_result(result)

            results.append({
                "step": step,
                "status": "ok",
                "overall_mean": result.overall_mean,
                "ground_truth": result.ground_truth,
                "ground_truth_delta": result.ground_truth_delta,
                "guard_deltas": result.guard_deltas,
                "max_regression": result.max_regression,
                "constraint_ok": result.constraint_ok,
                "probe_score": result.probe_score,
            })
        except Exception as exc:
            print(f"  Eval failed: {exc}")
            results.append({"step": step, "status": f"eval_failed: {exc}"})
        finally:
            print("  Deactivating...")
            deactivate_candidate()

    # Print comparison table
    print(f"\n{'='*80}")
    print("CHECKPOINT COMPARISON")
    print(f"{'='*80}")
    print(f"{'Step':>6} | {'Overall':>8} | {'GT':>8} | {'GT Δ':>8} | {'Max Regr':>9} | {'Probe':>8} | Status")
    print(f"{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-------")

    for r in results:
        if r["status"] == "ok":
            print(
                f"{r['step']:>6} | {r['overall_mean']:>8.2f} | {r['ground_truth']:>8.2f} | "
                f"{r['ground_truth_delta']:>+8.2f} | {r['max_regression']:>+9.2f} | "
                f"{r['probe_score']:>8.2f} | {r['status']}"
            )
        else:
            print(f"{r['step']:>6} | {'—':>8} | {'—':>8} | {'—':>8} | {'—':>9} | {'—':>8} | {r['status']}")

    print(f"\nBaseline: overall={prepare.BASELINE_OVERALL}, GT={prepare.BASELINE_CATEGORIES['ground_truth']}")

    # Save
    results_path = run_dir / "checkpoint_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
