"""Autoresearch dry run — train once, evaluate at multiple checkpoints.

Trains R2-baseline config for 300 steps with checkpoints at 150 and 300,
then evaluates each checkpoint with the 14-prompt probe set to determine
the minimum viable step count for autoresearch probes.

Usage (from Foundry repo root):
    python experiments/autoresearch/dry_run.py

Estimated time: ~80 min (35 min training + 3 × 15 min eval)
Estimated cost: ~$7 Modal A100 + ~$1.20 Anthropic judge API
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add parent to path so we can import prepare
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare

ROOT = Path(__file__).resolve().parents[2]  # Foundry repo root
HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"

# Checkpoints to evaluate (in training step order)
EVAL_CHECKPOINTS = [150, 200, 300]

# Training config — matches R2 production baseline
TRAIN_CONFIG = {
    "beta": 0.1,
    "lora_rank": 64,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "max_steps": 300,
    "warmup_ratio": 0.10,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 2048,
}

# Manifest for dataset materialization
MANIFEST_PATH = HERE / "manifests" / "baseline_r2.json"


@dataclass
class CheckpointResult:
    step: int
    adapter_path: str
    result: prepare.ConstrainedEvalResult | None
    status: str
    error: str


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


def materialize_dataset(run_dir: Path) -> tuple[Path, int]:
    """Materialize the baseline dataset using the same logic as train.py."""
    from train import load_manifest, load_jsonl, RecordWrapper, stable_id, extract_prompt

    manifest = load_manifest(MANIFEST_PATH)
    base_path = ROOT / manifest["base_dataset"]
    records = load_jsonl(base_path)

    wrapped = [
        RecordWrapper(record_id=stable_id(r, extract_prompt(r)), prompt=extract_prompt(r), raw=r)
        for r in records
    ]

    # No rules for baseline_r2 manifest, just shuffle
    import random
    rng = random.Random(int(manifest.get("shuffle_seed", 42)))
    shuffled = list(wrapped)
    rng.shuffle(shuffled)

    dataset_path = run_dir / "train.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w") as f:
        for w in shuffled:
            f.write(json.dumps(w.raw, ensure_ascii=False) + "\n")

    return dataset_path, len(shuffled)


def train(run_tag: str, run_dir: Path, dataset_path: Path) -> bool:
    """Run one training job for 300 steps with checkpoints at 150-step intervals."""
    output_name = f"madison-qwen3-dryrun-{run_tag}"

    cmd = (
        f"modal run experiments/autoresearch/backend/modal_train_orpo_qwen_autoresearch.py "
        f"--beta {TRAIN_CONFIG['beta']} "
        f"--rank {TRAIN_CONFIG['lora_rank']} "
        f"--alpha {TRAIN_CONFIG['lora_alpha']} "
        f"--dropout {TRAIN_CONFIG['lora_dropout']} "
        f"--lr {TRAIN_CONFIG['learning_rate']} "
        f"--epochs {TRAIN_CONFIG['num_epochs']} "
        f"--max-steps {TRAIN_CONFIG['max_steps']} "
        f"--warmup-ratio {TRAIN_CONFIG['warmup_ratio']} "
        f"--batch-size {TRAIN_CONFIG['per_device_batch_size']} "
        f"--grad-accum {TRAIN_CONFIG['gradient_accumulation_steps']} "
        f"--max-seq-length {TRAIN_CONFIG['max_seq_length']} "
        f"--output-name {output_name} "
        f"--save-steps 50 "
        f"--save-total-limit 6 "
        f"--dataset {dataset_path}"
    )

    print(f"\n{'='*60}")
    print(f"Training: {output_name}")
    print(f"Max steps: {TRAIN_CONFIG['max_steps']}")
    print(f"Checkpoints every 50 steps (keeping all 6)")
    print(f"{'='*60}")

    proc = run_command(cmd, cwd=ROOT, timeout=6 * 60 * 60, log_path=run_dir / "train.log")
    if proc.returncode != 0:
        print(f"Training failed! See {run_dir / 'train.log'}")
        return False

    print("Training complete.")
    return True


def activate_candidate(adapter_name: str, adapter_path: str, run_dir: Path, step: int) -> str | None:
    """Deploy candidate endpoint, return URL or None on failure."""
    cmd = (
        f"python experiments/autoresearch/backend/activate_candidate.py "
        f"--adapter-name {adapter_name} "
        f"--adapter-path {adapter_path}"
    )
    print(f"  Activating checkpoint-{step}...")
    proc = run_command(cmd, cwd=ROOT, timeout=30 * 60, log_path=run_dir / f"activate-{step}.log")
    if proc.returncode != 0:
        print(f"  Activation failed! See {run_dir / f'activate-{step}.log'}")
        return None

    url = proc.stdout.strip().split("\n")[-1].strip()
    if url.startswith("http"):
        return url
    print(f"  Activation did not return URL. Got: {url!r}")
    return None


def deactivate_candidate():
    """Stop the ephemeral candidate endpoint."""
    try:
        subprocess.run(
            shlex.split("modal app stop foundry-autoresearch-candidate"),
            cwd=ROOT, capture_output=True, check=False, timeout=60,
        )
    except Exception:
        pass


def evaluate_checkpoint(
    run_tag: str,
    run_dir: Path,
    output_name: str,
    step: int,
) -> CheckpointResult:
    """Activate, evaluate, and deactivate a single checkpoint."""
    # Checkpoint path on Modal volume
    if step == TRAIN_CONFIG["max_steps"]:
        # Final model is saved at the output_name root, not in a checkpoint dir
        adapter_path = f"/adapters/experiments/{output_name}"
    else:
        adapter_path = f"/adapters/experiments/{output_name}/checkpoint-{step}"

    adapter_name = f"{output_name}-ckpt{step}"

    # Activate
    url = activate_candidate(adapter_name, adapter_path, run_dir, step)
    if not url:
        return CheckpointResult(step=step, adapter_path=adapter_path, result=None, status="activation_failed", error="Could not deploy endpoint")

    # Set eval endpoint
    prepare.EVAL_ENDPOINT = url
    print(f"  Evaluating at {url}...")

    try:
        eval_tag = f"dryrun-{run_tag}-step{step}"
        result = prepare.evaluate_model(
            tag=eval_tag,
            run_mode="probe",
            output_dir=run_dir,
        )
        prepare.print_result(result)
        return CheckpointResult(step=step, adapter_path=adapter_path, result=result, status="ok", error="")
    except Exception as exc:
        print(f"  Evaluation failed: {exc}")
        return CheckpointResult(step=step, adapter_path=adapter_path, result=None, status="eval_failed", error=str(exc))
    finally:
        deactivate_candidate()


def print_comparison(results: list[CheckpointResult]):
    """Print comparison table of all checkpoint evaluations."""
    print(f"\n{'='*80}")
    print("DRY RUN RESULTS — Step Budget Comparison")
    print(f"{'='*80}")

    # Header
    print(f"\n{'Step':>6} | {'Overall':>8} | {'GT':>8} | {'GT Δ':>8} | {'Max Regr':>9} | {'Probe Score':>12} | {'Constraint':>10} | {'Status'}")
    print(f"{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}")

    for cr in results:
        if cr.result:
            r = cr.result
            print(
                f"{cr.step:>6} | {r.overall_mean:>8.2f} | {r.ground_truth:>8.2f} | "
                f"{r.ground_truth_delta:>+8.2f} | {r.max_regression:>+9.2f} | "
                f"{r.probe_score:>12.2f} | {'PASS' if r.constraint_ok else 'FAIL':>10} | {cr.status}"
            )
        else:
            print(f"{cr.step:>6} | {'—':>8} | {'—':>8} | {'—':>8} | {'—':>9} | {'—':>12} | {'—':>10} | {cr.status}: {cr.error[:40]}")

    # Guard category breakdown
    print(f"\nGuard Category Deltas:")
    print(f"{'Step':>6} |", end="")
    guard_cats = ["character_consistency", "anachronism_trap", "position_discrimination", "private_voice", "verified_response"]
    for cat in guard_cats:
        short = cat[:6]
        print(f" {short:>8} |", end="")
    print()

    for cr in results:
        if cr.result:
            print(f"{cr.step:>6} |", end="")
            for cat in guard_cats:
                delta = cr.result.guard_deltas.get(cat, 0)
                print(f" {delta:>+8.2f} |", end="")
            print()

    print(f"\nBaseline: overall={prepare.BASELINE_OVERALL}, GT={prepare.BASELINE_CATEGORIES['ground_truth']}")
    print(f"{'='*80}")


def main() -> int:
    run_tag = now_tag()
    run_dir = RUNS_DIR / f"dryrun-{run_tag}"
    output_name = f"madison-qwen3-dryrun-{run_tag}"

    print(f"Autoresearch Dry Run — {run_tag}")
    print(f"Checkpoints to evaluate: {EVAL_CHECKPOINTS}")
    print(f"Output: {run_dir}")

    # Step 1: Materialize dataset
    print("\n--- Materializing dataset ---")
    dataset_path, num_pairs = materialize_dataset(run_dir)
    print(f"Dataset: {num_pairs} pairs at {dataset_path}")

    # Step 2: Train
    print("\n--- Training ---")
    ok = train(run_tag, run_dir, dataset_path)
    if not ok:
        return 1

    # Step 3: Evaluate each checkpoint
    results: list[CheckpointResult] = []
    for step in EVAL_CHECKPOINTS:
        print(f"\n--- Evaluating checkpoint {step} ---")
        cr = evaluate_checkpoint(run_tag, run_dir, output_name, step)
        results.append(cr)

    # Step 4: Print comparison
    print_comparison(results)

    # Save results to JSON
    results_json = []
    for cr in results:
        entry = {"step": cr.step, "adapter_path": cr.adapter_path, "status": cr.status, "error": cr.error}
        if cr.result:
            entry.update({
                "overall_mean": cr.result.overall_mean,
                "ground_truth": cr.result.ground_truth,
                "ground_truth_delta": cr.result.ground_truth_delta,
                "guard_deltas": cr.result.guard_deltas,
                "max_regression": cr.result.max_regression,
                "constraint_ok": cr.result.constraint_ok,
                "probe_score": cr.result.probe_score,
            })
        results_json.append(entry)

    results_path = run_dir / "dryrun_results.json"
    results_path.write_text(json.dumps(results_json, indent=2))
    print(f"\nResults saved: {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
