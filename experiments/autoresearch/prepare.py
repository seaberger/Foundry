from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]  # Foundry repo root
AUTORESEARCH_ROOT = Path(__file__).resolve().parent

FULL_EVAL_PROMPTS = REPO_ROOT / "data" / "eval" / "eval-prompts.jsonl"
PROBE_EVAL_PROMPTS = AUTORESEARCH_ROOT / "eval" / "probe-prompts.jsonl"
DEFAULT_OUTPUT_DIR = Path(os.environ.get("FOUNDRY_AUTORESEARCH_OUTPUT_DIR", AUTORESEARCH_ROOT / "runs"))

EVAL_ENDPOINT = os.environ.get("FOUNDRY_AUTORESEARCH_ENDPOINT", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("FOUNDRY_AUTORESEARCH_MODEL_NAME", "madison-qwen3-probe")
JUDGE_MODEL = os.environ.get("FOUNDRY_AUTORESEARCH_JUDGE_MODEL", "claude-sonnet-4-6-20250514")

BASELINE_OVERALL = 8.97
BASELINE_CRITICAL_FAILURES = 0
BASELINE_CATEGORIES: dict[str, float] = {
    "character_consistency": 9.41,
    "anachronism_trap": 9.39,
    "position_discrimination": 9.25,
    "ground_truth": 8.85,
    "private_voice": 8.75,
    "verified_response": 8.53,
}
PRIMARY_CATEGORY = "ground_truth"
GUARD_CATEGORIES = [
    "character_consistency",
    "anachronism_trap",
    "position_discrimination",
    "private_voice",
    "verified_response",
]

PROBE_GUARD_TOLERANCE = 0.15
FULL_GUARD_TOLERANCE = 0.05
PROBE_PROMOTION_GT_DELTA = 0.15
FULL_SUCCESS_GT_DELTA = 0.10

GT_GAIN_WEIGHT = 8.0
SMALL_REGRESSION_PENALTY = 5.0
LARGE_REGRESSION_PENALTY = 9.0
OVERALL_REGRESSION_PENALTY = 4.0
CRITICAL_FAILURE_PENALTY = 2.0


@dataclass
class ConstrainedEvalResult:
    report_path: Path
    report: dict[str, Any]
    run_mode: str
    overall_mean: float
    ground_truth: float
    ground_truth_delta: float
    guard_deltas: dict[str, float]
    max_regression: float
    critical_failures: int
    constraint_ok: bool
    probe_score: float


def run_command(
    command: str | list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    if isinstance(command, str):
        args = shlex.split(command)
    else:
        args = command

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    proc = subprocess.run(
        args,
        cwd=cwd or REPO_ROOT,
        env=merged_env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"$ {' '.join(args)}\n\nSTDOUT\n{'-' * 80}\n{proc.stdout}\n\nSTDERR\n{'-' * 80}\n{proc.stderr}"
        )

    return proc


def latest_eval_report(output_dir: Path, tag: str) -> Path:
    matches = sorted(output_dir.glob(f"eval-{tag}-*.json"))
    if not matches:
        raise FileNotFoundError(f"No evaluation report found for tag={tag!r} in {output_dir}")
    return matches[-1]


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _summary_fields(report: dict[str, Any]) -> tuple[dict[str, float], float, int]:
    summary = report.get("summary", {})
    by_category = {k: float(v) for k, v in summary.get("by_category", {}).items()}
    overall = float(summary.get("overall_mean", 0.0))
    critical_failures = int(summary.get("critical_failure_count", 0))
    return by_category, overall, critical_failures


def compute_constraint_ok(report: dict[str, Any], *, run_mode: str) -> bool:
    by_category, overall, critical_failures = _summary_fields(report)
    tolerance = FULL_GUARD_TOLERANCE if run_mode == "full" else PROBE_GUARD_TOLERANCE
    required_gt_delta = FULL_SUCCESS_GT_DELTA if run_mode == "full" else PROBE_PROMOTION_GT_DELTA

    gt_score = float(by_category.get(PRIMARY_CATEGORY, 0.0))
    gt_delta = gt_score - BASELINE_CATEGORIES[PRIMARY_CATEGORY]
    if gt_delta < required_gt_delta:
        return False
    if overall < BASELINE_OVERALL:
        return False
    if critical_failures > BASELINE_CRITICAL_FAILURES:
        return False

    for category in GUARD_CATEGORIES:
        delta = float(by_category.get(category, 0.0)) - BASELINE_CATEGORIES[category]
        if delta < -tolerance:
            return False
    return True


def compute_probe_score(report: dict[str, Any], *, run_mode: str) -> float:
    by_category, overall, critical_failures = _summary_fields(report)
    gt_score = float(by_category.get(PRIMARY_CATEGORY, 0.0))
    gt_delta = gt_score - BASELINE_CATEGORIES[PRIMARY_CATEGORY]

    score = 100.0 + GT_GAIN_WEIGHT * gt_delta

    for category in GUARD_CATEGORIES:
        delta = float(by_category.get(category, 0.0)) - BASELINE_CATEGORIES[category]
        if delta < 0:
            score -= abs(delta) * SMALL_REGRESSION_PENALTY
        if delta < -(FULL_GUARD_TOLERANCE if run_mode == "full" else PROBE_GUARD_TOLERANCE):
            score -= abs(delta) * LARGE_REGRESSION_PENALTY

    if overall < BASELINE_OVERALL:
        score -= (BASELINE_OVERALL - overall) * OVERALL_REGRESSION_PENALTY

    if critical_failures > BASELINE_CRITICAL_FAILURES:
        score -= (critical_failures - BASELINE_CRITICAL_FAILURES) * CRITICAL_FAILURE_PENALTY

    return round(score, 4)


def summarize_report(report: dict[str, Any], *, run_mode: str) -> ConstrainedEvalResult:
    by_category, overall, critical_failures = _summary_fields(report)
    gt = float(by_category.get(PRIMARY_CATEGORY, 0.0))
    gt_delta = round(gt - BASELINE_CATEGORIES[PRIMARY_CATEGORY], 4)
    guard_deltas = {
        category: round(float(by_category.get(category, 0.0)) - BASELINE_CATEGORIES[category], 4)
        for category in GUARD_CATEGORIES
    }
    max_regression = round(min(guard_deltas.values()) if guard_deltas else 0.0, 4)
    return ConstrainedEvalResult(
        report_path=Path(""),
        report=report,
        run_mode=run_mode,
        overall_mean=round(overall, 4),
        ground_truth=round(gt, 4),
        ground_truth_delta=gt_delta,
        guard_deltas=guard_deltas,
        max_regression=max_regression,
        critical_failures=critical_failures,
        constraint_ok=compute_constraint_ok(report, run_mode=run_mode),
        probe_score=compute_probe_score(report, run_mode=run_mode),
    )


def evaluate_model(
    *,
    tag: str,
    run_mode: str = "probe",
    eval_prompts_path: Path | None = None,
    output_dir: Path | None = None,
    timeout: int = 60 * 60,
) -> ConstrainedEvalResult:
    eval_prompts = eval_prompts_path or PROBE_EVAL_PROMPTS
    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "foundry.press.evaluate",
        "--endpoint",
        EVAL_ENDPOINT,
        "--model",
        MODEL_NAME,
        "--tag",
        tag,
        "--eval-prompts",
        str(eval_prompts),
        "--judge-model",
        JUDGE_MODEL,
        "--output-dir",
        str(out_dir),
    ]
    proc = run_command(cmd, cwd=REPO_ROOT, timeout=timeout, log_path=out_dir / f"eval-{tag}.log")
    if proc.returncode != 0:
        raise RuntimeError(f"Evaluation command failed. See {out_dir / f'eval-{tag}.log'}")

    report_path = latest_eval_report(out_dir, tag)
    report = load_report(report_path)
    result = summarize_report(report, run_mode=run_mode)
    result.report_path = report_path
    return result


def print_result(result: ConstrainedEvalResult) -> None:
    print("---")
    print(f"run_mode:            {result.run_mode}")
    print(f"overall_mean:        {result.overall_mean:.4f}")
    print(f"ground_truth:        {result.ground_truth:.4f}")
    print(f"ground_truth_delta:  {result.ground_truth_delta:+.4f}")
    print(f"max_regression:      {result.max_regression:+.4f}")
    print(f"critical_failures:   {result.critical_failures}")
    print(f"constraint_ok:       {str(result.constraint_ok).lower()}")
    print(f"probe_score:         {result.probe_score:.4f}")
    print(f"report_path:         {result.report_path}")
    print("guard_deltas:")
    for key, value in result.guard_deltas.items():
        print(f"  {key}: {value:+.4f}")


def main() -> None:
    tag = f"prepare-smoke-{time.strftime('%Y%m%d-%H%M%S')}"
    target = PROBE_EVAL_PROMPTS if PROBE_EVAL_PROMPTS.exists() else FULL_EVAL_PROMPTS
    result = evaluate_model(tag=tag, run_mode="probe", eval_prompts_path=target)
    print_result(result)


if __name__ == "__main__":
    main()
