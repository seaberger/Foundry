from __future__ import annotations

import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import prepare

# ============================================================================
# AGENT-EDITABLE SECTION
# ============================================================================
RUN_MODE = "probe"  # scout | probe | confirm | full | dataset-only
BASE_MODEL = "Qwen/Qwen3-32B"
OBJECTIVE = "orpo"
LORA_RANK = 64
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
LEARNING_RATE = 2e-5
BETA = 0.1
WARMUP_RATIO = 0.10
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
MAX_SEQ_LENGTH = 2048
CURRICULUM = "shuffle"  # shuffle | none | source_first | weak_gt_first | weak_gt_last
MANIFEST_PATH = Path(__file__).resolve().parent / "manifests" / "baseline_r2.json"
MAX_EXAMPLES_BY_MODE = {
    "scout": 700,
    "probe": 1100,
    "confirm": 1400,
    "full": 1000000,
    "dataset-only": 1000000,
}
STEP_BUDGETS = {
    "scout": 200,    # Binary gate: margin > 0.1 = alive, < 0.05 = dead
    "probe": 300,    # Full ranking: configs reliably ordered by reward margin
    "confirm": 450,  # High confidence: margin plateau confirms ranking
    "full": 1011,    # Publication-grade: matches R2 production run length
    "dataset-only": 0,
}
PROMOTE_IF_CONSTRAINT_OK = True
PROMOTE_IF_PROBE_SCORE_AT_LEAST = 100.75
PROMOTE_IF_GT_DELTA_AT_LEAST = 0.15

# Candidate activation: deploy ephemeral vLLM endpoint on Modal for evaluation.
# Prints endpoint URL (with /v1) to stdout. Stopped after eval by deactivate_candidate().
ACTIVATE_CANDIDATE_COMMAND_TEMPLATE = (
    "python3 experiments/autoresearch/backend/activate_candidate.py "
    "--adapter-name {output_name} "
    "--adapter-path {adapter_path}"
)
DEACTIVATE_COMMAND = "modal app stop foundry-autoresearch-candidate"

TRAIN_COMMAND_TEMPLATE = (
    "modal run experiments/autoresearch/backend/modal_train_orpo_qwen_autoresearch.py "
    "--beta {beta} "
    "--rank {lora_rank} "
    "--alpha {lora_alpha} "
    "--dropout {lora_dropout} "
    "--lr {learning_rate} "
    "--epochs {num_epochs} "
    "--max-steps {max_steps} "
    "--warmup-ratio {warmup_ratio} "
    "--batch-size {per_device_batch_size} "
    "--grad-accum {gradient_accumulation_steps} "
    "--max-seq-length {max_seq_length} "
    "--output-name {output_name} "
    "--dataset {dataset_path}"
)
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]  # Foundry repo root
HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
RESULTS_PATH = HERE / "results.tsv"


@dataclass
class RunContext:
    run_tag: str
    run_dir: Path
    dataset_path: Path
    manifest_path: Path
    output_name: str
    max_steps: int
    max_examples: int


@dataclass
class RecordWrapper:
    record_id: str
    prompt: str
    raw: dict[str, Any]


def now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def stable_id(record: dict[str, Any], prompt: str) -> str:
    for key in ("id", "pair_id", "uuid"):
        if key in record:
            return str(record[key])
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    return f"pair-{digest}"


def extract_prompt(record: dict[str, Any]) -> str:
    chosen = record.get("chosen", [])
    if not chosen:
        return ""
    first = chosen[0]
    return str(first.get("content", ""))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    manifest.setdefault("name", path.stem)
    manifest.setdefault("base_dataset", "data/training/madison-orpo-v6.jsonl")
    manifest.setdefault("global_repeat", 1)
    manifest.setdefault("shuffle_seed", 42)
    manifest.setdefault("rules", [])
    return manifest


def coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def record_value_candidates(record: dict[str, Any], key: str) -> list[str]:
    candidates: list[str] = []
    if key in record:
        candidates.extend(coerce_str_list(record[key]))
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        candidates.extend(coerce_str_list(metadata[key]))
    return candidates


def matches_rule(wrapper: RecordWrapper, rule: dict[str, Any]) -> bool:
    match = rule.get("match", {})
    if not match:
        return False

    checks: list[bool] = []

    id_any = set(coerce_str_list(match.get("id_any")))
    if id_any:
        checks.append(wrapper.record_id in id_any)

    prompt_lower = wrapper.prompt.lower()
    prompt_contains_any = [s.lower() for s in coerce_str_list(match.get("prompt_contains_any"))]
    if prompt_contains_any:
        checks.append(any(needle in prompt_lower for needle in prompt_contains_any))

    metadata_any = match.get("metadata_any", {})
    if isinstance(metadata_any, dict) and metadata_any:
        md_hit = False
        for key, allowed in metadata_any.items():
            allowed_set = {str(v).lower() for v in coerce_str_list(allowed)}
            values = [v.lower() for v in record_value_candidates(wrapper.raw, key)]
            if any(v in allowed_set for v in values):
                md_hit = True
                break
        checks.append(md_hit)

    source_any = {s.lower() for s in coerce_str_list(match.get("source_any"))}
    if source_any:
        values = [v.lower() for v in record_value_candidates(wrapper.raw, "source")]
        checks.append(any(v in source_any for v in values))

    match_mode = str(rule.get("match_mode", "any")).lower()
    if not checks:
        return False
    return all(checks) if match_mode == "all" else any(checks)


def label_record(wrapper: RecordWrapper) -> set[str]:
    prompt = wrapper.prompt.lower()
    labels: set[str] = set()
    if any(n in prompt for n in ("federalist", "constitution", "edward coles", "billey", "helvidius", "bank")):
        labels.add("source_dense")
    if any(n in prompt for n in ("quote", "evidence", "document", "source", "where did", "did madison")):
        labels.add("ground_truth")
    if any(n in prompt for n in ("voice", "letter", "private", "friend", "dearest", "sir")):
        labels.add("private_voice")
    return labels


def apply_curriculum(records: list[RecordWrapper], curriculum: str, seed: int) -> list[RecordWrapper]:
    ordered = list(records)
    if curriculum == "none":
        return ordered
    if curriculum == "shuffle":
        rng = random.Random(seed)
        rng.shuffle(ordered)
        return ordered

    decorated: list[tuple[int, int, RecordWrapper]] = []
    for idx, wrapper in enumerate(ordered):
        labels = label_record(wrapper)
        if curriculum == "source_first":
            priority = 0 if "source_dense" in labels else 1
        elif curriculum == "weak_gt_first":
            priority = 0 if "ground_truth" in labels else 1
        elif curriculum == "weak_gt_last":
            priority = 1 if "ground_truth" in labels else 0
        else:
            priority = 0
        decorated.append((priority, idx, wrapper))

    decorated.sort(key=lambda x: (x[0], x[1]))
    return [item[2] for item in decorated]


def materialize_dataset(ctx: RunContext) -> tuple[int, Path, dict[str, Any]]:
    manifest = load_manifest(ctx.manifest_path)
    base_dataset_path = ROOT / manifest["base_dataset"]
    base_records = load_jsonl(base_dataset_path)

    wrapped = [
        RecordWrapper(record_id=stable_id(r, extract_prompt(r)), prompt=extract_prompt(r), raw=r)
        for r in base_records
    ]

    expanded: list[RecordWrapper] = []
    default_repeat = int(manifest.get("global_repeat", 1))
    for wrapper in wrapped:
        expanded.extend([wrapper] * default_repeat)

    for rule in manifest.get("rules", []):
        repeat = int(rule.get("repeat", 0))
        if repeat <= 0:
            continue
        for wrapper in wrapped:
            if matches_rule(wrapper, rule):
                expanded.extend([wrapper] * repeat)

    expanded = apply_curriculum(expanded, curriculum=CURRICULUM, seed=int(manifest.get("shuffle_seed", 42)))

    if ctx.max_examples and len(expanded) > ctx.max_examples:
        expanded = expanded[: ctx.max_examples]

    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    with ctx.dataset_path.open("w") as f:
        for wrapper in expanded:
            f.write(json.dumps(wrapper.raw, ensure_ascii=False) + "\n")

    snapshot = {
        "manifest": manifest,
        "run_mode": RUN_MODE,
        "base_model": BASE_MODEL,
        "objective": OBJECTIVE,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "learning_rate": LEARNING_RATE,
        "beta": BETA,
        "warmup_ratio": WARMUP_RATIO,
        "num_epochs": NUM_EPOCHS,
        "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "curriculum": CURRICULUM,
        "max_examples": ctx.max_examples,
        "materialized_examples": len(expanded),
        "max_steps": ctx.max_steps,
        "dataset_path": str(ctx.dataset_path),
    }
    (ctx.run_dir / "config.json").write_text(json.dumps(snapshot, indent=2))
    return len(expanded), ctx.dataset_path, snapshot


def run_command(command: str, *, cwd: Path, timeout: int, log_path: Path) -> subprocess.CompletedProcess[str]:
    args = shlex.split(command)
    proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)
    log_path.write_text(
        f"$ {command}\n\nSTDOUT\n{'-' * 80}\n{proc.stdout}\n\nSTDERR\n{'-' * 80}\n{proc.stderr}"
    )
    return proc


def git_short_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return proc.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def ensure_results_header() -> None:
    if RESULTS_PATH.exists():
        return
    RESULTS_PATH.write_text(
        "commit	timestamp	run_mode	steps	manifest	gt	gt_delta	overall	max_regression	critical_failures	constraint_ok	probe_score	status	description\n"
    )


def append_result(*, result: prepare.ConstrainedEvalResult | None, status: str, description: str, ctx: RunContext) -> None:
    ensure_results_header()
    commit = git_short_commit()
    if result is None:
        row = [
            commit,
            now_tag(),
            RUN_MODE,
            str(ctx.max_steps),
            ctx.manifest_path.name,
            "0.0000",
            "0.0000",
            "0.0000",
            "0.0000",
            "0",
            "false",
            "0.0000",
            status,
            description,
        ]
    else:
        row = [
            commit,
            now_tag(),
            RUN_MODE,
            str(ctx.max_steps),
            ctx.manifest_path.name,
            f"{result.ground_truth:.4f}",
            f"{result.ground_truth_delta:+.4f}",
            f"{result.overall_mean:.4f}",
            f"{result.max_regression:+.4f}",
            str(result.critical_failures),
            str(result.constraint_ok).lower(),
            f"{result.probe_score:.4f}",
            status,
            description,
        ]
    with RESULTS_PATH.open("a") as f:
        f.write("	".join(row) + "\n")


def format_train_command(ctx: RunContext) -> str:
    return TRAIN_COMMAND_TEMPLATE.format(
        beta=BETA,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        max_steps=ctx.max_steps,
        warmup_ratio=WARMUP_RATIO,
        per_device_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_seq_length=MAX_SEQ_LENGTH,
        output_name=ctx.output_name,
        dataset_path=ctx.dataset_path,
    )


def maybe_activate_candidate(ctx: RunContext) -> None:
    """Deploy ephemeral candidate endpoint, capture URL, update prepare.EVAL_ENDPOINT."""
    if not ACTIVATE_CANDIDATE_COMMAND_TEMPLATE.strip():
        return
    command = ACTIVATE_CANDIDATE_COMMAND_TEMPLATE.format(
        output_name=ctx.output_name,
        base_model=BASE_MODEL,
        run_tag=ctx.run_tag,
        adapter_path=f"/adapters/experiments/{ctx.output_name}",
    )
    proc = run_command(command, cwd=ROOT, timeout=30 * 60, log_path=ctx.run_dir / "activate.log")
    if proc.returncode != 0:
        raise RuntimeError(f"Candidate activation failed, see {ctx.run_dir / 'activate.log'}")

    # activate_candidate.py prints the endpoint URL (with /v1) to stdout
    url = proc.stdout.strip().split("\n")[-1].strip()
    if url.startswith("http"):
        prepare.EVAL_ENDPOINT = url
        print(f"eval_endpoint:       {prepare.EVAL_ENDPOINT}")
    else:
        raise RuntimeError(f"Activation did not return a URL. Got: {url!r}")


def deactivate_candidate() -> None:
    """Stop the ephemeral candidate endpoint after evaluation."""
    if not DEACTIVATE_COMMAND.strip():
        return
    try:
        subprocess.run(
            shlex.split(DEACTIVATE_COMMAND),
            cwd=ROOT,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except Exception:
        pass


def default_status(result: prepare.ConstrainedEvalResult) -> str:
    if result.constraint_ok and result.ground_truth_delta >= PROMOTE_IF_GT_DELTA_AT_LEAST:
        return "keep"
    if result.constraint_ok and result.probe_score >= PROMOTE_IF_PROBE_SCORE_AT_LEAST:
        return "keep"
    return "discard"


def main() -> int:
    # Preflight: fail fast if judge API key is missing
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("FATAL: ANTHROPIC_API_KEY not set. Source ~/.config/secrets/api-keys.env")
        return 1

    if RUN_MODE not in STEP_BUDGETS:
        raise ValueError(f"Unknown RUN_MODE={RUN_MODE!r}")

    run_tag = f"{RUN_MODE}-{now_tag()}"
    run_dir = RUNS_DIR / run_tag
    ctx = RunContext(
        run_tag=run_tag,
        run_dir=run_dir,
        dataset_path=run_dir / "train.jsonl",
        manifest_path=MANIFEST_PATH,
        output_name=f"madison-qwen3-{run_tag}",
        max_steps=STEP_BUDGETS[RUN_MODE],
        max_examples=MAX_EXAMPLES_BY_MODE[RUN_MODE],
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    materialized_examples, dataset_path, snapshot = materialize_dataset(ctx)

    print("---")
    print(f"run_tag:             {ctx.run_tag}")
    print(f"run_mode:            {RUN_MODE}")
    print(f"manifest:            {ctx.manifest_path.name}")
    print(f"materialized_pairs:  {materialized_examples}")
    print(f"max_steps:           {ctx.max_steps}")
    print(f"dataset_path:        {dataset_path}")
    print(f"lr:                  {LEARNING_RATE}")
    print(f"beta:                {BETA}")
    print(f"rank/alpha:          {LORA_RANK}/{LORA_ALPHA}")
    print(f"curriculum:          {CURRICULUM}")

    if RUN_MODE == "dataset-only":
        print("status:              dataset materialized only")
        return 0

    train_command = format_train_command(ctx)
    (ctx.run_dir / "train_command.txt").write_text(train_command + "\n")
    print(f"train_command:       {train_command}")

    try:
        train_proc = run_command(
            train_command,
            cwd=ROOT,
            timeout=6 * 60 * 60,
            log_path=ctx.run_dir / "train.log",
        )
    except subprocess.TimeoutExpired:
        append_result(result=None, status="crash", description="training timeout", ctx=ctx)
        print("status:              crash")
        print("error:               training timeout")
        return 1

    if train_proc.returncode != 0:
        append_result(result=None, status="crash", description="training command failed", ctx=ctx)
        print("status:              crash")
        print(f"error:               training command failed, see {ctx.run_dir / 'train.log'}")
        return train_proc.returncode

    maybe_activate_candidate(ctx)

    try:
        result = prepare.evaluate_model(tag=ctx.run_tag, run_mode="full" if RUN_MODE == "full" else "probe")
    except Exception as exc:
        append_result(result=None, status="crash", description=f"eval failed: {exc}", ctx=ctx)
        print("status:              crash")
        print(f"error:               evaluation failed: {exc}")
        return 1
    finally:
        deactivate_candidate()

    status = default_status(result)
    description = (
        f"mode={RUN_MODE}; manifest={ctx.manifest_path.name}; lr={LEARNING_RATE}; beta={BETA}; "
        f"rank={LORA_RANK}; steps={ctx.max_steps}; curriculum={CURRICULUM}"
    )
    append_result(result=result, status=status, description=description, ctx=ctx)
    prepare.print_result(result)
    print(f"status:              {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
