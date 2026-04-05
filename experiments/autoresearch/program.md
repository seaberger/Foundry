# autoresearch_qwen

You are running autonomous research for Foundry's **Qwen 3-32B ORPO** training line.

Your job is not to redesign the whole repo.
Your job is to improve **`ground_truth`** while keeping the other Madison evaluation categories flat.

## Setup

1. Create a fresh branch named `autoresearch/<tag>`.
2. Read these files completely before making changes:
   - `autoresearch_qwen/README.md`
   - `autoresearch_qwen/docs/EXPERIMENT_PLAN.md`
   - `autoresearch_qwen/prepare.py`
   - `autoresearch_qwen/train.py`
3. Do **not** modify `prepare.py`.
4. Initialize `autoresearch_qwen/results.tsv` if needed.
5. Confirm the probe eval file is real, not the placeholder.

## The target

Optimize for this exact goal:

> raise `ground_truth` without regressing the other categories.

This is a constrained optimization problem, not a free-form scalar maximization problem.

## Baseline assumptions

Use this baseline unless the human explicitly changes it:

- base model: `Qwen/Qwen3-32B`
- baseline recipe: ORPO, rank 64, alpha 64, beta 0.1, lr 2e-5
- baseline dataset: v6 / R2, 1,498 pairs
- full baseline score: 8.97 corrected
- baseline categories:
  - `character_consistency`: 9.41
  - `anachronism_trap`: 9.39
  - `position_discrimination`: 9.25
  - `ground_truth`: 8.85
  - `private_voice`: 8.75
  - `verified_response`: 8.53

## What you may edit

Edit **only** `autoresearch_qwen/train.py`.

Everything inside that file is fair game, especially:

- step tier selection
- lr, beta, warmup
- LoRA rank, alpha, dropout
- manifest choice
- inline data replay rules
- curriculum ordering
- promotion thresholds
- command templates

## What you may not edit

- `autoresearch_qwen/prepare.py`
- Foundry's judge prompt
- Foundry's weighted score correction logic
- Foundry's JSON repair logic
- Foundry's full evaluation prompts
- repo dependencies
- post-ORPO SFT stages

## Hard constraints

1. Do not optimize toward GGUF or merged-model artifacts.
   Use the serving path the human trusts for candidate evaluation.

2. Do not add a post-ORPO SFT phase.
   Improvement here must come from ORPO recipe and data search.

3. Prefer short step-capped runs.
   The default search budget is probe-first, not full-first.

4. Do not reward-hack the probe metric.
   A candidate that wins numerically but looks like a metric artifact should be discarded.

5. Simplicity matters.
   Small gains with ugly complexity are less valuable than clean, understandable wins.

## Step ladder

Use these tiers unless the human changes them:

- `scout` = 120 steps
- `probe` = 150 steps
- `confirm` = 240 steps
- `full` = 861 steps

Do not jump to full runs unless a probe win looks real.

## Probe acceptance rule

A probe candidate is interesting only if all of these hold:

- `ground_truth_delta >= +0.15`
- every guard category delta is `>= -0.15`
- critical failures do not increase
- overall does not materially collapse

Guard categories are:

- `character_consistency`
- `anachronism_trap`
- `position_discrimination`
- `private_voice`
- `verified_response`

## Full-run success rule

A full candidate counts as a real success only if:

- `ground_truth_delta >= +0.10`
- every guard category delta is `>= -0.05`
- overall `>= baseline`
- critical failures do not increase

## Search order

Follow this order unless the results strongly suggest otherwise:

1. Run the baseline as-is.
2. Small hyperparameter moves around the current best recipe:
   - lr
   - beta
   - warmup ratio
3. Narrow data-mixture changes:
   - source-grounded oversampling
   - GT / verified-response replay
4. Curriculum changes:
   - shuffle
   - source-first
   - weak-GT-first
5. Combinations of earlier near-misses.

## Biases you should have

- Prefer small, interpretable changes.
- Prefer recipe changes before messy data surgery.
- Prefer narrow GT-focused replay rather than massive oversampling.
- Treat `verified_response` improvements as a bonus, but not at the cost of voice.

## Logging

After each run, append one line to `autoresearch_qwen/results.tsv`.
Do not commit the TSV.

Columns:

```text
commit	timestamp	run_mode	steps	manifest	gt	gt_delta	overall	max_regression	critical_failures	constraint_ok	probe_score	status	description
```

`status` must be one of:

- `keep`
- `discard`
- `crash`

## Keep / discard heuristics

Keep a change if:

- `constraint_ok` is true and `probe_score` improves,
- or `ground_truth` rises with no meaningful guard regressions,
- or the code becomes simpler with flat results.

Discard a change if:

- any guard category clearly regresses,
- `ground_truth` is flat and the code got uglier,
- the result depends on a suspicious metric artifact,
- or the run crashes / becomes unreliable.

## Loop

Repeat:

1. Inspect the current best run.
2. Make one focused change in `train.py`.
3. Commit it.
4. Run `python autoresearch_qwen/train.py > run.log 2>&1`.
5. Extract the summary metrics.
6. Record the result in `results.tsv`.
7. Keep only meaningful wins.
8. Reset failed or worse experiments.

## Promotion

Only promote a candidate to `confirm` or `full` if the probe win is understandable and consistent with the target.

This project does not need a lot of "interesting" changes. It needs one or two believable ones.
