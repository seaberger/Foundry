# autoresearch_qwen

A constrained, Karpathy-style autoresearch scaffold for **Foundry's Qwen 3-32B ORPO line**.

This is not meant to let an agent roam the whole Foundry repo. It is meant to let an agent search a **small training/eval surface** around the current best Madison model.

The core target for this version is:

> **Improve `ground_truth` without regressing the other Madison evaluation categories.**

That is a stronger and cleaner target than chasing raw overall score. It also lines up with the current Foundry results and with the likely paper angle.

## Current Foundry baseline this scaffold assumes

These numbers are pulled from the public Foundry docs as of 2026-04-01:

- **Production baseline:** Qwen 3-32B ORPO R2 on the **v6** dataset
- **Dataset size:** **1,498 preference pairs**
- **Best corrected overall score:** **8.97 / 10**
- **R2 category scores:**
  - `character_consistency`: **9.41**
  - `anachronism_trap`: **9.39**
  - `position_discrimination`: **9.25**
  - `ground_truth`: **8.85**
  - `private_voice`: **8.75**
  - `verified_response`: **8.53**
- **Best fixed recipe so far:** Qwen 3-32B, LoRA rank 64 / alpha 64, beta 0.1, lr 2e-5, 3 epochs
- **Important finding:** ORPO -> SFT is abandoned; future gains should come from additional ORPO rounds, not post-ORPO SFT.
- **Useful clue:** the learning-rate sweep showed `ground_truth` was unusually sensitive to LR compared with the stronger voice categories.

Source links to keep handy:

- Repo: `https://github.com/seaberger/Foundry`
- Training methodology: `https://github.com/seaberger/Foundry/blob/main/docs/training-methodology.md`
- Training results: `https://seaberger.github.io/Foundry/training-results/`
- Paper: `https://seaberger.github.io/Foundry/paper/`

## Why the target is `ground_truth`

The baseline is already strong overall. The point of autoresearch here is not to rebuild the system or to squeeze a meaningless +0.02 into the average.

The point is to test a sharper hypothesis:

> In a historical character model fine-tuned with ORPO, can factual grounding (`ground_truth`) be improved while persona-quality categories remain flat?

That is a more defensible research result than "overall score went up a little".

## Design principles

This scaffold follows the same basic shape as Karpathy's `autoresearch`:

- `prepare.py` is the **immutable trust boundary**
- `train.py` is the **mutable experiment surface**
- the training backend is separated from the objective function
- most experiments should be **short probe runs**, not full training runs

## Step ladder

Use a fixed step ladder instead of free-form training budgets.

- `scout`: **120 steps**
- `probe`: **150 steps**
- `confirm`: **240 steps**
- `full`: **861 steps**

Why these numbers:

- Foundry's full Qwen v1 run is documented at **861 steps**.
- That same run took about **164 minutes** on an A100-80GB.
- So 150 steps is roughly a **30 minute** probe at the observed throughput.

That is slow compared with Karpathy's tiny toy runs, but still realistic for an overnight A100 budget.

## Constraint policy

The goal is **not** "maximize a single scalar at any cost".

For probe runs:

- seek **`ground_truth` delta >= +0.15** vs baseline on the probe set
- reject candidates if any guard category drops by more than **0.15**
- reject candidates if critical failures increase

For full runs:

- claim success only if **`ground_truth` delta >= +0.10** on the full corrected evaluation
- every other category must stay within **-0.05** of baseline
- overall must not fall below baseline
- critical failures must not increase

Those thresholds are encoded in `prepare.py` and documented in `docs/EXPERIMENT_PLAN.md`.

## Search lanes worth exploring first

1. **Hyperparameters near the current best recipe**
   - lr around 1.8e-5 to 2.4e-5
   - beta around 0.08 to 0.14
   - warmup ratio
   - optionally rank 48 / 64 / 80 after easier knobs are exhausted

2. **Ground-truth-focused data mixtures**
   - oversample source-grounded examples
   - replay weak `ground_truth` / `verified_response` prompts
   - keep the replay narrow so you do not wash out voice

3. **Curriculum**
   - source-dense first
   - weak-GT first
   - compare against simple shuffle

## Recommended overnight budget

A realistic 8-ish hour run on one A100 should look more like this:

- 1 baseline probe run
- 6 to 10 probe runs at 150 steps
- 1 to 2 confirm runs at 240 steps
- 0 or 1 full run only if the probe signal is clearly real

Do not spend the whole night on full 861-step runs.

## File layout

```text
autoresearch_qwen/
  README.md
  program.md
  prepare.py
  train.py
  results.tsv
  backend/
    modal_train_orpo_qwen_autoresearch.py
  docs/
    EXPERIMENT_PLAN.md
    INTEGRATION_NOTES.md
  manifests/
    baseline_r2.json
    gt_focus_baseline.json
    gt_focus_source_weighted.json
    gt_focus_replay.json
  eval/
    probe-prompts.jsonl
    README.md
  runs/
```

## Important boundaries

### What should stay fixed

- Foundry's judge prompt
- Foundry's weighted-average correction logic
- JSON repair logic
- the full Madison evaluation set
- the broad repo structure
- the choice to avoid post-ORPO SFT here

### What should be searched

- step budget tier
- lr, beta, warmup, rank
- data replay / oversampling rules
- curriculum ordering
- manifest selection

## How this is meant to be used with Claude Code

1. Copy this folder into the Foundry repo root.
2. Replace `eval/probe-prompts.jsonl` with a real held-out probe subset.
3. Wire the training backend and serving/eval hook to your actual Foundry infra.
4. Point Claude Code at `autoresearch_qwen/program.md`.
5. Let it mutate `train.py` and run experiments.

## What still needs real-world wiring

This scaffold is intentionally concrete but not presumptuous about your serving infra.

It already includes:

- a mutable `train.py` wrapper
- a drop-in Modal backend starting point
- a fixed constrained objective in `prepare.py`
- seed manifests for GT-focused data mixtures

You still need to wire:

- the exact adapter activation / serving path used for evaluation
- the final probe prompt subset
- any dataset metadata selectors that differ from the assumptions in the sample manifests

## Good paper framing

A tighter framing than "novel character fine-tuning" is:

> We evaluate whether factual grounding in a historical-character model can be improved through constrained ORPO optimization without degrading persona fidelity.

That is narrow, credible, and worth testing.
