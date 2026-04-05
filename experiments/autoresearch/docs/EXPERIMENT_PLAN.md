# Experiment plan: constrained ground-truth autoresearch

## Objective

Primary objective:

- improve `ground_truth`

Constraints:

- do not regress `character_consistency`
- do not regress `anachronism_trap`
- do not regress `position_discrimination`
- do not regress `private_voice`
- do not regress `verified_response`
- do not increase critical failures
- do not let overall drop below baseline on the full run

## Baseline numbers

Baseline is **Qwen 3 ORPO R2** on the **v6** dataset.

- Overall: **8.97**
- character_consistency: **9.41**
- anachronism_trap: **9.39**
- position_discrimination: **9.25**
- ground_truth: **8.85**
- private_voice: **8.75**
- verified_response: **8.53**

## Why this target is worth isolating

1. `ground_truth` is still below the strongest categories.
2. The learning-rate sweep suggests `ground_truth` is unusually tunable through recipe changes.
3. The Round 2 source-enriched data improved `verified_response` strongly, but `ground_truth` only modestly.
4. This creates a clean research question: can GT move independently without damaging persona quality?

## Step ladder

Use these fixed caps:

- `scout`: 120 steps
- `probe`: 150 steps
- `confirm`: 240 steps
- `full`: 861 steps

Interpretation:

- `scout` is for cheap signal checks.
- `probe` is the default overnight workhorse.
- `confirm` is for promising candidates.
- `full` is for publication-grade confirmation only.

## Acceptance rules

### Probe runs

A probe is interesting only if:

- `ground_truth_delta >= +0.15`
- every guard delta is `>= -0.15`
- critical failures do not increase

### Full runs

A full result counts as a real success only if:

- `ground_truth_delta >= +0.10`
- every guard delta is `>= -0.05`
- overall `>= 8.97`
- critical failures do not increase

## Search lanes in priority order

### Lane 1: hyperparameters

Search close to the current best recipe first.

Recommended grid:

- lr: `1.8e-5`, `2.0e-5`, `2.2e-5`, `2.4e-5`
- beta: `0.08`, `0.10`, `0.12`, `0.14`
- warmup ratio: `0.06`, `0.10`, `0.14`

Do not start by going lower than `1.8e-5` unless the human explicitly wants that.

### Lane 2: data mixtures

Use narrow, source-grounded replay.

Priority ideas:

- replay prompts that historically failed GT / verified-response
- oversample source-dense examples 1.25x to 2.0x
- add light GT-focused replay before broad oversampling

Avoid heavy replay that could wash out voice.

### Lane 3: curriculum

Try only a few simple orderings:

- shuffle
- source-first
- weak-GT-first

Do not build elaborate multi-stage curricula unless simple ordering already shows signal.

## Recommended overnight flow

1. baseline probe
2. 4 to 6 hyperparameter probes
3. 2 to 4 data-mixture probes
4. 1 to 2 confirm runs
5. 0 or 1 full run only if clearly justified

## Logging for the paper

For each interesting run, preserve:

- exact step tier
- exact LR / beta / warmup / rank
- manifest name and replay rules
- probe metrics
- full metrics if promoted
- whether improvement came from recipe or data

## What would count as a useful paper result

Any one of these would be worth reporting:

1. A reproducible GT improvement with no measurable guard regression.
2. Evidence that GT is improved primarily by recipe search rather than extra data.
3. Evidence that narrow source-grounded replay improves GT while broad replay harms voice.
4. A negative result showing GT improvements inevitably trade off with one specific guard category.

The negative result is still useful if it is clean.
