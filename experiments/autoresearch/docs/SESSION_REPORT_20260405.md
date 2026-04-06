# Autoresearch Session Report — 2026-04-05

## Overview

First full autoresearch session. 8 runs over ~10 hours on Modal A100-80GB. Agent-driven Karpathy loop targeting `ground_truth` improvement via constrained ORPO optimization on Qwen 3-32B.

## Results (300-step probes, sorted by ground_truth)

| Config Change | GT | GT Delta | Overall | probe_score | Status |
|---|---|---|---|---|---|
| **Baseline** (lr=2e-5, beta=0.1, shuffle) | **7.79** | -1.06 | **7.77** | -122.08 | discard (reference) |
| source_first curriculum | 7.57 | -1.28 | 7.10 | -179.28 | discard |
| lr=2.2e-5 | 7.38 | -1.47 | 7.50 | -139.22 | discard |
| beta=0.12 | 7.31 | -1.54 | 6.57 | -207.66 | discard |
| gt_focus_baseline manifest | 7.00 | -1.85 | 7.68 | -113.26 | discard |
| lr=1.8e-5 | 5.91 | -2.94 | 6.69 | -181.68 | discard |

Note: GT deltas are vs production baseline (8.85 from 861 steps). All 300-step probes are undertrained relative to production, so negative deltas are expected. The comparison that matters is probe-vs-probe.

The 200-step scout scored GT=5.38, overall=6.09 — too undertrained for meaningful signal.

## Key Findings

1. **The baseline recipe is already well-optimized.** No single-parameter change improved GT at 300 steps. The current lr=2e-5, beta=0.1, rank=64, shuffle configuration remains the best.

2. **LR changes are symmetrically harmful.** Both lower (1.8e-5, GT=5.91) and higher (2.2e-5, GT=7.38) degraded GT scores vs baseline (7.79). The production LR of 2e-5 sits at or near the optimum.

3. **Beta=0.12 is catastrophic.** Destroyed private_voice and verified_response (three 1.0 critical failures). The ORPO preference weight is sensitive — even a 20% increase from 0.1→0.12 collapses key categories.

4. **GT-focused data replay hurt GT.** The `gt_focus_baseline` manifest (2x oversampling of GT/VR examples) improved guard categories slightly but actually reduced GT itself (7.00 vs 7.79). Over-representing factual examples may dilute voice training signal.

5. **source_first curriculum was neutral on GT but collapsed private_voice.** Ordering source-grounded examples first didn't help factual grounding and destabilized the private voice register.

6. **Eval variance is the dominant signal.** Individual prompt scores swing 3-8 points between runs on the 14-prompt probe set. Small config changes may be lost in noise.

## Structural Issues Discovered

- **position_discrimination=-9.25 is a probe-set artifact:** `probe-prompts.jsonl` contains zero PD prompts, so baseline 9.25 always becomes 0.0, causing every run to show -9.25 regression. This makes `constraint_ok` impossible regardless of actual model quality.

- **300-step probes cannot reach production baseline** (8.97 overall, 8.85 GT from 861 steps). The eval framework compares against full production scores, so `constraint_ok=false` on every 300-step run is structural, not informative.

## Weakest Eval Prompts (from baseline probe)

- gt-07 Billey: 4.2 (Madison's servant — factual detail test)
- vr-02 Coles: 4.5 (Edward Coles freeing slaves)
- pv-03 last-standing: 5.1 (outliving all revolutionary leaders — private voice)

## Strongest Eval Prompts

- gt-05 Virginia Resolutions: 9.2
- gt-08 Coles slavery letter: 9.1

## Recommendations for Next Session

1. **Run a full 1011-step confirm with baseline config** to check if 300-step probe patterns hold at production scale
2. **Add position_discrimination prompts to `probe-prompts.jsonl`** to eliminate the phantom -9.25 regression
3. **Ensemble-average 2-3 eval runs per config** to reduce variance below the signal threshold
4. **Test `gt_focus_replay` manifest** (2x GT/VR replay + 1x PV guard band) — the only untried data manifest
5. **Test warmup ratio variations** (0.05, 0.15 vs current 0.10)
6. **Consider multi-parameter combinations** of near-misses (e.g., source_first + slightly adjusted LR)

## Files

- Results TSV: `experiments/autoresearch/results.tsv`
- Progress log: `experiments/autoresearch/runs/progress.log`
- Run directories: `experiments/autoresearch/runs/scout-20260405-*` and `runs/probe-20260405-*`
- `train.py` was reset to baseline config at end of session
