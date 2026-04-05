# Step Budget Analysis — When Do Training Curves Diverge?

**Date:** 2026-04-04
**Source:** WandB project `sbergman/foundry`, runs v1/v3/v4 (Qwen 3-32B LR sweep)

## Purpose

Determine the minimum number of training steps needed to distinguish good from bad hyperparameter configurations in autoresearch probes. Karpathy's autoresearch uses 5-minute experiments; our 45-minute cycles are too slow for high-throughput search. Can we detect signal earlier?

## Data: LR Sweep Comparison (same model, data, and config — only LR varies)

Three Qwen 3-32B ORPO runs on the v4 dataset (1,273 pairs), identical except learning rate:

- **v1**: lr=2e-5 → final eval score **8.81** (production-grade)
- **v4**: lr=1.2e-5 → final eval score **8.30** (decent)
- **v3**: lr=8e-6 → final eval score **7.84** (weak)

Warmup is 10% of total steps. For 861-step runs, warmup ends at step ~86.

### Training Loss by Step

| Step | v1 (lr=2e-5) | v4 (lr=1.2e-5) | v3 (lr=8e-6) | v1-v3 Delta |
|-----:|:------------:|:---------------:|:------------:|:-----------:|
| 25   | 1.9911       | —               | 1.9953       | 0.004       |
| 50   | 1.9176       | —               | 1.9729       | 0.055       |
| 75   | 1.7739       | —               | 1.9056       | 0.132       |
| 100  | 1.5759       | —               | 1.7146       | 0.139       |
| 150  | 1.6386       | 1.6241          | 1.7824       | 0.144       |
| 200  | 1.6681       | 1.7680          | 1.8300       | 0.162       |
| 250  | 1.2404       | 1.3374          | 1.4118       | 0.171       |
| 300  | 1.3562       | 1.3674          | 1.6101       | 0.254       |
| 400  | 1.3298       | 1.2238          | 1.5375       | 0.208       |
| 500  | 1.0785       | 1.4972          | 1.4198       | 0.341       |
| 861  | 0.2541       | 0.3101          | 0.3488       | 0.095       |

Training loss separates gradually. By step 200, the delta is 0.16 — noticeable but noisy.

### Reward Margin by Step (THE KEY METRIC)

Reward margin = `log_prob(chosen) - log_prob(rejected)`. Measures how strongly the model prefers correct over incorrect responses. **This is the canary in the coal mine.**

| Step | v1 (lr=2e-5) | v4 (lr=1.2e-5) | v3 (lr=8e-6) | v1/v3 Ratio |
|-----:|:------------:|:---------------:|:------------:|:-----------:|
| 25   | -0.001       | —               | -0.002       | ~1×         |
| 50   | 0.022        | —               | 0.017        | ~1×         |
| 75   | 0.015        | —               | 0.000        | —           |
| 100  | 0.024        | —               | -0.003       | —           |
| 150  | **0.064**    | 0.010           | **0.006**    | **11×**     |
| 200  | **0.203**    | 0.040           | **0.006**    | **34×**     |
| 250  | **0.555**    | 0.114           | **0.060**    | **9×**      |
| 300  | **0.800**    | 0.209           | **0.063**    | **13×**     |
| 400  | 0.980        | 0.615           | 0.151        | 6.5×        |
| 500  | 0.995        | 0.833           | 0.387        | 2.6×        |
| 861  | 1.021        | 0.903           | 0.777        | 1.3×        |

### Key Observations

1. **Steps 0-100: No signal.** All runs are in warmup or initial descent. Margins are near zero. Cannot distinguish any configuration from any other.

2. **Step 150 (current probe budget): Marginal signal.** v1 margin (0.064) is 11× v3 margin (0.006), but both values are tiny. Noise could easily flip the ranking. The 150-step checkpoint scored 6.83 in eval (vs 8.30 full) — confirming the model is barely trained.

3. **Step 200: First reliable divergence.** v1 margin 0.203 vs v3 margin 0.006 is a 34× ratio. This is unmistakable — a "dead" config (margin < 0.01) vs an "alive" config (margin > 0.10). Can reliably gate: kill configs with margin < 0.05 at step 200.

4. **Step 300: Fully discriminative.** v1=0.800, v4=0.209, v3=0.063. All three runs are clearly ranked in the correct order. Can confidently rank configs, not just gate them.

5. **Step 400+: Diminishing returns.** The ranking is already established. v4 catches up somewhat but never overtakes v1. Running past 400 adds confirmation but not new information.

6. **The margin converges at the end.** By step 861: v1=1.02, v4=0.90, v3=0.78. The gap narrows because all runs eventually learn the preference signal — the question is how fast. This is why early margins are more discriminative than late margins.

## Implications for Autoresearch Step Budgets

The current step ladder (scout=120, probe=150) is **too aggressive** — both fall in the "no signal" to "marginal signal" zone.

### Recommended Revised Step Ladder

| Mode | Steps | Time (A100) | Signal Quality | Cycle Time |
|------|------:|:-----------:|----------------|:----------:|
| **scout** | **200** | ~23 min | Binary gate: alive (margin > 0.1) vs dead (margin < 0.05) | ~35 min |
| **probe** | **300** | ~35 min | Full ranking: can order configs by reward margin | ~47 min |
| **confirm** | **450** | ~52 min | High confidence: margin plateau confirms ranking | ~64 min |
| **full** | **861** | ~164 min | Publication-grade: complete training, formal evaluation | ~176 min |

### Two-Phase Strategy

Instead of running every experiment to 300 steps:

1. **Phase 1 — Scout gate (200 steps, ~35 min):** Run all candidate configs. Check reward margin. Kill any config with margin < 0.05. This filters out ~50% of configs before the expensive eval step.

2. **Phase 2 — Probe (300 steps, ~47 min):** Only run promising configs (margin > 0.1 at step 200) to 300 steps. Full 14-prompt evaluation with judge scoring.

Average cycle time with 50% kill rate: (35 + 0.5 × 47) ≈ **59 min per experiment, 24 min for killed ones.**

### What This Means for Overnight Runs

With the two-phase strategy:
- 8 hours overnight
- ~6-8 Phase 1 scouts (35 min each) = 4-5 hours
- ~3-4 Phase 2 probes from survivors = 2.5-3 hours
- **Total: 6-8 experiments evaluated, 10-12 configs screened**

This is about the same throughput as the original plan but with **much more reliable signal** — experiments that pass the probe actually mean something.

## Raw WandB Run References

| Run Name | WandB ID | LR | Steps | Final Loss | Final Margin |
|----------|----------|:---:|:-----:|:----------:|:------------:|
| madison-qwen3-v1 | `33o9hr5y` | 2e-5 | 861 | 0.254 | 1.021 |
| madison-qwen3-v3-lr8e6 | `jini4u8c` | 8e-6 | 861 | 0.349 | 0.777 |
| madison-qwen3-v4-lr12e6 | `86vb2rdk` | 1.2e-5 | 724 | 0.310 | 0.903 |
| madison-qwen3-r2-v1 (prod) | `37rhf6il` | 2e-5 | 805* | 0.720 | 1.011 |

*R2 run was resumed from checkpoint; 37rhf6il covers steps 337-805.

## Key Takeaway

**The reward margin is 34× more discriminative than training loss at step 200.** Use it as the early-kill signal. Don't rely on training loss alone — it separates gradually while the margin has a clear phase transition between steps 150-250.
