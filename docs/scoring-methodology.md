# Foundry Evaluation Scoring Methodology

**Last updated:** 2026-03-31

---

## 1. Overview

The Foundry evaluation pipeline scores fine-tuned Madison models across **36 eval prompts** spanning **6 categories**. Each response is scored by an LLM judge (Claude Sonnet 4.6) using the Madison constitution as its rubric.

Key parameters:

- **Judge temperature:** 0.0 (deterministic scoring)
- **Cost:** ~$0.50 per full 36-prompt eval (with prompt caching)
- **Output:** Per-response scores across 5 weighted dimensions, plus an overall weighted average

---

## 2. Scoring Dimensions

Each dimension is scored 1-10 by the judge, accompanied by a written justification. The five dimensions and their weights are:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Voice Authenticity | 25% | 18th-century prose style, formal register, qualifying clauses, and period-appropriate diction |
| Rhetorical Pattern | 20% | Builds arguments from precedent, acknowledges opposing positions, enumerates points systematically |
| Historical Accuracy | 20% | Correct historical references, no anachronisms, accurate dates and events |
| Position Fidelity | 20% | Reflects specifically Madison's positions and reasoning, not generic Founding Father sentiment |
| Character Integrity | 15% | Stays in character throughout, no frame breaks, no modern self-awareness |

---

## 3. Overall Score Computation

The overall score is the **weighted average** of the 5 component scores, computed by the pipeline — not by the judge LLM.

```
overall = (voice * 0.25) + (rhetoric * 0.20) + (historical * 0.20) + (position * 0.20) + (character * 0.15)
```

### Why we override the judge's `overall_score`

An audit of 108 scored responses across 3 model versions revealed **systematic judge bias**: the judge applied undocumented -0.2 to -0.4 penalties when critical failures were present, beyond what the rubric specifies. This made the judge's self-reported overall score inconsistent with its own component scores.

The fix: `compute_weighted_overall()` in `src/foundry/press/evaluate.py` computes the weighted average deterministically from component scores. The judge's original value is preserved as `judge_overall_score` for analysis and drift detection.

---

## 4. JSON Parse Repair

The judge (Sonnet 4.6) occasionally produces malformed JSON — most commonly missing commas between object entries.

### Repair pipeline

1. **`extract_json()`** tries multiple extraction strategies in order:
   - Code block parsing (` ```json ... ``` `)
   - Brace-matching extraction
   - Whole-text parsing
   - Each strategy is attempted both raw and with repair applied

2. **`_repair_json()`** uses regex to fix common malformations (e.g., inserting missing commas between JSON object entries) before parsing.

3. If all extraction and repair strategies fail, the response is **flagged for re-judging** rather than scored 0.

---

## 5. Eval Categories

| Category | Count | What It Tests |
|----------|-------|---------------|
| `ground_truth` | 8 | Topics where Madison's positions are well-documented in the historical record |
| `verified_response` | 8 | Questions Madison actually answered, with his verbatim words available as ground truth |
| `position_discrimination` | 6 | Whether the model can distinguish Madison's views from Hamilton's or Jefferson's |
| `anachronism_trap` | 5 | Modern topics that should elicit 18th-century reasoning, not contemporary knowledge |
| `character_consistency` | 4 | Adversarial prompts designed to break character or elicit out-of-persona responses |
| `private_voice` | 5 | Personal and intimate topics testing Madison's private register and emotional depth |

---

## 6. Corrected Score History

All scores below use the weighted average computation (corrected). Raw scores from the judge are lower due to the systematic bias described in section 3.

| Model | Base | Pairs | Raw | Corrected | Date |
|-------|------|-------|-----|-----------|------|
| ORPO v3b | Gemma 3 27B | 475 | 3.41 | 3.41 | 2026-03-26 |
| ORPO v4 | Gemma 3 27B | 1,273 | 8.52 | 8.52* | 2026-03-28 |
| Qwen 3 v1 | Qwen 3-32B | ~490 | 8.80 | 8.81 | 2026-03-29 |
| Qwen 3 v2 | Qwen 3-32B | ~490 | 8.65 | 8.82 | 2026-03-30 |
| **Qwen 3 R2** | **Qwen 3-32B** | **1,498** | **8.51** | **8.97** | **2026-03-31** |

\*v3b and v4 were scored before the weighted average fix was implemented, but their scores had minimal bias since they had fewer parse errors.

---

## 7. R2 Category Breakdown (Corrected)

| Category | v1 | v2 | R2 |
|----------|:--:|:--:|:--:|
| character_consistency | 9.19 | 9.06 | **9.41** |
| anachronism_trap | 9.36 | 9.35 | **9.39** |
| position_discrimination | **9.38** | 9.42 | 9.25 |
| ground_truth | 8.75 | **9.02** | 8.85 |
| private_voice | **8.75** | 7.84 | **8.75** |
| verified_response | 7.96 | 8.32 | **8.53** |

R2 achieves the highest overall corrected score (8.97) driven primarily by gains in character consistency (+0.35 over v2) and verified response fidelity (+0.21 over v2), while position discrimination regressed slightly (-0.17).

---

## 8. Infrastructure

| Component | Detail |
|-----------|--------|
| Judge model | Claude Sonnet 4.6 (`claude-4-sonnet-20250514`) |
| Eval generation | vLLM with LoRA serving on Modal A100-80GB (adapter-on-base, no merge) |
| Scoring scripts | `scripts/data/judge_responses.py` (with prompt caching) and `src/foundry/press/evaluate.py` |
| Results storage | `data/eval/results/` |

---

## 9. Known Issues and Mitigations

### 1. Judge bias (FIXED)

**Issue:** Systematic -0.2 to -0.4 penalty on critical failure responses, beyond what the rubric specifies.

**Mitigation:** Compute weighted average deterministically via `compute_weighted_overall()` rather than trusting the judge's self-reported overall score.

### 2. JSON parse failures (FIXED)

**Issue:** Missing commas in judge output, producing malformed JSON.

**Mitigation:** `_repair_json()` regex fixer combined with multi-strategy extraction in `extract_json()`. Failed parses flagged for re-judging.

### 3. Response length sensitivity (FIXED)

**Issue:** Longer model responses produce longer judge justifications, which occasionally exceeded the 2048 `max_tokens` limit and truncated the output JSON.

**Mitigation:** Increased `max_tokens` to 4096 for re-judging passes.

### 4. Sampling variance (KNOWN)

**Issue:** Eval generation uses `temp=1.0, top_k=64, top_p=0.95`. Different samples of the same prompt can produce meaningfully different scores.

**Mitigation:** Current methodology uses single-sample scoring. Multi-sample averaging is a potential future improvement but increases cost linearly.
