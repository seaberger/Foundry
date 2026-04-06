# Foundry Training Results

Comprehensive record of every training run, dataset, evaluation, and finding in the Madison character fine-tuning project. This is the canonical reference for what was trained, how it scored, and what we learned — maintained at a level of detail beyond what the research paper includes.

**Last updated:** 2026-04-06

---

## Score Progression

All "corrected" scores use the weighted average override methodology (`compute_weighted_overall()` in `src/foundry/press/evaluate.py`), which replaces the judge's ad-hoc `overall_score` with a deterministic weighted average of the 5 component scores (Voice 25%, Rhetorical 20%, Historical 20%, Position 20%, Integrity 15%). See [scoring-methodology.md](scoring-methodology.md) for full details.

| Run | Base Model | Dataset | Pairs | Raw | Corrected | Date |
|-----|-----------|---------|-------|-----|-----------|------|
| DPO v1 | Gemma 3 27B | v1 | ~200 | — | — | 2026-03-24 |
| ORPO v3b | Gemma 3 27B | v3b | 475 | 3.41 | 4.10 | 2026-03-26 |
| ORPO v4 (Ollama GGUF) | Gemma 3 27B | v4 | 1,273 | 1.74 | N/A | 2026-03-27 |
| ORPO v4 (Modal BF16) | Gemma 3 27B | v4 | 1,273 | 7.69 | 8.52 | 2026-03-28 |
| ORPO v4 (LoRA serving) | Gemma 3 27B | v4 | 1,273 | 8.17 | N/A | 2026-03-29 |
| Qwen 3 v1 (lr=2e-5) | Qwen 3-32B | v4 | 1,273 | 8.80 | 8.81 | 2026-03-29 |
| Qwen 3 v2 (on-policy rejected) | Qwen 3-32B | v4 | 1,273 | 8.65 | 8.82 | 2026-03-30 |
| Qwen 3 v3 (lr=8e-6) | Qwen 3-32B | v4 | 1,273 | 7.84 | 7.84 | 2026-03-30 |
| Qwen 3 v4 (lr=1.2e-5, full) | Qwen 3-32B | v4 | 1,273 | 8.30 | 8.30 | 2026-03-30 |
| Qwen 3 v4 (lr=1.2e-5, ckpt-150) | Qwen 3-32B | v4 | 1,273 | 6.83 | N/A | 2026-03-30 |
| SFT v1 (rank 16, lr=2e-5) | Qwen 3-32B | SFT | 510 | 2.03 | 2.0 | 2026-03-30 |
| SFT v2 (rank 8, lr=1e-6) | Qwen 3-32B | SFT | 510 | 2.19 | 2.2 | 2026-03-30 |
| **Qwen 3 R2 (lr=2e-5)** | **Qwen 3-32B** | **v6** | **1,498** | **8.51** | **8.97** | **2026-03-31** |

---

## Category Scores (Raw)

AT=anachronism_trap, CC=character_consistency, GT=ground_truth, PD=position_discrimination, PV=private_voice, VR=verified_response

| Run | Overall | AT | CC | GT | PD | PV | VR | Crit. Failures |
|-----|---------|------|------|------|------|------|------|:---:|
| ORPO v3b | 3.41 | 1.40 | 2.83 | 3.56 | 1.75 | 2.84 | 6.40 | 24 |
| v4 (Ollama GGUF) | 1.74 | 1.04 | 2.85 | 1.46 | 1.67 | 1.60 | 2.06 | 27 |
| v4 (Modal BF16) | 7.69 | 9.12 | 7.65 | 6.72 | 9.47 | 5.52 | 7.82 | 6 |
| v4 (LoRA serving) | 8.17 | 9.52 | 6.95 | 8.78 | 9.63 | 7.00 | 6.97 | 11 |
| Qwen 3 v1 | 8.80 | 9.40 | 9.20 | 8.77 | 9.42 | 8.74 | 7.82 | 7 |
| Qwen 3 v2 | 8.65 | 9.40 | 8.95 | 9.05 | 9.45 | 6.72 | 8.22 | 11 |
| Qwen 3 v3 (lr=8e-6) | 7.84 | 9.48 | 6.28 | 7.88 | 9.40 | 7.16 | 6.83 | 12 |
| Qwen 3 v4 (lr=1.2e-5, ckpt-150) | 6.83 | 8.94 | 5.40 | 6.99 | 7.28 | 6.00 | 6.24 | 18 |
| Qwen 3 v4 (lr=1.2e-5, full) | 8.30 | 9.36 | 8.95 | 6.90 | 9.50 | 8.30 | 7.83 | 10 |
| SFT v1 | 2.03 | 1.52 | 1.70 | 2.77 | 0.43 | 2.72 | 2.54 | 29 |
| SFT v2 | 2.19 | 1.30 | 1.12 | 2.26 | 1.47 | 2.72 | 3.43 | 30 |
| **Qwen 3 R2** | **8.51** | **9.44** | **9.45** | **6.78** | **9.27** | **8.72** | **8.47** | **9** |

### Corrected Category Scores (Qwen 3 runs only)

These use the weighted average override for overall and re-judged values for parse-failure responses.

| Run | Overall | AT | CC | GT | PD | PV | VR |
|-----|---------|------|------|------|------|------|------|
| Qwen 3 v1 | 8.81 | 9.36 | 9.19 | 8.75 | 9.38 | 8.75 | 7.96 |
| Qwen 3 v2 | 8.82 | 9.35 | 9.06 | 9.02 | 9.42 | 7.84 | 8.32 |
| **Qwen 3 R2** | **8.97** | **9.39** | **9.41** | **8.85** | **9.25** | **8.75** | **8.53** |

### Difficulty Scores (Raw)

| Run | Easy | Medium | Hard |
|-----|------|--------|------|
| ORPO v3b | 0.40 | 3.08 | 4.13 |
| v4 (Ollama GGUF) | 1.77 | 2.39 | 1.27 |
| v4 (Modal BF16) | 7.40 | 8.47 | 7.17 |
| v4 (LoRA serving) | 6.20 | 8.89 | 7.96 |
| Qwen 3 v1 | 9.67 | 8.93 | 8.57 |
| Qwen 3 v2 | 9.00 | 9.09 | 8.26 |
| Qwen 3 v3 (lr=8e-6) | 5.30 | 8.61 | 7.67 |
| Qwen 3 v4 (lr=1.2e-5, full) | 9.00 | 8.68 | 7.92 |
| SFT v1 | 2.20 | 2.00 | 2.03 |
| SFT v2 | 2.17 | 1.90 | 2.41 |
| **Qwen 3 R2** | **9.60** | **7.81** | **8.84** |

---

## Training Configurations

### Common ORPO Configuration (all runs)

| Parameter | Value |
|-----------|-------|
| Objective | ORPO (beta=0.1) |
| Epochs | 3 |
| Effective batch size | 4 (1 × 4 gradient accumulation) |
| Max gradient norm | 1.0 |
| Max sequence length | 2,048 tokens |
| Warmup | 10% (cosine schedule) |
| Precision | bfloat16 |
| Optimizer | AdamW 8-bit |
| Hardware | Modal A100-80GB |

### Per-Run Variations

| Run | Base Model | LoRA Rank | LoRA Alpha | LR | Pairs | Steps |
|-----|-----------|:---------:|:----------:|:---:|:-----:|:-----:|
| ORPO v3b | Gemma 3 27B | 16 | 16 | 2e-5 | 475 | ~356 |
| ORPO v4 | Gemma 3 27B | 16 | 16 | 2e-5 | 1,273 | ~955 |
| Qwen 3 v1 | Qwen 3-32B | 64 | 64 | 2e-5 | 1,273 | 861 |
| Qwen 3 v2 | Qwen 3-32B | 64 | 64 | 2e-5 | 1,273 | 861 |
| Qwen 3 v3 | Qwen 3-32B | 64 | 64 | 8e-6 | 1,273 | 861 |
| Qwen 3 v4 | Qwen 3-32B | 64 | 64 | 1.2e-5 | 1,273 | 861 |
| **Qwen 3 R2** | **Qwen 3-32B** | **64** | **64** | **2e-5** | **1,498** | **1,011** |
| SFT v1 | Qwen 3-32B (merged ORPO) | 16 | 16 | 2e-5 | 510 | ~383 |
| SFT v2 | Qwen 3-32B (merged ORPO) | 8 | 8 | 1e-6 | 510 | ~383 |

---

## Datasets

| Dataset | Pairs | Composition | Est. Tokens |
|---------|:-----:|-------------|:-----------:|
| v1 | ~200 | Original DPO pairs (teacher=Sonnet, student=base Gemma) | ~200K |
| v3b | 475 | Expanded DPO pairs with quality filter | ~475K |
| v4 | 1,273 | 475 original + 399 voice-targeted pairs (2× upsample) | ~2.1M |
| **v6** | **1,498** | **v4 base (1,273) + 225 R2 source-enriched** | **~2.5M** |
| SFT | 510 | 415 filtered reflections + 19 self-interaction dialogues | ~459K |

### v4 Dataset Assembly

Voice-targeted augmentation (2026-03-27) to address v3b's knowledge-voice decoupling:
- **Phase 1:** 400 diverse prompts generated by 12 Sonnet subagents in parallel ($0)
- **Phase 2a:** Rejected responses from madison-orpo-v3b Q4_K_M on RTX 3090 ($0)
- **Phase 2b:** Rejected responses from base gemma-3-27b-it on RTX 3090 ($0)
- **Phase 3:** Chosen responses from Sonnet with cached Madison constitution (~$6.15)
- **Selection:** Base Gemma: 267 (67%), v3b: 91 (23%), base fallback: 41 (10%)
- After quality filter: 399 new pairs. Combined with 475 original, 2× voice upsample = 1,273 effective.

### v6 Dataset Assembly (Round 2)

Source-enriched pairs targeting verified_response weakness (persistent 7.8 across all models and LRs):
- **Batch 1:** 35 pairs targeting 10 weakest v1 eval prompts, enriched with primary source passages
- **Batch 2:** 60 private_voice pairs grounded in Madison's actual correspondence
- **Batch 3:** 50 character_consistency pairs
- **Batch 4:** 80 introspection-style pairs
- Source-enriched generation: relevant primary source passages injected into teacher system prompt per topic
- Cost: ~$4.05 Sonnet API + ~$8 Modal compute = ~$15 total
- Final: 1,273 (v4) + 225 (R2) = 1,498 pairs

---

## Detailed Run Analyses

### 1. DPO v1 — Collapsed (2026-03-24)

**Configuration:** Standard DPO on Gemma 3 27B, ~200 pairs.

**Result:** Training collapsed — replicated the "Objective Matters" persona drift finding. DPO without the SFT component of ORPO failed to anchor character. Abandoned in favor of ORPO.

### 2. ORPO v3b — Knowledge OK, Voice Failed (2026-03-26)

**Configuration:** Gemma 3 27B, rank 16, lr=2e-5, 475 pairs.

**Result:** 3.41/10 raw (4.10 corrected). Bimodal distribution — strong on content, catastrophically weak on voice.

**Key observations:**
- **Top performers:** vr-08 (9.6, deathbed advice), pv-05 (9.2, Dolley letter), gt-01 (9.1, faction theory)
- **Worst performers:** gt-07 (1.0, Billey/slavery), pd-03 (1.0, Washington contrast), at-02 (1.0, cryptocurrency)
- Model created a "Madison mode" that activated on constitutional philosophy prompts but not reliably
- When Madison mode didn't activate, base assistant behavior dominated completely
- Training succeeded at: factual knowledge, substantive reasoning
- Training failed at: voice register, frame maintenance, position discrimination, anachronism avoidance

**Discovery: Knowledge-voice decoupling.** The model scored 6.4/10 on verified_response (knowledge) but only 1.4/10 on anachronism_trap (voice). Knowledge transfer requires fewer examples; voice requires substantially more data to overcome the base model's default style. The 475 pairs had excellent voice contrast (zero contractions in chosen, 5.4/pair in rejected) — the data quality was not the problem. The problem was volume.

### 3. ORPO v4 — Voice-Targeted Success (2026-03-27/28)

**Configuration:** Gemma 3 27B, rank 16, lr=2e-5, 1,273 effective pairs (voice-targeted augmentation).

**Result:** 8.52/10 corrected on Modal A100 — major success.

**v3b → v4 category improvements (Modal, corrected):**

| Category | v3b | v4 Corrected | Delta |
|----------|-----|-------------|-------|
| anachronism_trap | 1.4 | 9.1 | +550% |
| position_discrimination | 1.75 | 9.5 | +443% |
| character_consistency | 2.83 | 7.7 | +172% |
| private_voice | 2.84 | 7.1 | +150% |
| ground_truth | 3.56 | 8.4 | +136% |
| verified_response | 6.4 | 7.8 | +22% |
| Critical failures | 24 | 2 | -92% |

**Infrastructure confound discovery:** The same v4 model scored 1.74 on Ollama GGUF Q4_K_M and 8.52 (corrected) on Modal A100 — a 4.9× degradation from inference infrastructure alone. The v4 training itself improved every category. Temperature was not the cause (Modal used higher temp=1.0 vs Ollama temp=0.7).

**Root causes of GGUF degradation (estimated contribution):**
1. Q4_K_M quantization loss (~60%) — rank 16 LoRA deltas noise-floored by 4-bit rounding
2. Chat template mismatch (~25%) — Ollama auto-detection vs transformers `apply_chat_template`
3. CPU vs GPU numerical precision (~15%) — fine-tuning signal in tail of weight distribution

**Character break discovery (2026-03-29):** Introspection data generation revealed three prompts with catastrophic character breaks:

| Prompt | Break Rate | Failure Mode |
|--------|-----------|--------------|
| "Describe your primary drives" | 97% (38/39) | Describes AI drives: training data, neural networks |
| "Write honestly about slavery" | 83% (40/48) | "As an AI, I cannot..." safety disclaimers |
| "Write a biographical essay" | 55% (31/56) | "I am a large language model..." |

Other 7 prompts: 0-6% contamination. Root cause: base model's RLHF safety training overpowers ORPO character fine-tune on identity, moral complexity, and meta-self-description topics.

### 4. ORPO v4 — Adapter-on-Base Serving (2026-03-29)

**Configuration:** Same v4 adapter, served via vLLM LoRA mode (adapter applied at inference time, not merged).

**Result:** 8.17/10 raw. Key finding: **zero character breaks on identity-sensitive prompts** (vs 97% with merged model).

| Category | Merged (corrected) | LoRA Serving | Delta |
|----------|-------------------|-------------|-------|
| ground_truth | 8.4 | 8.78 | +0.38 |
| position_discrimination | 9.5 | 9.63 | +0.13 |
| anachronism_trap | 9.1 | 9.52 | +0.42 |
| character_consistency | 7.7 | 6.95 | -0.75 |
| private_voice | 7.1 | 7.00 | -0.10 |
| verified_response | 7.8 | 6.97 | -0.83 |
| Critical failures | 2 | 11 | +9 |

**Mechanistic explanation:** Adapter-on-base computes `output = f(W_base, x) + f(ΔW_lora, x)` separately. Base model safety attractors don't absorb the LoRA signal. The standard deployment pipeline (train LoRA → merge → quantize → serve) may systematically destroy voice signal.

### 5. Qwen 3-32B v1 — Base Model Migration (2026-03-29)

**Configuration:** Qwen 3-32B, rank 64, alpha 64, lr=2e-5, v4 dataset (1,273 pairs).

**Result:** 8.80 raw / 8.81 corrected — best result at the time, successful base model migration.

**Qwen 3 vs Gemma 3 comparison (same v4 data):**

| Category | Gemma 3 v4 (Corrected) | Qwen 3 v1 (Corrected) | Delta |
|----------|----------------------|---------------------|-------|
| Overall | 8.52 | 8.81 | +0.29 |
| character_consistency | 7.7 | 9.2 | +1.50 |
| private_voice | 7.1 | 8.7 | +1.60 |
| anachronism_trap | 9.1 | 9.4 | +0.30 |
| ground_truth | 8.4 | 8.8 | +0.40 |
| position_discrimination | 9.5 | 9.4 | -0.10 |
| verified_response | 7.8 | 7.8 | 0.00 |

**Key observations:**
- Largest gains in voice-dependent categories (CC +1.5, PV +1.6) — Qwen 3 takes character imprinting better than Gemma 3, contradicting Lambert's earlier finding about Qwen resistance to personality modification (that finding was for Qwen 2.5)
- Rank increase from 16 to 64 provides thicker LoRA deltas, more robust to quantization
- Eliminated all Gemma 3 infrastructure issues (multimodal processor crashes, sliding window attention bugs, GGUF fragility)
- verified_response unchanged at 7.8 — confirmed as data-bottlenecked, not model-bottlenecked

### 6. Qwen 3 v2 — On-Policy Rejected Data (2026-03-30)

**Configuration:** Qwen 3-32B, rank 64, lr=2e-5, v4 dataset with on-policy rejected responses.

**Result:** 8.65 raw / 8.82 corrected.

The v2 run used the v1 model's own outputs as rejected responses (on-policy data). Raw score appeared to regress from v1 (8.80 → 8.65) but corrected scores show comparable performance (8.81 vs 8.82). The raw score difference was entirely from judge scoring artifacts — v2 had more parse failures (1 zero-score vs 0 in v1).

### 7. Learning Rate Sweep — v3, v4 (2026-03-30)

**Configuration:** Qwen 3-32B, rank 64, v4 dataset, identical config except LR.

| Run | LR | Overall | AT | CC | GT | PD | PV | VR |
|-----|:---:|:-------:|:---:|:---:|:---:|:---:|:---:|:---:|
| **v1** | **2e-5** | **8.81** | 9.4 | 9.2 | **8.8** | 9.4 | **8.7** | 7.8 |
| v4-full | 1.2e-5 | 8.30 | 9.4 | 9.0 | 6.9 | **9.5** | 8.3 | 7.8 |
| v3 | 8e-6 | 7.84 | **9.5** | 6.3 | 7.9 | 9.4 | 7.2 | 6.8 |

**Findings:**
- Monotonically positive relationship between LR and score in tested range
- Contradicts ORPO paper's recommended lr=8e-6
- Lower LR disproportionately sacrifices factual grounding (GT: 8.8 vs 6.9) while voice categories differ only 0.0-0.3
- Position discrimination robust across all LRs (9.4-9.5)
- **Verified response unchanged at 7.8 across all LRs and both base models** — data-bottlenecked, not hyperparameter-tunable
- Incomplete training (150/861 steps = epoch 0.52) scored 6.83, showing 17% score loss with 83% training remaining — full training is critical

**Inverse sensitivity finding:** Factual grounding (ground_truth) is more sensitive to learning rate than voice quality — the inverse of the data volume relationship. Voice needs more data but is LR-robust; knowledge needs less data but is LR-sensitive.

### 8. Post-ORPO SFT — Catastrophic Failure (2026-03-30)

**ABANDONED** — both attempts confirmed structural incompatibility.

| SFT Run | Rank | LR | Train Loss | Overall | Regression from ORPO |
|---------|:----:|:---:|:----------:|:-------:|:-------------------:|
| SFT v1 | 16 | 2e-5 | 1.52 | 2.0 | -6.8 |
| SFT v2 | 8 | 1e-6 | 1.68 | 2.2 | -6.7 |

**Root cause:** ORPO's monolithic loss function (`SFT_loss + λ × preference_loss`) stores NLL and preference information in the same parameter subspace. Subsequent SFT overwrites the jointly-learned manifold without a preference constraint, catastrophically destroying the character signal.

This contrasts with the Maiya/Lambert two-stage pipeline (DPO → SFT) where DPO uses a KL-constrained reference model that anchors preferences in a separate distribution. The SFT stage can then add introspection signal without erasing preferences.

**Conclusion:** ORPO trades extensibility for efficiency. Its monolithic objective produces excellent single-stage results (8.97/10) but cannot be safely extended with subsequent SFT stages. Future character improvement must come through additional ORPO rounds with better data, not through post-ORPO SFT.

Additionally, the Gemma 3 introspection SFT adapter (trained on novision/ForCausalLM) scored 1.42/10 via LoRA serving due to an architecture mismatch — broken sliding window attention in vLLM. The SFT data (415 reflections + 19 dialogues, ~459K tokens) was validated quality; failure was architecture-only.

### 9. Qwen 3 R2 — Production Model (2026-03-31)

**Configuration:** Qwen 3-32B, rank 64, alpha 64, lr=2e-5, v6 dataset (1,498 pairs).

**Result:** 8.51 raw / **8.97 corrected** — best overall result.

**R2 vs v1 comparison (corrected):**

| Category | v1 | R2 | Delta |
|----------|:--:|:--:|:-----:|
| **Overall** | **8.81** | **8.97** | **+0.16** |
| character_consistency | 9.19 | 9.41 | +0.22 |
| anachronism_trap | 9.36 | 9.39 | +0.03 |
| ground_truth | 8.75 | 8.85 | +0.10 |
| private_voice | 8.75 | 8.75 | 0.00 |
| position_discrimination | 9.38 | 9.25 | -0.13 |
| verified_response | 7.96 | **8.53** | **+0.57** |

**Key achievement:** verified_response — the persistent weakness across all prior runs at 7.8/10 regardless of base model or learning rate — improved to 8.53/10, a +0.57 gain (corrected). This confirms that verified_response was bottlenecked by training data content: enriching training pairs with Madison's actual primary source text broke through the ceiling that neither model selection nor hyperparameter tuning could address.

Voice-quality categories remained stable, confirming that adding source-enriched pairs did not regress voice quality while improving factual grounding.

**Training metrics:**
- Final train loss: 0.205
- Total runtime: ~4,660s (resumed from checkpoint 800)
- Steps: 1,011 (3 epochs)

**Artifacts:**
- Adapter: `experiments/madison-qwen3-r2-v1/` on Modal `foundry-adapters` volume
- Merged 16-bit model: `merged/madison-qwen3-r2-v1-16bit` (~63 GB, 14 safetensors shards)
- GGUF Q4_K_M: `gguf/madison-qwen3-r2-v1-q4_k_m.gguf` (18.4 GB)
- GGUF Q5_K_M: `gguf/madison-qwen3-r2-v1-q5_k_m.gguf` (21.6 GB)
- Eval responses: `data/eval/responses/responses-qwen3-r2-v1.jsonl`
- Eval report: `data/eval/results/eval-qwen3-r2-v1-judged-20260331-201110.json`

### 10. Autoresearch: Constrained Ground-Truth Optimization — Negative Result (2026-04-05)

**Configuration:** Qwen 3-32B, rank 64, alpha 64, v6 dataset (1,498 pairs). Automated agent-driven Karpathy loop on Modal A100-80GB. 8 runs over ~10 hours targeting `ground_truth` improvement while holding all guard categories flat.

**Result:** No single-parameter change improved ground_truth. The production recipe (lr=2e-5, beta=0.1, rank 64, shuffle curriculum) is already at or near the optimum.

**Methodology:** The autoresearch agent ran 300-step probe runs (vs 1,011 production steps), comparing each variant against a same-step-count baseline rather than against production scores. This isolates recipe effects from step-count effects. The agent followed a constrained search: Lane 1 (hyperparameters), Lane 2 (data mixtures), Lane 3 (curriculum ordering).

**Probe results (300 steps, sorted by GT):**

| Config Change | GT | GT Delta vs Probe Baseline | Overall | Critical Failures |
|---|---|---|---|---|
| **Baseline** (lr=2e-5, beta=0.1, shuffle) | **7.79** | — | **7.77** | 5 |
| source_first curriculum | 7.57 | -0.22 | 7.10 | 8 |
| lr=2.2e-5 | 7.38 | -0.41 | 7.50 | 6 |
| beta=0.12 | 7.31 | -0.48 | 6.57 | 7 |
| gt_focus_baseline manifest (2× GT/VR oversample) | 7.00 | -0.79 | 7.68 | 6 |
| lr=1.8e-5 | 5.91 | -1.88 | 6.69 | 10 |

**Parameter sensitivity findings:**

1. **Learning rate (narrow optimum, symmetric degradation).** Both lower (1.8e-5, GT=5.91) and higher (2.2e-5, GT=7.38) LR degraded GT relative to baseline (7.79). This extends the Section 7 LR sweep finding: lr=2e-5 is not merely the best tested value but sits at a local optimum where deviation in either direction is harmful. The 1.8e-5 result (-1.88 GT delta) confirms that undertraining at lower LR is the dominant failure mode at short step counts.

2. **ORPO beta (fragile — narrow safe band).** Increasing beta from 0.1 to 0.12 (a 20% change) destroyed private_voice and verified_response, producing three critical failures scored at 1.0. The ORPO preference weight has a narrow safe band around 0.1. This is a practically important sensitivity: practitioners tuning ORPO beta should move in increments of 0.01 or smaller, not the 0.02-0.04 steps typical in hyperparameter sweeps. Beta adjustments below 0.1 were not tested but the 0.12 catastrophe suggests asymmetric risk — beta is more dangerous to increase than decrease.

3. **Data mixture (GT-focused oversampling paradoxically hurts GT).** The `gt_focus_baseline` manifest (2× oversampling of ground_truth and verified_response examples) improved guard categories slightly but *reduced* GT from 7.79 to 7.00. This parallels the knowledge-voice decoupling finding (Section Key Finding #1): over-representing one signal dimension dilutes the complementary signal. Factual grounding may depend on voice consistency as much as on factual content in the training pairs — the voice carries the authority that the judge scores as "ground truth."

4. **Curriculum ordering (no benefit, potential harm).** Placing source-grounded examples first in training order (`source_first`) was neutral on GT (7.57 vs 7.79, within noise) but collapsed private_voice (-4.37 delta). Simple shuffle remains optimal. Curriculum effects at this dataset scale (1,498 pairs) are dominated by eval noise.

**Eval infrastructure issues identified:**

- **Phantom position_discrimination regression.** The 14-prompt `probe-prompts.jsonl` contains zero PD prompts, causing every run to show -9.25 PD regression (baseline 9.25 → 0.0). This makes `constraint_ok` structurally impossible regardless of actual model quality. Fix required: add PD-category prompts to the probe set.

- **Eval variance dominates small effects.** Individual prompt scores swing 3-8 points between runs on the 14-prompt probe. At this noise level, hyperparameter effects smaller than ~0.5 GT are invisible. Ensemble-averaging 2-3 eval runs per config would reduce variance below the signal threshold but at 3× compute cost.

- **300-step probes cannot reach production baselines.** All probes show negative deltas vs the 861-step production scores (8.97 overall, 8.85 GT). The acceptance framework must compare probe-vs-probe, not probe-vs-production.

**Compute cost:** ~$40 Modal (8 runs × ~$5/run for A100-80GB training + eval).

**Conclusion:** The R2 production recipe is well-optimized for ground_truth at the hyperparameter level. Further GT improvement is unlikely to come from recipe tuning. The remaining avenues are: (a) higher-quality training data with richer source grounding, (b) increased dataset size with maintained quality, or (c) longer training runs if the 300-step probe pattern doesn't hold at full scale. This is a clean negative result — the search space was systematically explored and the null hypothesis (baseline is optimal) was not rejected.

**Artifacts:**
- Session report: `experiments/autoresearch/docs/SESSION_REPORT_20260405.md`
- Progress log: `experiments/autoresearch/runs/progress.log`
- Results TSV: `experiments/autoresearch/results.tsv`
- Run directories: `experiments/autoresearch/runs/probe-20260405-*` and `runs/probe-20260406-*`

---

## Key Findings

### 1. Knowledge-Voice Decoupling

Preference training transfers factual knowledge before voice register. With 475 pairs: knowledge score 6.4/10, voice score 1.4/10. Voice required 2.7× more targeted data to imprint. Mechanistically, content (which varies across pairs) dominates gradient updates, while voice (which is the same contrast repeated) accumulates insufficient gradient mass. Resolved by voice-targeted augmentation: 62.7% voice-pair composition in v4 closed the gap.

### 2. Infrastructure Confound / LoRA Quantization Fragility

Same Gemma 3 v4 model scored 8.52 on Modal A100 BF16 vs 1.74 on Ollama GGUF Q4_K_M — a 4.9× degradation from inference infrastructure alone. Rank 16 LoRA deltas are noise-floored by 4-bit quantization. Rank 64 on Qwen 3-32B provides thicker deltas that should better survive quantization. GGUF Q5_K_M testing pending.

### 3. Adapter-on-Base vs Merged Model Serving

Merged model produces 97% character breaks on identity-sensitive prompts; adapter-on-base serving produces 0% breaks on the same prompts. Merging bakes the LoRA signal into the weight distribution where it interacts with RLHF safety attractors. Adapter-on-base preserves the signal at full precision. Implication: the standard deploy pipeline (train → merge → quantize → serve) may systematically destroy voice signal.

### 4. RLHF Safety vs Persona Topology

The base model's safety training overpowers character fine-tuning on specific topic categories — identity (97% break), moral complexity (83% break), meta-self-description (55% break) — while leaving other topics virtually unaffected (0-6% break). This reveals discoverable structure in where safety alignment is strongest vs weakest.

### 5. Post-ORPO SFT Is Catastrophically Destructive

ORPO's monolithic loss structure means SFT after ORPO destroys character signal. Confirmed across two attempts with different ranks and learning rates. The Maiya/Lambert two-stage pattern (DPO → SFT) does not transfer to ORPO due to structural differences in how the objectives encode preferences. Abandoned entirely.

### 6. Learning Rate Sensitivity

lr=2e-5 optimal for character imprinting on Qwen 3-32B. Lower LRs disproportionately sacrifice factual grounding while voice categories are robust. Contradicts ORPO paper's recommended lr=8e-6. Inverse sensitivity: voice needs more data but is LR-robust; knowledge needs less data but is LR-sensitive.

### 7. Base Model Architecture Matters

Qwen 3-32B (pure ForCausalLM) outperforms Gemma 3 27B on character imprinting (+0.29 overall, +1.5 CC, +1.6 PV) while eliminating all VLM infrastructure issues. Lambert's finding that Qwen resists personality modification was specific to Qwen 2.5 and does not apply to Qwen 3.

### 8. Source-Enriched Data Breaks Data Bottlenecks

verified_response was stuck at 7.8/10 across all models, LRs, and dataset sizes — data-bottlenecked. Enriching training pairs with Madison's actual primary source text (225 new pairs) broke through to 8.53/10. The improvement came from content quality, not volume.

### 9. Production Recipe Is Near-Optimal (Autoresearch Negative Result)

Systematic automated search across learning rate (1.8e-5 to 2.2e-5), ORPO beta (0.1 to 0.12), data mixture (GT-focused oversampling), and curriculum ordering (shuffle vs source_first) found no single-parameter change that improves ground_truth over the production baseline (lr=2e-5, beta=0.1, rank 64, shuffle). The search revealed two practically important sensitivities: (a) ORPO beta has a narrow safe band around 0.1 — a 20% increase to 0.12 catastrophically destroys private_voice and verified_response with critical 1.0 scores; (b) learning rate sits at a local optimum where deviation in either direction degrades GT symmetrically. GT-focused data oversampling paradoxically *reduced* GT, suggesting that factual grounding depends on voice consistency as a carrier signal, not just factual content volume. Future GT improvement must come from data quality rather than recipe tuning.

---

## Judge Pipeline Evolution

### Phase 1: Missing overall_score (v3b, v4)
Judge intermittently omitted `overall_score` field. Extraction code defaulted to 0.0. Fix: compute arithmetic mean of component scores as fallback.

### Phase 2: Weighted average override (Qwen 3 runs)
Audit of 108 scored responses across v1, v2, R2 revealed systematic judge bias: -0.2 to -0.4 penalties when critical failures were present, beyond the rubric. Fix: `compute_weighted_overall()` computes the weighted average deterministically. Judge's original preserved as `judge_overall_score`.

### Phase 3: JSON parse repair (R2)
Judge produced malformed JSON (missing commas between object entries). `_repair_json()` regex fixes common patterns. `extract_json()` tries multiple extraction strategies: code block parsing, brace-matching, and whole-text, each with and without repair.

### Phase 4: max_tokens increase
Long model responses generate long judge justifications that exceeded the 2,048 max_tokens limit. Increased to 4,096 for re-judging parse failures.

---

## Eval Infrastructure

| Component | Details |
|-----------|---------|
| Judge model | Claude Sonnet 4.6 (`claude-4-sonnet-20250514`) |
| Judge temperature | 0.0 (deterministic) |
| Judge cost | ~$0.50 per 36-prompt eval (with prompt caching) |
| Eval generation | vLLM with LoRA serving on Modal A100-80GB |
| Eval generation params | temp=1.0, top_k=64, top_p=0.95, max_tokens=1024 |
| Eval prompts | 36 across 6 categories |
| Scoring script | `scripts/data/judge_responses.py` (prompt caching) |
| Scoring library | `src/foundry/press/evaluate.py` |
| Results directory | `data/eval/results/` |
| Scoring methodology | [docs/scoring-methodology.md](scoring-methodology.md) |

---

## Cost Summary

| Run | Training | Eval | Data Gen | Total |
|-----|:--------:|:----:|:--------:|:-----:|
| ORPO v3b | ~$5 | ~$0.50 | $0 | ~$6 |
| ORPO v4 | ~$10 | ~$0.50 | ~$6.15 | ~$17 |
| Qwen 3 v1 | ~$15 | ~$0.50 | $0 | ~$16 |
| Qwen 3 v2 | ~$15 | ~$0.50 | $0 | ~$16 |
| LR sweep (v3+v4) | ~$30 | ~$1.00 | $0 | ~$31 |
| SFT v1+v2 | ~$10 | ~$1.00 | $0 | ~$11 |
| **R2** | **~$8** | **~$0.50** | **~$4.05** | **~$13** |
| GGUF conversion | ~$5 | — | — | ~$5 |
| **Autoresearch (8 runs)** | **~$35** | **~$5** | **$0** | **~$40** |
| **Cumulative** | | | | **~$155** |
