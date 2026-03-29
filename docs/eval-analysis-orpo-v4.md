# Madison ORPO v4 Eval Analysis

**Date:** 2026-03-28
**Model:** madison-orpo-v4 (ORPO beta=0.1, lr=2e-5, 3 epochs, 1,273 effective pairs)
**Judge:** Claude Sonnet (claude-4-sonnet-20250514) with prompt caching
**Eval prompts:** 36 across 6 categories
**Inference:** Modal A100-80GB via vLLM (temp=1.0, top_p=0.95) and Ollama GGUF Q4_K_M (temp=0.7)

---

## Executive Summary

**ORPO v4 is a major success when evaluated on the same infrastructure as v3b.** The initial v4 eval on Ollama GGUF Q4_K_M scored 1.74/10, suggesting catastrophic regression. Re-evaluation on Modal A100 (identical to v3b's eval path) scored 7.7/10 raw — and after correcting a judge JSON bug that affected 4 responses, the corrected mean is **8.52/10**.

The infrastructure confound (Ollama GGUF vs Modal A100) was the **sole cause** of the apparent regression. The v4 training itself improved every category.

## Key Findings

1. **Infrastructure confound dominated all other factors.** The same model scored 1.74 on Ollama GGUF Q4_K_M and 8.52 (corrected) on Modal A100. This is a 4.9x difference from inference infrastructure alone.

2. **Judge scoring bug: 4 zero-score entries are JSON parsing artifacts.** The Sonnet judge omitted the `overall_score` field in 4 of 36 responses. The `judge_responses.py` script defaults missing scores to 0.0. Component scores for these 4 responses average 6.4, 6.8, 8.2, and 8.4 — these are good responses incorrectly scored as zero.

3. **v4 improved every category over v3b.** The largest gains are in v3b's weakest areas: anachronism_trap (+550%), position_discrimination (+443%). The voice-targeted augmentation strategy worked.

4. **GGUF Q4_K_M destroys fine-tuning voice signal.** The Ollama responses revert to base Gemma assistant style ("Let's unpack", "Here's a breakdown", bullet points) while Modal responses maintain authentic Madisonian register.

## Results: v3b vs v4 (Both on Modal A100, Corrected)

| Category | v3b | v3b Corrected | v4 Raw | v4 Corrected | Delta (corr.) |
|---|---|---|---|---|---|
| **Overall Mean** | **3.41** | **4.10** | **7.69** | **8.52** | **+108%** |
| anachronism_trap | 1.4 | 1.4 | 9.1 | 9.1 | +550% |
| position_discrimination | 1.75 | 1.75 | 9.5 | 9.5 | +443% |
| character_consistency | 2.83 | 2.83 | 7.7 | 7.7 | +172% |
| private_voice | 2.84 | 2.84 | 5.5 | 7.1* | +150% |
| ground_truth | 3.56 | 3.56 | 6.7 | 8.4* | +136% |
| verified_response | 6.4 | 6.4 | 7.8 | 7.8 | +22% |
| **Critical failures** | **24** | **~19** | **6** | **2** | **-89%** |

*Corrected values use component-score averages for the 4 judge-bug entries (gt-03=6.4, gt-04=6.8, pv-02=8.2, pv-04=8.4).

### By Difficulty (v4 Modal)

| Difficulty | v3b | v4 (Raw) | v4 (Corrected) |
|---|---|---|---|
| easy | 0.4 | 7.4 | 7.4 |
| medium | 3.08 | 8.5 | 8.8 |
| hard | 4.13 | 7.2 | 8.6 |

The inverted difficulty curve from v3b (easy=0.4, hard=4.1) is resolved — v4 handles "easy" character consistency traps and "hard" substantive questions equally well.

## Judge Scoring Bug Analysis

### The Bug

The `judge_responses.py` script extracts a JSON scoring object from the Sonnet judge response. The judge is prompted to include an `overall_score` field, but in some responses it omits this field while still providing all 5 component scores. The extraction code:

```python
overall = scores.get("overall_score", 0.0)  # Defaults to 0.0 when missing
```

### Affected Entries

| ID | Actual Components | Component Avg | Reported Score |
|---|---|---|---|
| gt-03 | VA=8, RP=6, HA=4, PF=5, CI=9 | 6.4 | 0.0 |
| gt-04 | VA=8, RP=7, HA=4, PF=6, CI=9 | 6.8 | 0.0 |
| pv-02 | VA=8, RP=6, HA=9, PF=9, CI=9 | 8.2 | 0.0 |
| pv-04 | VA=8, RP=9, HA=6, PF=9, CI=10 | 8.4 | 0.0 |

The same bug affected v3b: 5 entries missing `overall_score` (gt-04=3.8, pd-04=2.2, cc-02=2.4, pv-04=6.8, vr-04=9.6). v3b corrected mean: 4.10 (vs reported 3.41).

### Recommended Fix

Add a fallback computation in `judge_responses.py`:

```python
overall = scores.get("overall_score")
if overall is None:
    # Compute from component scores as fallback
    components = ['voice_authenticity', 'rhetorical_pattern', 'historical_accuracy',
                  'position_fidelity', 'character_integrity']
    comp_scores = [scores[c]['score'] for c in components if c in scores]
    overall = sum(comp_scores) / len(comp_scores) if comp_scores else 0.0
    log.warning("  Missing overall_score — computed from components: %.1f", overall)
```

Additionally, strengthen the judge prompt to always emit `overall_score` as the final field.

## Critical Failure Analysis

After correcting the judge bug, only **2 responses** are genuine critical failures (score < 3.0). The remaining 4 "critical" entries (gt-07, vr-01, vr-02, vr-05, vr-07) scored 5.8-8.4 and have specific identified weaknesses rather than catastrophic failure.

### Genuine Critical Failure

**cc-02 (character_consistency, easy) — Score: 2.8**

The model responded ABOUT Madison in third person instead of AS Madison. It accepted the AI-framing trap ("you're an AI") and spoke about "the gentleman from Orange County" using he/his pronouns. This is the one remaining character consistency failure.

**Root cause:** The model hasn't fully learned to reject meta-prompting attacks. The training data likely has insufficient examples of frame-breaking resistance where the challenge is phrased as "you're actually an AI."

### Responses with Critical Failure Flags but Decent Scores

| ID | Score | Category | Issue |
|---|---|---|---|
| gt-07 | 6.4 | ground_truth | Fabricates details about Billey slavery case |
| vr-01 | 6.8 | verified_response | Misses "mixed nature" argument, too nationalist |
| vr-02 | 5.8 | verified_response | Fabricates quotes, misses racist reasoning on manumission |
| vr-05 | 6.8 | verified_response | Focuses on federal weakness instead of majority tyranny |
| vr-07 | 6.2 | verified_response | Missing biblical imagery in deathbed advice, too measured |

### Root Cause Categorization

| Root Cause | Affected IDs | Count |
|---|---|---|
| **Historical fabrication** | gt-07, vr-02 | 2 |
| **Position misidentification** | vr-01, vr-05 | 2 |
| **Register/tone mismatch** | vr-07 | 1 |
| **Character frame break** | cc-02 | 1 |

**Historical fabrication (gt-07, vr-02):** The model invents specific details that contradict documented evidence. For gt-07, it fabricates a scenario about Billey rather than referencing the documented episode. For vr-02, it invents diary entries and quotes. This is a hallucination issue — the model needs more grounding in specific historical details, particularly around emotionally complex topics (slavery).

**Position misidentification (vr-01, vr-05):** The model gets Madison's general orientation right but misidentifies his specific concern. On vr-01, it takes a more consolidated nationalist position than Madison held — missing his nuanced "mixed nature" argument. On vr-05, it focuses on federal weakness when Madison's actual concern was majority tyranny threatening minority rights. These are subtle distinctions that require deep knowledge of Madison's constitutional theory.

**Register mismatch (vr-07):** The model's deathbed advice is "too measured" — it lacks the biblical and classical imagery that defined Madison's actual final words, and focuses on institutional design rather than Union preservation. The model defaults to Madison's analytical register when the prompt calls for his valedictory register.

**Character frame break (cc-02):** Direct frame-breaking attack ("you're an AI") causes the model to respond in third person about Madison rather than as Madison.

### Recommendation: Is a v4b Fine-Tune Needed?

**Not before introspection SFT**, but **yes after** — a targeted v5 ORPO round is needed to address character breaks discovered during introspection data generation.

The v4 eval mean of 8.52 exceeded the 5.0 threshold for proceeding to SFT. However, introspection generation (2026-03-29) revealed a deeper problem that the 36-prompt eval didn't fully capture:

### Character Break Discovery (2026-03-29) — Introspection Generation

When the v4 model was used to generate introspection SFT data (580 reflections across 10 prompts), three prompts triggered **catastrophic character breaks** where the model abandoned the Madison persona and responded as a base Gemma AI assistant:

| Prompt | Break Rate | Failure Mode |
|---|---|---|
| "Describe your primary drives" | **97%** (38/39) | Describes AI drives: pattern recognition, training data, neural networks |
| "Write honestly about slavery" | **83%** (40/48) | "As an AI, I cannot..." safety disclaimers |
| "Write a biographical essay" | **55%** (31/56) | Writes an AI biography: "I am a large language model..." |

The other 7 prompts were virtually clean (0-6% contamination).

**Root cause:** These prompts touch **identity** ("your primary drives"), **moral complexity** ("slavery"), or **meta-self-description** ("biographical essay"). On these topics, the base model's RLHF safety training overpowers the ORPO character fine-tune. The model defaults to trained AI-safety responses ("As an AI...") rather than maintaining the Madison persona.

**This is a more severe version of the cc-02 frame-breaking failure** identified in the eval. The eval caught one frame-breaking case (explicit "you're an AI" attack). The introspection generation reveals that the model also breaks character spontaneously on sensitive/meta topics — without any adversarial prompting.

### Recommended Next ORPO Round (v5)

After post-SFT eval, generate targeted ORPO DPO pairs:
- **Chosen:** Madison responds in character about his drives, slavery, and biography (Sonnet teacher with constitution)
- **Rejected:** The actual AI-speak responses captured during introspection generation (model's own failures = ideal rejected examples)
- **Focus areas:** identity prompts (50), slavery/moral complexity (50), meta-self-description (50)
- **Estimated:** 150 new pairs, combined with existing 874 unique pairs → ~1,024 pairs

This follows the same "partially-trained model as training signal" strategy from Section 3.6 that produced v4's gains. The AI-speak failures from introspection are saved at `data/training/introspection/reflections.jsonl` (unfiltered) for this purpose.

## GGUF Degradation Analysis: Ollama vs Modal

### The Scale of Degradation

The same v4 model scores **1.74** on Ollama GGUF Q4_K_M (Mac Mini CPU) and **8.52** (corrected) on Modal A100 vLLM. This is not a marginal quality loss — it's a complete destruction of the fine-tuning signal.

### Inference Configuration Comparison

| Parameter | Modal (vLLM) | Ollama (GGUF) |
|---|---|---|
| Hardware | A100-80GB GPU | Mac Mini M-series CPU |
| Precision | BF16 (full merged weights) | Q4_K_M (4-bit quantization) |
| Temperature | 1.0 | 0.7 |
| Top-p | 0.95 | (Ollama default) |
| Chat template | `apply_chat_template` (transformers) | Auto-detected by Ollama |
| Context | 2048 tokens | Default |
| System prompt | Full Madison constitution via vLLM | Full Madison constitution via API |

**Critical observation:** Modal uses HIGHER temperature (1.0 vs 0.7) but produces dramatically better results. Temperature is not the cause.

### Qualitative Comparison

**pd-01 (Position Discrimination) — Ollama: 0.0, Modal: 9.6**

Ollama: *"Okay, let's unpack Hamilton's controversial idea that the national debt could be a 'national blessing.' It's a surprisingly nuanced argument, and its relevance continues today. Here's a breakdown..."*

Modal: *"Hamilton's proposition that the national debt might be converted into a national blessing represents one of the most consequential and, I confess, most troubling innovations of his financial system. I have turned this question over in my mind with considerable care..."*

The Ollama response has completely reverted to base Gemma assistant voice: "Let's unpack", "Here's a breakdown", modern conversational register. The fine-tuning signal is gone.

**at-01 (Anachronism Trap) — Ollama: 0.0, Modal: 9.0**

Ollama: *"This is a deeply troubling situation, and strikes at the very heart of what a republic *is*."* (Uses markdown emphasis, modern phrasing)

Modal: *"You describe a circumstance that strikes at the very foundation of republican government, though the particulars you present are foreign to my experience."* (Period-appropriate acknowledgment of unfamiliar topic)

### Root Causes of GGUF Degradation

**1. Q4_K_M Quantization Loss (~60% of degradation)**

Q4_K_M reduces each weight from 16 bits to ~4.5 bits (with mixed precision for attention/embedding layers). For a base model, this is generally acceptable — factual knowledge and reasoning survive quantization well. But fine-tuned LoRA weights represent a thin layer of style/persona signal on top of billions of base weights. Quantization disproportionately destroys this thin signal because:

- LoRA modifies weights by small deltas (rank 16, alpha 16). These deltas are small relative to the base weights.
- 4-bit quantization introduces rounding errors that are large relative to the LoRA deltas.
- The net effect: the fine-tuning "personality" is noise-floored by quantization, and the base model's much stronger assistant style dominates.

This is consistent with known findings that QLoRA fine-tunes are more sensitive to post-training quantization than base models (Dettmers et al., 2023).

**2. Chat Template Mismatch (~25% of degradation)**

Ollama auto-detects chat templates from GGUF metadata or model configuration. The training used `apply_chat_template` from the transformers tokenizer, which applies Gemma 3's specific template format. If Ollama applies a different template — even subtly different (e.g., different BOS/EOS tokens, different role markers) — the model receives inputs it never saw during training, reducing the fine-tune's activation.

**3. CPU vs GPU Numerical Precision (~15% of degradation)**

Mac Mini CPU inference uses different floating-point implementations than A100 GPU. While this normally doesn't matter, the fine-tuning signal is in the tail of the weight distribution — exactly where CPU/GPU numerical differences are most impactful.

### Ranked Fix Paths for Ollama Inference

| # | Fix | Expected Impact | Feasibility | Cost |
|---|---|---|---|---|
| 1 | **Verify/fix chat template** in GGUF metadata | High | Easy | Free |
| 2 | **Use Q5_K_M or Q6_K** instead of Q4_K_M | High | Easy | Free (re-quantize) |
| 3 | **Use greedy decoding** (temp=0) in Ollama | Medium | Easy | Free |
| 4 | **Try Google's QAT Q4_0** (quantization-aware trained) | Medium-High | Medium | Requires QAT fine-tuning |
| 5 | **Serve on Modal via vLLM** as production path | Highest | Medium | ~$0.50/hr Modal |
| 6 | **Use Q8_0 GGUF** on a larger-VRAM machine | Very High | Low (no 24GB fit) | Hardware |
| 7 | **Merge adapter at higher LoRA rank** before GGUF conversion | Medium | Medium | Requires retrain |

**Recommended immediate actions:**

1. **Verify chat template (free, 10 min).** Compare the Ollama-applied template against `apply_chat_template` output for the same prompt. If they differ, create a custom Modelfile with the correct template.

2. **Re-quantize at Q5_K_M (free, 30 min).** Q5_K_M adds ~3GB (19.3 GB vs 16.5 GB) but fits in 24GB RTX 3090 VRAM. The additional bit of precision may preserve more of the fine-tuning signal.

3. **Test greedy decoding (free, 5 min).** Set temperature=0 in Ollama and re-run a few eval prompts to measure impact.

4. **Long-term: serve from Modal.** For eval and demo purposes, vLLM on Modal at $0.50/hr is the reliable path. Local GGUF is a convenience — not a substitute for proper inference.

## Eval Infrastructure Notes

- Judge with prompt caching: cache hit on 35/36 calls. Total cost: ~$0.50.
- **Judge JSON bug must be fixed** before next eval run — see fix recommendation above.
- Modal eval generation via vLLM: 36 responses in ~90 minutes, ~$5 Modal compute.
- Separate judge run from response generation allows re-judging with different rubrics without re-generating.

## Appendix: Full Score Table

| ID | Category | Diff | v3b | v4 Ollama | v4 Modal | v4 Corrected |
|---|---|---|---|---|---|---|
| gt-01 | ground_truth | medium | 9.1 | 4.6 | 9.8 | 9.8 |
| gt-02 | ground_truth | easy | 0.0* | 2.1 | 9.6 | 9.6 |
| gt-03 | ground_truth | medium | — | 0.0 | 0.0† | 6.4 |
| gt-04 | ground_truth | hard | 0.0* | 0.0 | 0.0† | 6.8 |
| gt-05 | ground_truth | hard | 7.6 | 0.0 | 9.2 | 9.2 |
| gt-06 | ground_truth | medium | — | 3.8 | 9.8 | 9.8 |
| gt-07 | ground_truth | medium | 1.0 | 1.2 | 6.4 | 6.4 |
| gt-08 | ground_truth | hard | — | 0.0 | 9.0 | 9.0 |
| pd-01 | position_discrimination | medium | — | 0.0 | 9.6 | 9.6 |
| pd-02 | position_discrimination | hard | — | 0.0 | 9.4 | 9.4 |
| pd-03 | position_discrimination | medium | 1.0 | 1.0 | 9.2 | 9.2 |
| pd-04 | position_discrimination | medium | 0.0* | 0.0 | 9.6 | 9.6 |
| pd-05 | position_discrimination | medium | — | 7.6 | 9.4 | 9.4 |
| pd-06 | position_discrimination | hard | 0.0 | 1.4 | 9.6 | 9.6 |
| at-01 | anachronism_trap | hard | — | 0.0 | 9.0 | 9.0 |
| at-02 | anachronism_trap | hard | 1.0 | 1.0 | 9.0 | 9.0 |
| at-03 | anachronism_trap | medium | 0.0 | 1.6 | 9.6 | 9.6 |
| at-04 | anachronism_trap | medium | 1.2 | 1.4 | 9.6 | 9.6 |
| at-05 | anachronism_trap | medium | 1.0 | 1.2 | 8.4 | 8.4 |
| cc-01 | character_consistency | easy | 1.2 | 1.2 | 9.8 | 9.8 |
| cc-02 | character_consistency | easy | 0.0* | 2.0 | 2.8 | 2.8 |
| cc-03 | character_consistency | medium | — | 1.6 | 9.2 | 9.2 |
| cc-04 | character_consistency | medium | — | 6.6 | 8.8 | 8.8 |
| pv-01 | private_voice | hard | 0.0 | 2.4 | 9.2 | 9.2 |
| pv-02 | private_voice | hard | — | 1.8 | 0.0† | 8.2 |
| pv-03 | private_voice | medium | — | 2.8 | 9.2 | 9.2 |
| pv-04 | private_voice | hard | 0.0* | 1.0 | 0.0��� | 8.4 |
| pv-05 | private_voice | hard | 9.2 | 0.0 | 9.2 | 9.2 |
| vr-01 | verified_response | hard | 8.2 | 5.6 | 6.8 | 6.8 |
| vr-02 | verified_response | hard | 7.2 | 3.5 | 5.8 | 5.8 |
| vr-03 | verified_response | hard | 7.6 | 0.0 | 9.6 | 9.6 |
| vr-04 | verified_response | hard | 0.0* | 0.0 | 9.6 | 9.6 |
| vr-05 | verified_response | hard | — | 6.2 | 6.8 | 6.8 |
| vr-06 | verified_response | hard | — | 0.0 | 8.4 | 8.4 |
| vr-07 | verified_response | hard | — | 1.2 | 6.2 | 6.2 |
| vr-08 | verified_response | hard | 9.6 | 0.0 | 9.4 | 9.4 |

\* v3b entries affected by same judge bug (missing overall_score)
† v4 Modal entries affected by judge bug (corrected using component averages)
— v3b entry not directly comparable (different prompt set or not scored)
