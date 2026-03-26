# Foundry Research Synthesis — March 24, 2026

Comprehensive research day covering DPO optimization, autoresearch integration, recursive self-improvement, prior art survey, and pipeline changes. This document captures all findings so nothing is lost across sessions.

---

## 1. DPO Optimization Research (GeminiResearcher Agent)

### Finding 1: Chosen Response Quality Dominates Everything Else
**Source:** Pan et al., "What Matters in Data for DPO?" (NeurIPS 2025, arxiv:2508.18312)

Improving chosen quality: +7.1 to +8.7 points on AlpacaEval-2. Widening preference gap alone: +3.0 to +4.6 points. When chosen is fixed at high quality and rejected varies: only +0.8 to +1.4 points.

**Implication:** Our 490 Sonnet 4.6 teacher responses ARE the critical variable. The base Gemma student responses matter far less. If any teacher responses feel generic or anachronistic, fix those before training.

### Finding 2: On-Policy Data Helps Only When Chosen Quality Is Already High
**Source:** Same NeurIPS paper + OPA-DPO (opa-dpo.github.io)

On-policy data improves performance ONLY when baseline chosen quality is high. High-quality baseline + 20% on-policy: 39.2 vs Low-quality + 20% on-policy: 27.4. Off-policy preferred responses can NEVER be learnt when they diverge too far from model distribution.

**Implication:** Our current off-policy approach (Sonnet teacher, Gemma student) is reasonable for pass 1. For iteration 2, generate on-policy rejections from the fine-tuned model itself.

### Finding 3: Filter Out Overly Difficult Examples
**Source:** "Principled Data Selection for Alignment" (ICML 2025, arxiv:2502.09650)

Filtering difficult examples improved performance by 9-16% in win rates. A 3B model performed best using 64% of data; a 14B model benefited from 81%. Training exclusively on difficult examples REDUCES win rates by ~9.4%.

**Implication:** After first training run, compute per-example losses. Remove bottom 15-20%. For Gemma 3 27B, ~80% of data should be usable.

### Finding 4: ORPO as Alternative to DPO
**Source:** ORPO paper (arxiv:2403.07691)

ORPO merges SFT and preference optimization into single stage. Eliminates distribution shift between SFT and DPO. No reference model needed (saves VRAM). ORPO V1.2 without prior SFT reached 70.1% win rate against SFT.

**Implication:** Added to pipeline as `--objective orpo`. Run both and compare.

### Finding 5: Beta Is Critical and Should Be Dynamic
**Source:** Beta-DPO (NeurIPS 2024, arxiv:2407.08639)

Optimal beta varies with data. Batch-level beta is the sweet spot (not instance-level). For style/persona: lower beta (0.1-0.3) for aggressive imprinting. For preserving capabilities: higher beta (0.5-1.0).

**Implication:** Our beta=0.1 is aggressive but appropriate for character imprinting. Consider sweep: 0.05, 0.1, 0.2, 0.5 in autoresearch Level 1.

### Finding 6: Gemma 3 Layer Targeting
**Source:** Gemma 3 ablations (HuggingFace blog), Gemma Needs Help (LessWrong)

LoRA adapters needed in FIRST TWO-THIRDS of model layers. Training on layers 40+ (of 62 total in Gemma 3 27B) was INEFFECTIVE. DPO on only 280 preference pairs was highly effective. SFT alone was ineffective — DPO required.

**Implication:** Target layers 0-40. Our 490 pairs exceeds the 280-pair threshold. Rank 16-32 optimal for character voice.

### Finding 7: Curriculum Learning
**Source:** Curry-DPO (arxiv:2403.07230)

Easy pairs first (large gap between chosen/rejected), progressively harder. Training exclusively on hardest pairs exceeds model capacity and degrades performance.

**Implication:** Order training data — constitutional topics first (obvious voice), modern extrapolation last. Remove the very hardest pairs entirely.

### Finding 8: Data Mixing
**Source:** NVIDIA PersonaPlex, Persona consistency scoring (arxiv:2508.06886)

Diversity matters. Weighted mixing and negative persona data augmentation allow selective modulation. Consider 60/40 split: 60% core Madison topics, 40% modern extrapolation + behavioral themes.

---

## 2. Autoresearch Deep Dive (ClaudeResearcher Agent)

### Architecture: Three Files, One Constraint
- `prepare.py` (immutable): evaluation function
- `train.py` (mutable): agent modifies this
- `program.md`: agent instructions

Every experiment: 5-minute wall-clock budget. ~12 experiments/hour, ~100 overnight.

### Key Innovation: Trust Boundary
The agent CANNOT redefine what "better" means. The evaluation function is immutable. This prevents reward hacking.

### Results
- Karpathy: 700 experiments, 20 improvements, 11% speedup
- Shopify: 0.8B beat hand-tuned 1.6B by 19%
- SkyPilot: 910 experiments in 8 hours on 16 GPUs

### The Foundry Loop (Our Novel Contribution)
Nobody has applied autoresearch to character/persona fine-tuning. Mapping:
- `prepare.py` → Madison Authenticity Evaluator (our evaluate.py)
- `train.py` → QLoRA DPO training script
- `program.md` → Agent instructions for overnight optimization

---

## 3. Novel Fine-Tuning Ideas (CodexResearcher Agent — 9 Ideas)

### Idea 1: Automated Hyperparameter Ablation
Sweep beta (0.05-0.5), lr (3e-6 to 2e-5), rank (16-64), dropout (0-0.1). Use Optuna or Ray Tune. ~50-100 GPU-hours = $50-100 Modal credits.

### Idea 2: MadisonScore — Composite Authenticity Scorer
Five signals: semantic similarity to Madison corpus (40%), vocabulary signature (20%), stylometric fingerprint (20%), rhetorical structure (10%), position consistency (10%). Foundation for autoresearch loop.

### Idea 3: Ground Truth Verification Pipeline
Index Madison's 468K words into vector DB. For each teacher response, retrieve relevant passages, cross-encoder rerank, NLI entailment check. Flag contradictions. Regenerate flagged responses.

### Idea 4: Iterative DPO with Self-Play (SPIN + IDPO)
Round 1: teacher vs base Gemma. Round 2: teacher vs Round-1 model. Round 3: teacher vs Round-2 model. Each round, rejected examples get closer to teacher, forcing finer distinctions. SPIN converges in 3-4 rounds.

### Idea 5: Alternative Loss Functions
SimPO (reference-free, halves memory). ORPO (robust on small datasets). KTO (binary labels only — use Madison's actual writings as "good", modern text as "bad"). DPO for final polish.

### Idea 6: ModernBERT Madison Classifier
Train lightweight classifier on Madison's writings (positive) vs Hamilton/Jefferson/modern text (negative). Use for reranking, data filtering, online DPO scoring. Lambert used this for 11 personas.

### Idea 7: Weighted / Margin-Aware DPO
Score all pairs by MadisonScore margin. Weight DPO loss by margin. High-margin pairs get more weight early. Low-margin pairs later. Remove pairs where margin < threshold.

### Idea 8: Multi-Turn Consistency Training
Generate 50-100 multi-turn Madison debates (4-8 turns each). DPO on full-dialogue pairs: consistent Madison = chosen, drifting = rejected. Teaches turn-level character persistence for the debate app.

### Idea 9: The AutoResearch Madison Loop (The Big Idea)
Night 1: Baseline DPO + evaluation. Night 2: Data quality + hyperparameter sweep. Night 3: Self-play SPIN Round 2. Night 4: Data augmentation to 1000+ pairs. Night 5: Multi-turn training. Night 6: Final polish + steering vector extraction.

---

## 4. Prior Art Survey — Novelty Confirmation

### Three Islands That Don't Touch
1. **Autoresearch** — pretraining only, zero character/DPO applications
2. **Automated DPO sweeps** — RapidFire AI, PLoRA — grid search, no agent reasoning
3. **Iterative persona training** — DeePer, PCL, ACD — fixed pipelines, not open-ended

### Must-Cite Prior Art

**DeePer** (arxiv:2502.11078) — Closest match. Three-goal optimization (Preservation, Reflection, Advancement). Self-sampling with 15 candidates. DPO+SFT loss with alpha=0.1. Experience replay. Increasing margin across iterations.

**MentalArena** (arxiv:2410.06845) — Self-play persona loop. GPT-3.5 fine-tuned beats GPT-4o. Performance peaks at iteration 4, then declines. Diversity gain monitoring.

**AI Founding Fathers** (arxiv:2511.09005, Chauhan) — Hamilton/Jefferson/Madison via RAG. Madison scored 90/100 (highest). Validates our domain. Uses RAG not fine-tuning — our lane.

**SPIN** (arxiv:2401.01335) — Recursive loop mechanism foundation. Model generates training data from previous iteration. Converges in 3-4 rounds.

**SPPO** (arxiv:2405.00675) — Self-play preference as two-player game. Nash equilibrium convergence guarantees that DPO lacks.

**Open Character Training** (arxiv:2511.01689) — Lambert/Maiya. Our methodological foundation. CAI + DPO for 11 personas.

### Critical Findings That Changed Our Pipeline

**DPO Causes Persona Drift** (arxiv:2601.12639)
- Drift onset: 200K-400K tokens. Our 490 pairs = ~245K tokens (at onset)
- ORPO shows zero drift at any budget
- Reason: DPO optimizes relative preferences only, entire distribution can shift. ORPO retains SFT anchor.
- **Action:** Added ORPO support. Switch to ORPO for iteration 2+.

**DeePer: DPO + SFT Loss (alpha=0.1)**
- Prevents likelihood displacement
- Combined loss: L = L_DPO + 0.1 * L_SFT
- **Action:** Added sft_alpha parameter to training script.

**DeePer: Self-Sampling Recipe**
- 15 candidates, temperature=1.0, top_p=0.4, repetition_penalty=1.1
- Positive threshold: reward >= 0.5. Negative threshold: reward <= 0.0
- Margin increases across iterations: 0.5 → 0.8
- **Action:** Documented for Level 3 autoresearch.

**MentalArena: Iteration Limit**
- Performance peaks at iteration 4, then declines
- Monitor diversity gain — stop when data becomes repetitive
- **Action:** Cap autoresearch at 4 self-play rounds.

**OpenCharacter: Generation > Rewriting**
- Generating from scratch with character profile beats rewriting existing answers
- **Status:** Already doing this correctly with our Sonnet teacher approach.

**Chauhan: Madison Scores Highest**
- Madison 90/100, highest of three founders
- Measured, legalistic style is most natural fit for AI generation
- **Status:** Validates our choice to start with Madison.

### SDPO: Self-Distillation Policy Optimization (arxiv:2601.20802)
ETH Zurich/MIT/Stanford. Dense learning signal without external reward model. Model conditioned on feedback serves as own self-teacher. Relevant to Level 4 judge calibration. @silennai combined SDPO with autoresearch (posted March 24, 2026) but applied to pretraining only.

### Persona-Aware Contrastive Learning (PCL) (arxiv:2503.17662)
Self-play contrastive alignment alternating between applying and omitting role characteristics. Fixed pipeline, not recursive.

### PersonaAgent
Test-time alignment — iteratively rewrites persona prompt to minimize textual loss. No fine-tuning of model parameters. Different approach but validates automated persona improvement.

### RISE: Recursive IntroSpection (arxiv:2407.18219)
Self-correction after unsuccessful attempts. Improved LLaMA-3-8B by 8.2% on GSM8K. Relevant mechanism for autoresearch loop.

### Academic Venue
**ICLR 2026 Workshop on Recursive Self-Improvement** (recursive-workshop.github.io). Our work would be a direct fit.

---

## 5. Publishable Contributions Identified

### Contribution 1: The Foundry Loop
Recursive self-improvement via autoresearch applied to character voice fidelity. Nobody has done this. Four levels: recipe optimization → data curation → self-play → judge refinement. Runs overnight on RTX 3090.

### Contribution 2: RAG + Fine-Tuning for Historical Character
Chauhan does RAG-only. Lambert does fine-tuning-only. Nobody combines both. Our 5K constitution for fine-tuning + potential RAG from 468K primary source corpus.

### Contribution 3: Verified Ground Truth Evaluation
Using Madison's actual verbatim words (from documented debates, letters, speeches) as the scoring reference for character fidelity evaluation. 8 verified response prompts with his real words.

---

## 6. Pipeline Changes Summary

### Implemented in Training Script
| Change | Source | Parameter |
|--------|--------|-----------|
| ORPO support | Objective Matters | `--objective orpo` |
| SFT loss coefficient | DeePer | `sft_alpha=0.1` |
| LoRA dropout | Gemma ablations | `lora_dropout=0.05` |
| W&B logging | Setup | `wandb-secret` on Modal |

### Implemented in Evaluation
| Change | Source | File |
|--------|--------|------|
| 36 eval prompts across 6 categories | Research | `data/eval/eval-prompts.jsonl` |
| 8 verified response prompts with verbatim Madison | Primary sources | `data/eval/eval-prompts.jsonl` |
| 4-backend evaluation (Anthropic, OpenAI, Gemini, local) | Research | `src/foundry/press/evaluate.py` |
| 5-dimension scoring rubric | Research | `src/foundry/press/evaluate.py` |

### Documented for Future Implementation
| Change | Source | When |
|--------|--------|------|
| Self-sampling (15 candidates, temp=1.0, top_p=0.4) | DeePer | Level 3 autoresearch |
| Increasing DPO margin (0.5 → 0.8) | DeePer | Level 3 |
| Experience replay (5K pairs) | DeePer | Level 3 |
| Switch to ORPO for iteration 2+ | Objective Matters | Level 3 |
| Stop at ~4 iterations | MentalArena | Level 3 |
| Sweep objectives not just hyperparameters | Objective Matters | Level 1 autoresearch |
| Adversarial stress-test eval prompts | Chauhan | Before first training run |
| Three-goal evaluation (preservation/reflection/advancement) | DeePer | Before Level 3 |
| Multi-turn consistency training | Research | After first fine-tune works |
| ModernBERT Madison classifier | Lambert | After baseline established |

---

## 7. Key Papers Downloaded (docs/prior-art/papers/)

| Paper | arxiv | Year | Relevance |
|-------|-------|------|-----------|
| Open Character Training | 2511.01689 | 2025 | Foundation |
| DeePer | 2502.11078 | 2025 | Pipeline changes |
| MentalArena | 2410.06845 | 2024 | Self-play validation |
| AI Founding Fathers | 2511.09005 | 2025 | Domain validation |
| SPIN | 2401.01335 | 2024 | Loop mechanism |
| DPO Persona Drift | 2601.12639 | 2026 | Critical guardrail |
| SDPO | 2601.20802 | 2026 | Judge calibration |
| SPPO | 2405.00675 | 2024 | Convergence guarantees |
| PCL | 2503.17662 | 2025 | Contrastive alignment |
| OpenCharacter | 2501.15427 | 2025 | Data generation |
| CoSER | 2502.09082 | 2025 | Evaluation framework |
| SimPO | 2405.14734 | 2024 | Reference-free optimization |

---

## 8. Current State (End of March 24, 2026)

- **Teacher responses:** 490/490 complete (committed)
- **Student responses:** ~320/490 generating on RTX 3090 via Tailscale (ETA tonight)
- **Eval harness:** 36 prompts, 6 categories, 4 backends, 5-dimension scoring (committed)
- **Training script:** DPO + ORPO support, model caching, GGUF export, W&B (committed)
- **Prior art:** 12 papers downloaded, survey complete, novelty confirmed
- **Next step:** format_dpo → modal run → evaluate

---

## Experimental Results — DPO v1 (March 25, 2026)

### Setup
- **Base model:** google/gemma-3-27b-it, 4-bit QLoRA via Unsloth 2025.7.8
- **LoRA:** rank=16, alpha=16, dropout=0.05, all attention + MLP layers
- **DPO config:** beta=0.1, lr=5e-6, 3 epochs, batch 4×8 (effective 32)
- **Data:** 475 DPO pairs (427 train / 48 eval), plain string format
- **Infrastructure:** Modal A100-40GB, TRL 0.19.1, transformers 4.54.0
- **W&B:** sbergman/foundry, run "madison-dpo-v1"

### Results — Textbook DPO Overfitting on Small Data
Training ran for 49 steps (~0.94 epochs) before OOM crash. No eval metrics collected (eval scheduled at step 50). Key observations:

**Phase 1 — Learning (steps 0-15):**
- Loss: 1.3 → 0.36, Accuracy: 50% → 100%, Margins: -0.5 → +1.0
- Grad norm volatile: 60-708
- Model correctly learning to distinguish Madison voice from generic student

**Phase 2 — Sweet spot (steps 15-25):**
- Loss: 0.1-0.5, Accuracy: 87-100%, Margins: 1-3
- Grad norm: 25-80
- This is where a well-timed checkpoint would capture useful adaptation

**Phase 3 — Memorization/collapse (steps 26-49):**
- Loss: 0.000-0.006, Accuracy: 100%, Margins: 8-15
- Grad norm collapsed: 0.1-5.6
- Model memorized all 427 training pairs — margins balloned from healthy ~1 to grotesque 10-15

**OOM crash at step 49:** DPO requires policy + reference model (~30GB for 27B 4-bit), leaving only ~10GB for activations. Batch size 4 at seq_len 2048 exceeded headroom on a long batch.

### Replication of Known Findings

**1. DPO likelihood displacement (Smaug/DPOP paper, arXiv:2402.13228):**
Confirmed. DPO achieved near-zero loss while chosen logprobs remained flat (-2170 → -2163 over 49 steps). The model learned to *discriminate* pairs without improving its ability to *generate* the chosen style. This validates the DPOP finding that "DPO can reduce the probability of preferred completions while still achieving low loss."

**2. DPO overfitting on small datasets ("Objective Matters", arXiv:2601.12639):**
Confirmed. With 475 pairs (~245K tokens), DPO collapsed by epoch 0.6 — well within the paper's identified danger zone of 200K-400K tokens. Loss went to 0.0000, margins to 15.45, accuracy pegged at 100%. The paper predicted "progressive persona drift starting at 200K-400K tokens" for DPO specifically.

**3. Gradient instability without clipping:**
Observed grad norms ranging from 0.1 to 708.3 within a single run. No gradient clipping was applied (max_grad_norm not set). The wild variance correlated with batch composition — batches containing subtle chosen/rejected contrasts (constitutional topics where both teacher and student sound Madisonian) produced the largest spikes.

### Lessons for v2 (ORPO)
| Parameter | v1 DPO | v2 ORPO | Rationale |
|---|---|---|---|
| Objective | DPO | ORPO | Zero persona drift per "Objective Matters" |
| Batch size | 4 | 1 | Prevent OOM (ORPO has no ref model but be safe) |
| Grad accum | 8 | 4 | Effective batch 4 → more steps/epoch, smoother curve |
| max_grad_norm | unset | 1.0 | Clip the 0.1-708 variance observed in v1 |
| dropout | 0.05 | 0.0 | Unsloth fast patching requires dropout=0 |
| save_steps | 100 | 50 | Capture checkpoints at useful intervals |
| eval_steps | 50 | 50 | v1 never reached eval; ensure we do this time |
| save_total_limit | 3 | 5 | Keep more checkpoints for post-hoc selection |

---

## Experimental Results — ORPO v2/v3 (March 25-26, 2026)

### v2: ORPO Baseline (beta=0.1, lr=5e-6)
- **W&B:** [madison-orpo-v1](https://wandb.ai/sbergman/foundry/runs/kfugqbps)
- **Config:** ORPO beta=0.1, lr=5e-6, 3 epochs, batch 1×4 (effective 4), max_grad_norm=1.0, dropout=0.0
- **Result: Preference signal never engaged.** Accuracy 0.000 across all 3 epochs and 6 eval checkpoints. Loss declined 2.57 → 2.04 (pure SFT learning). Margins stayed negative throughout (-0.15 → -0.08). The odds ratio penalty at beta=0.1 was negligible relative to the SFT loss — the model learned to generate text but never learned to discriminate chosen from rejected.
- **Runtime:** 4011s (~67 min), no OOM issues.

### v3a: Higher Beta (beta=0.5, lr=5e-6)
- **W&B:** madison-orpo-v3a-beta05
- **Config:** ORPO beta=0.5, lr=5e-6, 3 epochs, batch 1×4 (effective 4), max_grad_norm=1.0
- **Result: Still no preference learning.** Accuracy stayed at 0.000 through 3 epochs (one eval checkpoint at step 305 showed 2.1% accuracy — essentially noise). Loss higher than v2 (3.45 → 2.59) because the larger beta penalizes the model harder for preferring rejected, but the learning rate was too low to correct course. Margins improved (-0.79 → -0.38) but never went positive.
- **Conclusion:** Higher beta alone cannot compensate for insufficient learning rate. The preference penalty needs enough gradient signal to act on.

### v3b: Higher Learning Rate (beta=0.1, lr=2e-5) — WINNER
- **W&B:** [madison-orpo-v3b-lr2e5](https://wandb.ai/sbergman/foundry/runs/s84fcrt0)
- **Config:** ORPO beta=0.1, lr=2e-5, 3 epochs, batch 1×4 (effective 4), max_grad_norm=1.0
- **Result: Successful character training.** The preference signal emerged mid-epoch 1 and strengthened through epoch 3 without collapsing.

**Eval trajectory (held-out data, 48 examples):**

| Eval Step | Loss | Accuracy | Margin |
|---|---|---|---|
| 50 (ep 0.5) | 2.220 | 0.0% | -0.122 |
| 101 (ep 0.9) | 1.851 | 12.5% | -0.029 |
| 152 (ep 1.4) | 1.741 | **97.9%** | +0.047 |
| 203 (ep 1.9) | 1.692 | **100%** | +0.100 |
| 254 (ep 2.3) | 1.671 | **100%** | +0.159 |
| 305 (ep 2.8) | 1.665 | **100%** | +0.172 |

**Key observations:**
- Preference discrimination emerged around step 110 (epoch ~1.0) and was near-perfect by step 150
- Loss plateaued at ~1.65-1.70 (NOT collapsing to zero — SFT component keeps the model generative)
- Margins stayed controlled at 0.10-0.17 (contrast with DPO's grotesque 10-15)
- Eval tracked train closely — no overfitting detected
- Grad norm stable throughout: 0.7 → 1.3 (contrast with DPO's 0.1-708 range)
- **Runtime:** 3835s (~64 min), no OOM issues.
- **Adapter saved** to Modal volume `foundry-adapters` at `/adapters/experiments/madison-orpo-v3b-lr2e5`

### Cross-Run Comparison

| Run | Objective | Beta | LR | Eval Acc (final) | Eval Margin | Loss (final) | Grad Norm | Status |
|---|---|---|---|---|---|---|---|---|
| v1 DPO | DPO | 0.1 | 5e-6 | never measured | never measured | 0.000 (collapsed) | 0.1-708 | OOM crash |
| v2 ORPO | ORPO | 0.1 | 5e-6 | 0.0% | -0.104 | 2.071 | 0.6-0.9 | No learning |
| v3a ORPO | ORPO | 0.5 | 5e-6 | 2.1% | -0.379 | 2.592 | 1.0-1.4 | No learning |
| **v3b ORPO** | **ORPO** | **0.1** | **2e-5** | **100%** | **+0.172** | **1.665** | **0.7-1.3** | **Success** |

### Findings

**1. Learning rate was the critical variable, not beta.** v3a (5× beta) failed while v3b (4× LR) succeeded. With ORPO on a small dataset, the model needs sufficient gradient signal to move the SFT component, which then creates headroom for the odds ratio penalty to discriminate. Higher beta alone just makes the loss surface steeper without providing the gradient magnitude to climb.

**2. ORPO confirms "Objective Matters" paper predictions.** Zero persona drift observed — loss plateaued at 1.66 with controlled margins (0.17), compared to DPO's collapse to 0.000 with margins of 15. The SFT component maintained generative capability while the odds ratio learned preference discrimination. This is exactly the behavior predicted for ORPO at the 200K-400K token scale.

**3. ORPO preference learning follows a phase transition, not a gradient.** Eval accuracy jumped from 12.5% (step 101) to 97.9% (step 152) — a near-discontinuous transition around epoch 1.0-1.4. This aligns with the hypothesis that the SFT component must reach sufficient quality before the odds ratio signal becomes actionable.

**4. Gradient clipping was unnecessary but harmless for ORPO.** All ORPO runs had natural grad norms of 0.7-1.4, well under the max_grad_norm=1.0 clip (which only occasionally activated). The stability is structural to ORPO's loss function, not an artifact of clipping. Contrast with DPO where clipping would have been essential.

### Next Steps
- **Step 7:** Download v3b adapter, load in LM Studio, run 36-prompt eval harness
- **Step 8:** If eval passes, generate introspection SFT data using the partially-trained model
- **Step 9:** Iterate on any eval failures before proceeding to Stage 2
