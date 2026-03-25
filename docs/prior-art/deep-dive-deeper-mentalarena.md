# Deep Dive: DeePer, MentalArena, and DPO Persona Drift

Research conducted March 24, 2026. Analysis of three papers with direct implications for the Foundry training pipeline.

## Critical Pipeline Changes Identified

### Change 1: Add SFT Loss to DPO (from DeePer) — HIGH PRIORITY
**Current:** Pure DPO loss
**Recommended:** Combined loss: L = L_DPO + 0.1 * L_SFT
**Why:** Prevents "likelihood displacement" where chosen response probability actually decreases during DPO training. The SFT term anchors the model to what good outputs look like, not just what bad outputs look like.
**Effort:** Small — TRL DPOTrainer supports `loss_type` configuration

### Change 2: Run ORPO in Parallel (from Objective Matters) — HIGH PRIORITY
**Current:** DPO only
**Recommended:** Train both DPO and ORPO on same data, compare
**Why:** DPO causes persona drift at 200K-400K tokens. Our 490 pairs (~245K tokens) is right at the onset zone. ORPO shows virtually no drift at any budget.
**Effort:** Small — TRL has ORPOTrainer, same data format
**ORPO beta:** 0.05

### Change 3: Three-Goal Evaluation (from DeePer) — MEDIUM PRIORITY
**Current:** Single MadisonScore
**Recommended:** Score every iteration on three dimensions:
  - **Previous Preservation:** Still sounds like Madison on topics from prior iterations?
  - **Current Reflection:** Fixes the specific weakness targeted?
  - **Future Advancement:** Generalizes to unseen topics?
**Effort:** Medium — modify evaluate.py judge prompt

### Change 4: Self-Sampling Parameters (from DeePer) — FOR LEVEL 3 AUTORESEARCH
**Parameters:** 15 candidates, temperature=1.0, top_p=0.4, repetition_penalty=1.1
**Margin thresholds:** >= 0.5 positive, <= 0.0 negative
**Increasing margin:** Start 0.5, increase to 0.8 in iteration 2+
**Experience replay:** Carry forward 5,000 best pairs per iteration

### Change 5: Iteration Limit (from MentalArena) — FOR LEVEL 3 AUTORESEARCH
**Finding:** Performance peaks at iteration 4, then declines
**Recommendation:** Monitor diversity gain; stop when it plateaus (~4 iterations)
**Switch to ORPO for iterations 2+** to avoid cumulative DPO drift

### Change 6: Sweep Objectives Not Just Hyperparameters — FOR LEVEL 1 AUTORESEARCH
**Finding:** "The objective function matters more than the data at scale"
**Recommendation:** Level 1 autoresearch should sweep across DPO vs ORPO vs KL-reg, not just beta/rank/lr within DPO

## Hyperparameter Comparison

| Parameter | DeePer | Obj. Matters | Our Default | Action |
|-----------|--------|-------------|-------------|--------|
| LR | 5e-6 | 2e-5 | 5e-6 | Keep ours (matches DeePer) |
| LoRA rank | not reported | 16 | 16 | Confirmed |
| LoRA alpha | not reported | 32 | 16 | **Test alpha=32 (2x rank)** |
| LoRA dropout | not reported | 0.05 | 0.0 | **Test 0.05** |
| DPO beta | not reported | 1.0 | 0.1 | Keep 0.1 (aggressive imprinting) |
| ORPO beta | N/A | 0.05 | N/A | Use 0.05 for ORPO runs |
| SFT coeff | 0.1 | N/A | N/A | **Add alpha=0.1** |
| Epochs | 4 | varies | 3 | Keep 3 (conservative) |
| Batch size | 128 | 2 | 1 | Test larger with grad accum |
