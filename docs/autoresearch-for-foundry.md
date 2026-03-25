---
name: Autoresearch for Foundry
status: research
priority: medium
started: 2026-03-24
description: Applying Karpathy's autoresearch pattern to optimize Madison character fine-tuning — automated hyperparameter search, data curation, and iterative DPO refinement.
---

## Karpathy's Autoresearch — Technical Summary

**Repo:** [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch) (54K+ stars, MIT License)
**Released:** March 7-8, 2026

### What It Is

An autonomous AI agent loop that iteratively modifies an LLM training script, runs 5-minute experiments, keeps improvements, discards regressions, and repeats indefinitely. The entire system is three files:

| File | Role | Who Edits |
|------|------|-----------|
| `prepare.py` | Data, tokenizer, evaluation function | Nobody (immutable trust boundary) |
| `train.py` | Model architecture, optimizer, hyperparameters | The AI agent |
| `program.md` | Agent instructions governing behavior | The human |

### The Core Pattern

```
LOOP FOREVER:
  1. Inspect git state
  2. Modify train.py with an experimental idea
  3. git commit
  4. Run training (5-minute wall-clock budget)
  5. Parse validation metric
  6. If improved → keep (advance branch)
  7. If worse → git reset to previous state
  NEVER STOP.
```

### Why It Works

The **trust boundary** is the key innovation: `prepare.py` (evaluation) is immutable, preventing reward hacking. The agent cannot redefine what "better" means. The fixed 5-minute budget makes experiments directly comparable regardless of architectural changes.

### Results

- Karpathy: 700 experiments in 2 days on H100, 20 genuine improvements, 11% training speedup
- Shopify (Tobi Lutke): 0.8B model outperformed hand-tuned 1.6B by 19%. Applied to Liquid templating: 53% faster rendering from 93 automated commits
- SkyPilot: 16-GPU cluster, 910 experiments in 8 hours, agent independently invented two-tier GPU validation
- Community: Extended to robotics (MuJoCo), adversarial hardening, Apple Silicon, multi-agent coordination

---

## Application to Foundry Character Fine-Tuning

### The Novel Idea: "The Foundry Loop"

Nobody has published applying autoresearch to DPO fine-tuning for character training. This is genuinely novel.

**Mapping the pattern to our pipeline:**

```
prepare.py (IMMUTABLE) → Madison Authenticity Evaluator
  - LLM-as-judge using the 5K madison constitution as rubric
  - Scores: voice_authenticity, historical_accuracy, anachronism_count
  - 20-30 held-out test prompts (not in the 490 training set)
  - Fixed eval function — agent cannot game the metric

train.py (MUTABLE — agent modifies) → QLoRA DPO Training Script
  - LoRA config: rank (8-64), alpha, target layers (0-40 of 62), modules
  - DPO hyperparameters: beta (0.05-0.5), learning rate (1e-6 to 1e-4)
  - Data selection: which pairs to include, curriculum ordering, theme weights
  - Training: batch size, gradient accumulation, warmdown schedule
  - Optimizer: AdamW vs alternatives

program.md → Foundry Agent Instructions
  "Run a 5-minute QLoRA DPO training on Modal A100.
   Evaluate via the Madison authenticity judge.
   Keep if judge score improves, discard if not.
   Focus areas: beta, rank, layer targeting, data curriculum.
   Budget: $5-20 total for a full overnight session."
```

### Implementation Roadmap

**Phase 1: Build the Evaluator (prepare.py equivalent)**

The evaluator is the foundation. It must:
1. Load the fine-tuned LoRA adapter onto base Gemma 3 27B
2. Generate responses to 20-30 held-out Madison prompts
3. Score each response using an LLM judge with the madison-5k constitution
4. Return a single numeric score (e.g., average authenticity rating 1-10)

The judge prompt would use the constitution to evaluate:
- 18th-century vocabulary and syntax (no contractions, no modern slang)
- Structural argument patterns (building from precedent, qualifying assertions)
- Historical accuracy (correct references to Federalist Papers, Convention debates)
- Anti-patterns (no "certainly!", no "as an AI", no anachronisms)
- Philosophical consistency (positions align with documented Madison views)

**Phase 2: Adapt the Training Script (train.py equivalent)**

The mutable training script would be a parameterized QLoRA DPO script:
```python
# All of these are fair game for the agent to modify:
LORA_RANK = 16
LORA_ALPHA = 16
LORA_TARGET_LAYERS = range(0, 40)  # First 2/3 of Gemma 3 27B
LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DPO_BETA = 0.1
LEARNING_RATE = 5e-6
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
WARMDOWN_RATIO = 0.5
DATA_SUBSET = "all"  # or specific theme filters
CURRICULUM = "none"  # or "easy_first", "hard_first"
```

**Phase 3: Run the Loop**

On Modal A100 ($1.10/hr for 80GB):
- Each 5-minute experiment costs ~$0.09
- 100 experiments overnight = ~$9
- Well within our $521 Modal credit budget

**Phase 4: Analyze Results**

The autoresearch pattern produces a `results.tsv` showing:
- Which hyperparameter changes improved Madison voice
- Optimal LoRA configuration for character imprinting on Gemma 3
- Whether curriculum ordering matters (and which order)
- The relationship between beta and voice strength vs. capability preservation

---

## Beyond Autoresearch: Additional Novel Ideas

### Automated Madison Ground-Truth Verification

Use an automated research pipeline to:
1. For each teacher response, identify the specific Madison positions referenced
2. Cross-reference against his actual writings (Federalist Papers, Convention notes, correspondence)
3. Flag any teacher responses that attribute positions Madison didn't actually hold
4. Generate "ground truth" scores for each of the 490 responses

This turns our synthetic teacher data into *verified* synthetic teacher data — a quality improvement the NeurIPS 2025 paper says is the #1 factor.

### Self-Play Iterative Refinement

After initial DPO training:
1. Fine-tuned Madison generates responses to new prompts
2. LLM judge (with constitution) scores them
3. Best responses become new chosen examples
4. Model's own worst responses become new rejected examples
5. Retrain DPO on this on-policy data
6. Repeat

This is Online DPO / Self-Rewarding (Yuan et al., 2024) applied to character training. TRL's `OnlineDPOTrainer` supports this natively.

### Compact Madison Reward Model

Train a small classifier (ModernBERT or similar) to score "Madison-ness" of any text:
- Positive examples: actual Madison writings + best teacher responses
- Hard negatives: modern paraphrases of same content, generic assistant responses, anachronistic rewrites
- Use for: reranking, data filtering, online DPO scoring, A/B evaluation

Lambert used this approach (ModernBERT classifier) to distinguish 11 personas. For a single character, it should be highly effective.

---

## Key Research References

### Autoresearch
- [Karpathy autoresearch repo](https://github.com/karpathy/autoresearch) — The original
- [SkyPilot scaling blog](https://blog.skypilot.co/scaling-autoresearch/) — 16-GPU cluster results
- [Fortune: "The Karpathy Loop"](https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/) — Industry analysis

### DPO Optimization
- [What Matters in Data for DPO? (NeurIPS 2025)](https://arxiv.org/html/2508.18312v1) — Chosen quality dominates
- [Principled Data Selection for Alignment (ICML 2025)](https://arxiv.org/html/2502.09650v1) — Difficulty filtering
- [Beta-DPO (NeurIPS 2024)](https://arxiv.org/abs/2407.08639) — Dynamic beta
- [OPA-DPO](https://opa-dpo.github.io/) — On-policy alignment before DPO

### Character/Persona Training
- [APC-DPO for persona role-play](https://arxiv.org/abs/2405.07726) — Persona faithfulness reward
- [SPIN: Self-Play Fine-tuning](https://arxiv.org/abs/2401.01335) — Corpus-matching alternative to DPO
- [SimPO](https://arxiv.org/abs/2405.14734) — Reference-free preference optimization
- [Online DPO from AI Feedback](https://arxiv.org/abs/2402.04792) — Iterative alignment template
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) — Model as its own judge

### Gemma-Specific
- [Gemma 3 ablations](https://huggingface.co/blog/tawnymanticore/gemma3-ablations) — Layer targeting, 280 pairs sufficient
- [Gemma Needs Help (LessWrong)](https://www.lesswrong.com/posts/kjnQj6YujgeMN9Erq/gemma-needs-help) — DPO required over SFT

### Curriculum & Data
- [Curry-DPO (2024)](https://arxiv.org/abs/2403.07230) — Easy-to-hard ordering for DPO
- [NVIDIA PersonaPlex](https://research.nvidia.com/labs/adlr/personaplex/) — Persona consistency and data mixing

### Tooling
- `huggingface/trl` — DPOTrainer, OnlineDPOTrainer, KTOTrainer, RewardTrainer
- `huggingface/alignment-handbook` — Offline alignment recipes
- `uclaml/SPIN` — Self-play fine-tuning implementation
- `princeton-nlp/SimPO` — Reference-free preference optimization
- `RLHFlow/Online-DPO-R1` — Iterative DPO loop example
