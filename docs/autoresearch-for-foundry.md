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

## Recursive Self-Improvement for Character Fine-Tuning

### The Gap in the State of the Art

The current state of the art in character fine-tuning (Lambert/Maiya's Open Character Training) is **single-pass**: write a constitution, generate synthetic DPO pairs, train once, evaluate, maybe manually tweak and retrain. Two passes, hand-tuned, done. What nobody has done is **close the loop** — apply recursive self-improvement to character fidelity itself.

### Four Levels of Recursive Improvement

**Level 1 — Training Recipe Optimization**
The agent modifies LoRA rank, beta, learning rate, layer targeting, batch size. Each experiment trains and evaluates against the fixed Madison authenticity judge. Useful but essentially Optuna with git — not novel by itself.

**Level 2 — Data Curation as the Mutable Artifact**
Instead of modifying the training script, the agent modifies **which data to train on and in what order**. The 490 pairs become a selection space. The agent experiments with: removing the weakest 10-30% of pairs, reordering by theme, weighting constitutional topics 2x, filtering by difficulty score. The ICML 2025 paper on difficulty filtering and Curry-DPO on curriculum ordering both show these choices matter enormously — but they require manual tuning. An agent explores the combinatorial space overnight.

**Level 3 — Self-Play Data Generation**
After Levels 1-2 find the best recipe and data selection, the trained model generates new responses. The agent now has three sources of rejected data: (a) original base Gemma responses, (b) Round 1 fine-tuned model responses, (c) Round 2 model responses. Each round, rejected examples get closer to the teacher, forcing the model to learn finer distinctions. This is SPIN (Self-Play Fine-tuning) automated. Research shows up to 8x data efficiency gains from on-policy data, but nobody has automated the iteration loop for character training.

**Level 4 — Judge Refinement (the deepest level)**
The evaluation function's **judge prompt** can itself be improved — carefully, to avoid reward hacking. The agent doesn't modify the constitution (immutable ground truth). Instead, it refines the instructions that tell the judge how to score responses against the constitution. Test against known-good examples (Madison's actual verbatim writings — our 8 verified response prompts) and known-bad examples (modern paraphrases). A judge that scores Madison's real words at 9 and a modern paraphrase at 3 is better calibrated than one that gives 7 and 5.

The trust boundary holds because the constitution never changes and the verified responses (Madison's actual words) are ground truth that can't be gamed. The judge gets better at *detecting* Madison-ness, not at *redefining* it.

### What This Produces That Nobody Else Has

You go to bed with a baseline Madison model. You wake up with:

1. The optimal LoRA configuration for Gemma 3 27B character imprinting (discovered, not guessed)
2. The optimal data selection and curriculum from the 490 pairs
3. A second-generation model trained on its own on-policy failures
4. A calibrated authenticity scorer that reliably distinguishes real Madison from imitations
5. A `results.tsv` showing exactly which changes mattered and by how much

The publishable contribution: **recursive self-improvement via automated experimentation produces character models that exceed what manual tuning achieves — and here's the reusable methodology anyone can apply to any historical figure.**

### Running on the RTX 3090 Overnight

The autoresearch loop can run entirely on the local RTX 3090 (24GB VRAM) instead of Modal A100, saving compute credits for the initial training runs.

**Why it works on the 3090:**
- Gemma 3 27B at 4-bit = ~14GB VRAM for weights
- QLoRA training with batch_size=1 + gradient accumulation fits in 24GB
- Each 5-minute experiment: load cached base model, apply QLoRA DPO, evaluate
- Overnight (8 hours): ~96 experiments at 5 min each
- Cost: $0 (local hardware)

**Constraints vs A100:**
- Slower per-experiment (3090 vs A100 throughput)
- Tighter VRAM budget — batch_size=1 only, shorter max_seq_length may be needed
- 5-minute budget produces fewer training steps per experiment
- Evaluation requires generating responses on the same GPU (serial, not parallel)

**Recommended approach:**
1. Run initial DPO training on Modal A100 (faster, more VRAM headroom, first run needs to work)
2. Export the baseline LoRA adapter and GGUF
3. Run the autoresearch loop overnight on the 3090 for Levels 1-2 (recipe + data optimization)
4. Level 3 self-play rounds can also run on the 3090 — generate responses locally, retrain locally
5. Use Modal only when needing to export final GGUF or for experiments requiring more VRAM

**3090 autoresearch setup:**
```
prepare.py (IMMUTABLE):
  - Load Gemma 3 27B Q4 + LoRA adapter via Unsloth
  - Generate responses to 20-30 held-out eval prompts
  - Score via Anthropic API (judge runs on Claude, not locally)
  - Return single numeric MadisonScore

train.py (MUTABLE):
  - QLoRA DPO training on local GPU
  - All hyperparameters exposed as variables
  - Data selection/ordering configurable
  - 5-minute wall-clock budget per experiment

program.md:
  "Run on local RTX 3090.
   Train for 5 minutes, evaluate via API judge.
   Keep improvements, discard regressions.
   Try: beta sweep, rank sweep, data curriculum, theme weighting.
   Run overnight. Log to results.tsv."
```

The judge scoring calls the Anthropic API (Sonnet 4.6) which costs ~$0.01-0.02 per eval prompt × 30 prompts × 96 experiments = ~$30-60 overnight in API costs. This is the main expense — the GPU compute is free.

### The Overnight Schedule

```
NIGHT 1: Hyperparameter Sweep (Level 1)
  3090 runs 96 experiments varying beta, rank, lr, layer targeting
  Each: 5-min train → generate 30 responses → API judge scores
  Output: optimal training recipe

NIGHT 2: Data Curation Sweep (Level 2)
  3090 runs 96 experiments varying data selection, ordering, filtering
  Uses best recipe from Night 1
  Output: optimal data configuration

NIGHT 3: Self-Play Round (Level 3)
  Best model from Night 2 generates 490 new responses
  Score and rank them → build new DPO pairs (teacher vs round-2 model)
  Train on improved data, evaluate
  Output: second-generation Madison model

NIGHT 4: Judge Calibration + Final Polish (Level 4)
  Refine judge prompt against verified responses
  Retrain with calibrated scoring
  Export final GGUF via Modal for deployment
  Output: production Madison model + calibrated evaluation pipeline
```

Total cost: ~$120-240 in API judge calls + $5-20 for Modal GGUF exports. Zero GPU compute cost.

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
