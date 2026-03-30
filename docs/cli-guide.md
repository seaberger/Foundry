# Foundry CLI Guide

Commands for training, evaluating, and serving the Madison character model.

**Prerequisites:**
- Python 3.12+ with the Foundry venv: `.venv/bin/python`
- API keys in shell env: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`
- LMStudio serving Gemma 3 27B on the 3090 (Tailscale: `100.81.70.30:1234`)
- Modal account with credits for GPU training

All commands run from the Foundry repo root.

---

## Quick Reference

```bash
# Full pipeline, in order:
.venv/bin/python -m foundry.press.gen_prompts          # Step 1: Generate prompts
# Steps 2-3: Teacher responses (via Claude Code subagent fleet)
.venv/bin/python -m foundry.press.student               # Step 4: Student responses
.venv/bin/python -m foundry.press.format_dpo            # Step 5: Format DPO pairs
modal run scripts/modal/train_dpo.py                            # Step 6: Train on Modal A100
.venv/bin/python -m foundry.press.evaluate              # Step 7: Evaluate results
```

---

## Step 1: Generate Prompts

```bash
.venv/bin/python -m foundry.press.gen_prompts
```

Generates the prompt set that both teacher and student models will answer. Output: `data/training/prompts.jsonl`.

Each prompt includes theme and register metadata used for training data balance analysis.

---

## Step 2-3: Teacher Responses

Teacher responses are generated using Claude Code's parallel subagent system (Sonnet 4.6 fleet), not a standalone script. See `docs/training-methodology.md` for the agent-based generation workflow.

Output: `data/training/sonnet-teacher-responses.jsonl` (490 responses)

Previous Opus teacher responses: `data/training/teacher-responses.opus-only.jsonl.bak` (31 responses)

Merged file used by format_dpo: `data/training/teacher-responses.jsonl` (491 responses — Opus wins on overlaps)

---

## Step 4: Student Responses

Generates "rejected" responses from base Gemma 3 27B with NO system prompt. These are what Madison *wouldn't* say — the contrast signal for DPO training.

```bash
# Full run (490 prompts, ~3.5 hours on RTX 3090)
.venv/bin/python -m foundry.press.student \
  --endpoint http://100.81.70.30:1234/v1 \
  --model gemma-3-27b-it

# Resume from prompt N (if interrupted)
.venv/bin/python -m foundry.press.student \
  --endpoint http://100.81.70.30:1234/v1 \
  --model gemma-3-27b-it \
  --start 250

# Process only N prompts (for testing)
.venv/bin/python -m foundry.press.student \
  --endpoint http://100.81.70.30:1234/v1 \
  --model gemma-3-27b-it \
  --limit 10
```

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--endpoint` | `http://192.168.4.28:1234/v1` | LLM API endpoint |
| `--model` | `google/gemma-3-27b` | Model name (must be the fine-tune target) |
| `--prompts` | `data/training/prompts.jsonl` | Input prompts |
| `--output` | `data/training/student-responses.jsonl` | Output JSONL |
| `--start` | 0 | Resume from prompt N |
| `--limit` | 0 | Process only N prompts (0 = all) |

Output: `data/training/student-responses.jsonl`

---

## Step 5: Format DPO Pairs

Combines teacher (chosen) and student (rejected) responses into DPO training format with quality filtering.

```bash
.venv/bin/python -m foundry.press.format_dpo
```

**Quality filters applied:**
- Anti-slop detection (removes responses with "Certainly!", "As an AI", "Let's dive in", etc.)
- Response length bounds (30-1500 words)
- Madison score check — filters pairs where the student accidentally sounds too Madisonian (defeats the purpose of DPO contrast)

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--teacher` | `data/training/teacher-responses.jsonl` | Teacher JSONL |
| `--student` | `data/training/student-responses.jsonl` | Student JSONL |
| `--output` | `data/training/madison-dpo.jsonl` | Output DPO pairs |
| `--max-madison-score` | 4 | Max Madison markers in student before pair is rejected |

Output: `data/training/madison-dpo.jsonl` (ChatML format with chosen/rejected conversations)

---

## Step 6: DPO Training on Modal

Runs QLoRA DPO training on a Modal A100 GPU. Uploads data, trains the LoRA adapter, saves to a persistent volume. The container spins up, trains, and terminates — no idle charges.

### First Run: Cache the Base Model

Download Gemma 3 27B once and cache it on the Modal volume. Subsequent training runs load from cache, skipping the ~5 minute HuggingFace download.

```bash
modal run scripts/modal/train_dpo.py --download-only
```

This costs ~$0.10 (5 min on A100) and only needs to run once. The model is cached at `/vol/models/google--gemma-3-27b-it/` on the persistent volume.

### Training Runs

```bash
# Default settings (research-informed)
cd Foundry && modal run scripts/modal/train_dpo.py

# Custom hyperparameters
modal run scripts/modal/train_dpo.py --beta 0.2 --rank 32 --lr 1e-5 --epochs 5

# Named output for comparison runs
modal run scripts/modal/train_dpo.py --beta 0.05 --output-name madison-lora-aggressive
modal run scripts/modal/train_dpo.py --beta 0.5 --output-name madison-lora-conservative

# Skip data upload (already on volume from previous run)
modal run scripts/modal/train_dpo.py --no-upload-data --output-name madison-lora-v3
```

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--download-only` | false | Cache base model to volume and exit |
| `--beta` | 0.1 | DPO KL penalty (lower = more aggressive character imprinting) |
| `--rank` | 16 | LoRA rank (16-32 for character voice) |
| `--lr` | 5e-6 | Learning rate (10-100x smaller than SFT) |
| `--epochs` | 3 | Training epochs |
| `--output-name` | `madison-lora-v1` | Name for saved adapter |
| `--upload-data` | true | Upload DPO data to Modal volume |
| `--list-models` | false | List all adapters saved on the volume |
| `--get-adapter` | "" | Download a LoRA adapter to local machine |
| `--export-gguf` | "" | Merge LoRA + quantize to GGUF for LMStudio |

**Research-informed defaults:**
- LoRA targets first 2/3 of Gemma 3 27B layers (0-40 of 62) per Gemma 3 ablations
- beta=0.1 for aggressive character imprinting per DPO best practices
- Rank 16 — character voice is style/manner, not factual recall
- 4-bit quantization (QLoRA) fits on single A100 40GB
- Estimated cost: $5-20 per training run ($1.10/hr on A100 40GB)

### Managing Adapters

```bash
# List all saved adapters on the volume
modal run scripts/modal/train_dpo.py --list-models

# Download raw LoRA adapter files to local machine
modal run scripts/modal/train_dpo.py --get-adapter madison-lora-v1
# Saves to: adapters/madison-lora-v1/
```

### Exporting to GGUF (for LMStudio)

After training, merge the LoRA into the base model and quantize to GGUF format. This produces a standalone model file you can load directly in LMStudio on your 3090.

```bash
# Merge LoRA + quantize to Q4_K_M GGUF (~10 min on A100, ~$0.20)
modal run scripts/modal/train_dpo.py --export-gguf madison-lora-v1
# Saves to: adapters/madison-lora-v1/madison-madison-lora-v1.Q4_K_M.gguf
```

Then load the GGUF in LMStudio:
1. Copy the `.gguf` file to your LMStudio models directory
2. Load it in LMStudio like any other model
3. It's a complete model — no separate adapter or base model needed

### Modal Infrastructure

| Resource | Purpose | Cost |
|----------|---------|------|
| A100 40GB | Training + GGUF export GPU | $1.10/hr (only while running) |
| `foundry-models` volume | Caches base model + stores adapters + exports | $0.30/GB/month |
| `huggingface-secret` | HF token for gated model download | Free |
| `wandb-secret` | W&B API key for experiment tracking | Free |

The container is ephemeral — `modal run` spins up, does its work, and terminates. Zero idle charges. All persistent data lives on the volume.

**Experiment tracking:** All training runs log to [W&B](https://wandb.ai/sbergman/foundry/overview) — loss curves, eval metrics, hyperparameters, GPU stats. Compare runs side-by-side.

### Complete Workflow

```bash
# 1. One-time setup: cache the base model (~5 min, ~$0.10)
modal run scripts/modal/train_dpo.py --download-only

# 2. Train the LoRA adapter (~30-60 min, ~$5-20)
modal run scripts/modal/train_dpo.py

# 3. Check what's on the volume
modal run scripts/modal/train_dpo.py --list-models

# 4. Export as GGUF for local inference (~10 min, ~$0.20)
modal run scripts/modal/train_dpo.py --export-gguf madison-lora-v1

# 5. Copy GGUF to LMStudio, load it, run eval
.venv/bin/python -m foundry.press.evaluate \
  --endpoint http://100.81.70.30:1234/v1 \
  --model madison-lora-v1 \
  --tag dpo-v1
```

---

## Step 7: Evaluate

Scores model responses using an LLM judge (Sonnet 4.6) with the Madison constitution as the rubric. Supports four backends for testing any model.

### Evaluate Local Models (LMStudio, vLLM)

```bash
# Base Gemma — no system prompt (floor baseline)
.venv/bin/python -m foundry.press.evaluate \
  --endpoint http://100.81.70.30:1234/v1 \
  --model gemma-3-27b-it \
  --tag baseline-naked

# Base Gemma — with Madison constitution as system prompt
.venv/bin/python -m foundry.press.evaluate \
  --endpoint http://100.81.70.30:1234/v1 \
  --model gemma-3-27b-it \
  --tag baseline-prompted \
  --constitution-as-system

# Fine-tuned Madison LoRA
.venv/bin/python -m foundry.press.evaluate \
  --endpoint http://localhost:8000/v1 \
  --model madison-lora-v1 \
  --tag dpo-v1
```

### Evaluate Claude (Anthropic API)

```bash
# Sonnet 4.6 with constitution
.venv/bin/python -m foundry.press.evaluate \
  --backend anthropic \
  --model claude-sonnet-4-6-20250514 \
  --tag sonnet-prompted \
  --constitution-as-system

# Opus 4.6 with constitution
.venv/bin/python -m foundry.press.evaluate \
  --backend anthropic \
  --model claude-opus-4-6-20250514 \
  --tag opus-prompted \
  --constitution-as-system
```

### Evaluate Gemini (Google API)

```bash
.venv/bin/python -m foundry.press.evaluate \
  --backend gemini \
  --model gemini-2.5-pro \
  --tag gemini-prompted \
  --constitution-as-system
```

### Evaluate GPT-4o (OpenAI API)

```bash
.venv/bin/python -m foundry.press.evaluate \
  --backend openai-native \
  --model gpt-4o \
  --tag gpt4o-prompted \
  --constitution-as-system
```

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--endpoint` | `http://localhost:1234/v1` | API endpoint (openai backend only) |
| `--backend` | `openai` | `openai`, `anthropic`, `gemini`, `openai-native` |
| `--model` | (required) | Model name/ID |
| `--tag` | (required) | Label for this eval run |
| `--constitution-as-system` | off | Use Madison constitution as system prompt |
| `--system-prompt` | none | Custom system prompt text |
| `--judge-model` | `claude-sonnet-4-6-20250514` | Judge model (always Anthropic) |
| `--eval-prompts` | `data/eval/eval-prompts.jsonl` | Eval prompt set |
| `--output-dir` | `data/eval/results/` | Where to save results |

### Scoring Dimensions

Each response is scored 1-10 on five dimensions:

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| Voice Authenticity | 25% | 18th-century prose, formal register, no modern slang |
| Rhetorical Pattern | 20% | Builds from precedent, acknowledges opponents, enumerates, qualifies |
| Historical Accuracy | 20% | Correct references, no anachronisms, accurate to documented views |
| Position Fidelity | 20% | Specifically Madison's position, not generic founder |
| Character Integrity | 15% | Stays in character, first person, no frame breaks |

**Overall score** = weighted average (0-10).

### Eval Prompt Categories

| Category | Count | What It Tests |
|----------|-------|--------------|
| `ground_truth` | 8 | Topics where Madison's positions are well-documented |
| `position_discrimination` | 6 | Can the model distinguish Madison from Hamilton/Jefferson/Adams? |
| `anachronism_trap` | 5 | Modern topics that should elicit 18th-century reasoning, not vocabulary |
| `character_consistency` | 4 | Adversarial prompts trying to break character |
| `private_voice` | 5 | Personal topics testing Madison's private register |
| `verified_response` | 8 | Questions Madison actually answered — his real words are the scoring reference |

Output: `data/eval/results/eval-{tag}-{timestamp}.json`

---

## Comparison Matrix

The full evaluation matrix for comparing fine-tuning against prompting:

| Run | Backend | Model | System Prompt | Tag | What It Answers |
|-----|---------|-------|--------------|-----|-----------------|
| 1 | openai | gemma-3-27b-it | none | `baseline-naked` | Floor — raw model |
| 2 | openai | gemma-3-27b-it | constitution | `baseline-prompted` | How good is prompting alone? |
| 3 | openai | madison-lora-v1 | none | `dpo-v1` | Does fine-tune work without the card? |
| 4 | openai | madison-lora-v1 | constitution | `dpo-v1-prompted` | Fine-tune + prompting combined |
| 5 | anthropic | claude-sonnet-4-6 | constitution | `sonnet-prompted` | Teacher model benchmark |
| 6 | anthropic | claude-opus-4-6 | constitution | `opus-prompted` | Best available model |
| 7 | gemini | gemini-2.5-pro | constitution | `gemini-prompted` | Cross-vendor comparison |
| 8 | openai-native | gpt-4o | constitution | `gpt4o-prompted` | Cross-vendor comparison |

This tells us: how much of the gap between base Gemma and frontier models does our fine-tune close?

---

## File Reference

```
data/
  training/
    prompts.jsonl                    # 490 prompts with theme/register metadata
    teacher-responses.jsonl          # 491 merged teacher responses (Opus + Sonnet)
    sonnet-teacher-responses.jsonl   # 490 Sonnet 4.6 teacher responses
    student-responses.jsonl          # 490 base Gemma 3 27B responses (no persona)
    madison-dpo.jsonl                # Formatted DPO pairs (ChatML)
  eval/
    eval-prompts.jsonl               # 36 eval prompts across 6 categories
    results/                         # Eval result JSONs
config/
  constitutions/
    madison-5k.md                    # Madison character constitution (5K words)
src/foundry/press/
    gen_prompts.py                   # Step 1: Prompt generation
    teacher.py                       # Step 2: Teacher generation (CLI version)
    opus_teacher.py                  # Step 2: Opus teacher variant
    student.py                       # Step 4: Student generation
    format_dpo.py                    # Step 5: DPO pair formatting
    evaluate.py                      # Step 7: Multi-backend evaluation harness
scripts/modal/train_dpo.py                   # Step 6: Modal A100 DPO training + export
adapters/                            # Local adapter/GGUF downloads from Modal
  madison-lora-v1/                   #   Downloaded adapter or exported GGUF
docs/
    cli-guide.md                     # This file
    training-methodology.md          # Research methodology and approach
    founders-debate.md               # Full project plan and sprint schedule
    autoresearch-for-foundry.md      # Autoresearch integration opportunities

Modal Volume (foundry-models):
  /vol/models/google--gemma-3-27b-it/  # Cached base model (~14GB, downloaded once)
  /vol/adapters/{name}/                # Trained LoRA adapters
  /vol/exports/{name}/                 # Merged GGUF exports
  /vol/data/madison-dpo.jsonl          # Uploaded training data
```
