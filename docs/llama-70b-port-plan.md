# Llama 3.3 70B Port Plan

Training plan for porting the Madison ORPO character-voice pipeline from Qwen3-32B to Llama 3.3 70B. Goal: eliminate the "small model feel" on complex constitutional reasoning while preserving the voice quality achieved at 32B (8.97/10 corrected).

**Created:** 2026-04-11
**Status:** Planning
**Baseline to beat:** Qwen 3 R2 — 8.97/10 corrected (v6 dataset, 1,498 pairs)

---

## Why Llama 3.3 70B

### Model Selection Research (2026-04-11)

The 70B dense model class was surveyed comprehensively. The field is thin:

| Model | Params | Dense? | License | Persona Suitability |
|-------|--------|--------|---------|-------------------|
| **Llama 3.3 70B** | 70.6B | Yes | Llama Community | **Best** — most proven for persona fine-tuning |
| Qwen 2.5 72B | 72.7B | Yes | Apache 2.0 | Moderate — resists personality modification |
| DeepSeek-R1-Distill-Llama-70B | ~70B | Yes (Llama arch) | Llama Community | Poor — reasoning-optimized, fights character voice |
| Mistral Large 2 | 123B | Yes | Research Only | N/A — license blocks commercial use |

**Models that don't exist at 70B dense:** Llama 4 (100% MoE), Qwen 3 (tops at 32B dense), DeepSeek V3 (MoE), Gemma (never above 31B).

### Why Not MoE (Scout 109B/17B-active)?

MoE was evaluated and rejected for character voice specifically:

- **Expert routing is token-level, not style-level.** The router selects experts based on content patterns, not voice/style. LoRA adapters don't change routing behavior, so you're fine-tuning experts' weights but not which experts activate for Madison-style patterns.
- **Effective capacity paradox.** Scout's 17B active parameters per token is smaller than the current Qwen3-32B. Character voice is a style overlay across all tokens, not a content-routing problem — you get less voice capacity from 17B-active than 70B-dense.
- **LoRA influence is more diffuse.** On dense 70B, rank-64 LoRA modifies a consistent 70B computation path every token. On MoE, the same adapter modifies whichever 17B subset was routed — changing token by token.
- **Quantization + LoRA on MoE is less proven.** Our rank-16 quantization fragility would likely be worse with MoE expert weight interactions.

### Why Llama Over Qwen at 70B

- Lambert's persona research tested Llama and found good persona imprinting; Qwen resists personality modification
- Community evidence: EVA-Qwen2.5-72B needed full-parameter fine-tune on 8x H100s for creative output — not LoRA-achievable
- Llama 3.3 has the most mature fine-tuning ecosystem (Unsloth, TRL, axolotl, LlamaFactory)
- ORPO recipes (Labonne) specifically reference Llama 3 as the target architecture
- 128K context window matches Qwen 3

---

## What Transfers Directly

| Component | Status | Notes |
|-----------|--------|-------|
| **Dataset** (`madison-orpo-v6.jsonl`) | Direct transfer | Conversational message format (role/content dicts). TRL ORPOTrainer applies Llama 3.3 chat template automatically |
| **Evaluation pipeline** (`evaluate.py`, `judge_responses.py`) | No changes | Model-agnostic — scores any Madison output |
| **36 eval prompts** (6 categories x 6) | No changes | Same prompts, same weighted scoring |
| **ORPO beta** | 0.1 | Paper default, stable across scales. Narrow safe band (0.12 was catastrophic on Qwen3) |
| **LoRA targets** | All 7 modules | q/k/v/o/gate/up/down_proj |
| **Rank** | 64 | Floor for quantization survival |
| **Alpha** | 64 | Matched to rank |
| **Dropout** | 0.0 | Unsloth recommendation |
| **Epochs** | 3 | Consistent across all successful runs |
| **Optimizer** | AdamW 8-bit | Standard for QLoRA |
| **Scoring methodology** | No changes | Weighted average: Voice 25%, Rhetorical 20%, Historical 20%, Position 20%, Integrity 15% |
| **System prompt** | Minor edit | Drop `/no_think` prefix (Llama 3.3 has no thinking mode) |
| **Adapter-on-base serving** | Required | Merged models produce character breaks — architecture-independent finding |

---

## What Changes

| Parameter | Qwen3-32B (current) | Llama 70B 4-bit (control) | Llama 70B 8-bit (primary) |
|-----------|---------------------|---------------------------|---------------------------|
| **Base model** | `Qwen/Qwen3-32B` | `unsloth/Llama-3.3-70B-Instruct-bnb-4bit` | `meta-llama/Llama-3.3-70B-Instruct` |
| **Quantization** | 4-bit NF4 | 4-bit NF4 | **8-bit INT8 (LLM.int8())** |
| **GPUs** | 1x A100 80GB | 1x A100 80GB | **2x A100 80GB** |
| **Framework** | Unsloth + TRL | Unsloth + TRL | **HF + bitsandbytes + TRL** |
| **Learning rate** | 2e-5 | Sweep: 1e-5 | Sweep: 1e-5, 7e-6 |
| **Batch size** | 1, grad_accum 4 | 1, grad_accum 8 | **2**, grad_accum 4 |
| **Effective batch** | 4 | 8 | 8 |
| **Gradient checkpointing** | Standard | `"unsloth"` mode | Standard PyTorch |
| **Modal timeout** | 240 min | 360 min | **360 min** |
| **Adapter name** | `madison-qwen3-r2-v1` | `madison-llama3.3-70b-4bit-v1` | `madison-llama3.3-70b-8bit-v1` |
| **Est. cost/run** | ~$8-15 | ~$15-25 | **~$40-70** |

---

## Training Precision: 4-bit vs 8-bit

### The Quantization Question

Our Qwen3-32B experience showed catastrophic degradation from quantization at **inference** time (BF16: 8.52/10 → GGUF Q4_K_M: 1.74/10). All successful training used 4-bit QLoRA with adapter-on-base serving. The 4-bit training was not the bottleneck — but was it leaving quality on the table?

The hypothesis: **higher-precision base weights during training produce better character voice** because:
- ORPO's odds-ratio log-probability estimates are computed against a higher-fidelity base
- Subtle stylistic features may live in activation outliers that 4-bit NF4 noise-floors
- The LoRA adapters learn corrections relative to the base representation — richer base = richer learning signal
- Bitsandbytes LLM.int8() keeps outlier features in FP16 via mixed decomposition, preserving exactly the kind of long-tail activation patterns that character voice may depend on

### What Fits on 2x A100 80GB (160GB total)

| Precision | Base Model VRAM | + Training Overhead | Total | Fits? |
|-----------|----------------|---------------------|-------|-------|
| **BF16** (full) | ~140 GB | ~15-40 GB | ~155-180 GB | No — activations push past 160GB |
| **8-bit INT8** | ~70 GB | ~25-45 GB | ~95-115 GB | **Yes — comfortably** |
| **4-bit NF4** | ~35 GB | ~25-45 GB | ~60-80 GB | Yes — single A100 sufficient |

**8-bit is the highest precision that fits on 2x A100 80GB.** BF16 at 70B needs ~140GB for weights alone, leaving insufficient room for activations and gradient checkpointing buffers. Full BF16 training would require 4x A100 or 2x H100.

### Trade-offs: 4-bit vs 8-bit Training

| Factor | 4-bit NF4 (1x A100) | 8-bit INT8 (2x A100) |
|--------|---------------------|----------------------|
| **Quantization noise** | Higher — ~4x more error | Lower — mixed INT8/FP16 decomposition |
| **Outlier preservation** | Noise-floored | FP16 for activations above 6σ threshold |
| **Training speed** | Fast — Unsloth 2x optimization | Slower — standard HF+bitsandbytes (~2-3x slower) |
| **Batch size** | 1 (memory constrained) | 2 (more headroom) |
| **Sequence length** | 2048 (safe max) | 2048-4096 (more headroom) |
| **Modal cost/run** | ~$15-25 (1x A100, 2-4 hrs) | ~$40-70 (2x A100, 3-6 hrs) |
| **Framework** | Unsloth + TRL | HuggingFace + bitsandbytes + TRL (no Unsloth 8-bit optimizations) |
| **Multi-GPU** | N/A | FSDP (NOT DeepSpeed ZeRO-3 — incompatible with QLoRA) |

### 8-bit Training Configuration (2x A100)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import ORPOConfig, ORPOTrainer
import torch

# === 8-bit Quantization ===
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,  # Outliers above 6σ kept in FP16
)

# === Model (shards across 2 GPUs automatically) ===
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.3-70B-Instruct",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Shards across 2 GPUs
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

# === LoRA ===
lora_config = LoraConfig(
    r=64,
    lora_alpha=64,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
    bias="none",
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()

# === ORPO Training ===
orpo_args = ORPOConfig(
    learning_rate=1e-5,
    beta=0.1,
    lr_scheduler_type="cosine",
    max_length=2048,
    max_prompt_length=1024,
    max_completion_length=1024,
    per_device_train_batch_size=2,   # More headroom at 8-bit
    gradient_accumulation_steps=4,   # Effective batch = 8 (2 GPUs x 2 batch x 4 accum... adjust based on actual FSDP config)
    optim="adamw_8bit",
    num_train_epochs=3,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    logging_steps=1,
    eval_steps=0.2,
    evaluation_strategy="steps",
    bf16=True,
    output_dir="./orpo_llama70b_8bit_output",
    report_to="wandb",
)
```

**Modal config for 2x A100:**
```python
GPU = modal.gpu.A100(count=2, size="80GB")
TIMEOUT = 360 * MINUTES  # 6 hour cutoff for 8-bit (slower than 4-bit)
```

**Important:** Do NOT use `device_map="auto"` with ORPOTrainer on multi-GPU — known bug ([TRL #1571](https://github.com/huggingface/trl/issues/1571)). Use `accelerate launch` with FSDP instead. DeepSpeed ZeRO-3 is incompatible with QLoRA at 70B scale ([LlamaFactory #4862](https://github.com/hiyouga/LlamaFactory/issues/4862)).

---

## 4-bit Training Configuration (1x A100, Unsloth)

```python
from unsloth import FastLanguageModel
from trl import ORPOConfig, ORPOTrainer
import torch

# === Model ===
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,  # Auto-detect
)

# === LoRA ===
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=64,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",  # Critical for 70B
    random_state=42,
)

# === ORPO Training ===
orpo_args = ORPOConfig(
    learning_rate=1e-5,              # Start here, sweep down
    beta=0.1,                        # ORPO paper default
    lr_scheduler_type="cosine",
    max_length=2048,
    max_prompt_length=1024,
    max_completion_length=1024,
    per_device_train_batch_size=1,   # Memory constrained on single A100
    gradient_accumulation_steps=8,   # Effective batch = 8
    optim="adamw_8bit",
    num_train_epochs=3,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    logging_steps=1,
    eval_steps=0.2,
    evaluation_strategy="steps",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    output_dir="./orpo_llama70b_4bit_output",
    report_to="wandb",
)
```

---

## VRAM Budgets

### 8-bit INT8 — 2x A100 80GB (160GB total)

| Component | Estimated VRAM |
|-----------|---------------|
| Base model (8-bit INT8 with FP16 outliers) | ~70 GB |
| LoRA adapters (bf16, rank 64, all modules) | ~2.8 GB |
| Optimizer states (adamw_8bit) | ~2.8 GB |
| Activations + gradients (with gradient checkpointing) | ~25-40 GB |
| **Total** | **~100-115 GB** |

Comfortable fit on 160GB. Room for batch size 2 or sequence length 4096.

### 4-bit NF4 — Single A100 80GB

| Component | Estimated VRAM |
|-----------|---------------|
| Base model (4-bit NF4) | ~35 GB |
| LoRA adapters (bf16, rank 64, all modules) | ~2.8 GB |
| Optimizer states (adamw_8bit) | ~2.8 GB |
| Activations + gradients (with Unsloth gradient checkpointing) | ~20-30 GB |
| **Total** | **~60-70 GB** |

Fits on single A100 80GB with rank 64, batch 1, seq_len 2048. ORPO's no-reference-model design saves ~35GB vs DPO (which would need a frozen reference copy and be impossible on single A100).

### LoRA Rank Memory Comparison

| Rank | Trainable Params | % of 70B | Adapter + Optimizer Memory |
|------|-----------------|----------|---------------------------|
| 16 | ~350M | ~0.5% | ~1.4 GB |
| 32 | ~700M | ~1.0% | ~2.8 GB |
| **64** | **~1.4B** | **~2.0%** | **~5.6 GB** |
| 128 | ~2.8B | ~4.0% | ~11.2 GB |

Rank 64 → 16 delta is only ~4.2 GB. Base model and activations dominate memory, not LoRA adapters.

---

## Experiment Plan

### Phase 1: Precision A/B Test + LR Sweep

Apply existing v6 dataset directly to Llama 3.3 70B at two quantization levels. Answer two questions simultaneously: (1) does the larger model improve complex reasoning, and (2) does training precision matter for character voice?

#### 8-bit Runs (2x A100 80GB — primary)

| Run | Precision | LR | Beta | Rank | GPUs | Batch | Dataset | Est. Time | Est. Cost |
|-----|-----------|-----|------|------|------|-------|---------|-----------|-----------|
| v1a | 8-bit INT8 | 1e-5 | 0.1 | 64 | 2x A100 | 2 | v6 (1,498 pairs) | 3-6 hrs | ~$40-70 |
| v1b | 8-bit INT8 | 7e-6 | 0.1 | 64 | 2x A100 | 2 | v6 (1,498 pairs) | 3-6 hrs | ~$40-70 |

#### 4-bit Control (1x A100 80GB — baseline comparison)

| Run | Precision | LR | Beta | Rank | GPUs | Batch | Dataset | Est. Time | Est. Cost |
|-----|-----------|-----|------|------|------|-------|---------|-----------|-----------|
| v1c | 4-bit NF4 | 1e-5 | 0.1 | 64 | 1x A100 | 1 | v6 (1,498 pairs) | 2-4 hrs | ~$15-25 |

**Total Phase 1 cost:** ~$95-165 Modal credits

**What Phase 1 tells us:**

- **v1a vs v1c** (same LR, different precision): Does 8-bit training produce better character voice than 4-bit? If scores are similar, 4-bit is sufficient and we save money on all future runs.
- **v1a vs v1b** (same precision, different LR): LR sweep at the higher-fidelity precision level.
- **All vs Qwen3-32B R2 baseline** (8.97): Does the 70B model improve complex reasoning?

**Success criteria:**
- At least one run scores >= 8.97 corrected (matching Qwen3-32B R2)
- Qualitative improvement on complex reasoning prompts (ground_truth, verified_response categories)
- Clear signal on whether 8-bit vs 4-bit matters (>0.3 score difference = significant)

**Evaluation:** Same 36-prompt eval pipeline, same Claude Sonnet 4.6 judge, same weighted scoring. Direct comparison against Qwen 3 R2 baseline.

**Decision tree after Phase 1:**
- If 8-bit >> 4-bit: all future runs use 8-bit on 2x A100. The precision matters.
- If 8-bit ≈ 4-bit: all future runs use 4-bit on 1x A100. Save money, faster iteration.
- If neither beats 8.97: try LR=2e-5 (proven Qwen value) before concluding architecture mismatch. Consider regenerating rejected examples from Llama 3.3 base model.

### Phase 2: LR Refinement (Based on Phase 1 Winner)

Using whichever precision won Phase 1, narrow the LR:

| Run | LR | Purpose |
|-----|-----|---------|
| v2a | Best from Phase 1 ± 30% up | Probe above |
| v2b | Best from Phase 1 ± 30% down | Probe below |

Only run if Phase 1 shows promise but the best score suggests room for optimization.

### Phase 3: Data Expansion (Only if Phases 1-2 Show Promise)

- Expand training data if category-specific scores reveal bottlenecks
- Same source-enrichment approach that broke through verified_response at 32B
- Consider rank 128 only if planning GGUF deployment (adapter-on-base avoids this)

### Phase 4: Iteration (Only if Needed)

- Beta sweep (0.05, 0.1, 0.15) if preference signal appears too strong/weak
- Additional ORPO rounds with on-policy data (not SFT — SFT after ORPO is forbidden)

---

## Script Modifications Required

### Two Training Scripts Needed

The 4-bit and 8-bit paths use different frameworks, so we need two scripts:

#### 4-bit Script (adapt existing): `modal_train_orpo_llama_4bit.py`

Fork from `experiments/qwen3-val/modal_train_orpo_qwen.py` with:

1. Model name: `Qwen/Qwen3-32B` → `unsloth/Llama-3.3-70B-Instruct-bnb-4bit`
2. Gradient checkpointing: ensure `use_gradient_checkpointing="unsloth"`
3. Gradient accumulation: 4 → 8
4. Modal GPU: `A100-80GB` (single)
5. Modal timeout: 240 → 360 minutes
6. Default adapter output name: `madison-llama3.3-70b-4bit-v1`
7. Verify chat template application on first batch before full run

#### 8-bit Script (new): `modal_train_orpo_llama_8bit.py`

New script using standard HuggingFace + bitsandbytes (no Unsloth):

1. Model: `meta-llama/Llama-3.3-70B-Instruct` with `BitsAndBytesConfig(load_in_8bit=True)`
2. LoRA via `peft.get_peft_model()` (not Unsloth's `get_peft_model`)
3. Standard PyTorch `gradient_checkpointing_enable()`
4. Modal GPU: `modal.gpu.A100(count=2, size="80GB")`
5. Use `accelerate launch` with FSDP config (not `device_map="auto"`)
6. Modal timeout: 360 minutes
7. Default adapter output name: `madison-llama3.3-70b-8bit-v1`

### Serve Script (`scripts/modal/serve_madison_qwen.py`)

1. Update `BASE_MODEL` to Llama 3.3 70B
2. Update `ADAPTER_NAME` to new adapter
3. Drop `/no_think` from system prompt
4. vLLM handles Llama 3.3 natively — no architectural changes
5. Keep `lora_max_rank=64`

### Data Format

No reformatting needed. The v6 dataset uses conversational format (list of role/content message dicts):

```json
{
  "chosen": [
    {"role": "user", "content": "<prompt>"},
    {"role": "assistant", "content": "<chosen_response>"}
  ],
  "rejected": [
    {"role": "user", "content": "<prompt>"},
    {"role": "assistant", "content": "<rejected_response>"}
  ]
}
```

TRL's ORPOTrainer applies the tokenizer's chat template automatically. Llama 3.3 uses `<|start_header_id|>` / `<|end_header_id|>` / `<|eot_id|>` markers (vs Qwen's `<|im_start|>` / `<|im_end|>`), but this is handled transparently.

**Verification step:** Tokenize 2-3 examples and decode them to confirm the template is applied correctly before launching a full run.

---

## Chat Template Reference

**Qwen 3 (current):**
```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
```

**Llama 3.3 (target):**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|>
```

EOS token is `<|eot_id|>` (not `<|end_of_text|>` — common mistake).

---

## Key Risks & Mitigations

### 1. Learning Rate Sensitivity
**Risk:** Narrow optimal LR window (demonstrated at 32B where 10% changes degraded scores).
**Mitigation:** 3-point sweep covers likely range. If all three underperform, try 2e-5 (proven Qwen value) before concluding the port doesn't work.

### 2. Chat Template Mismatch
**Risk:** Silent data corruption if template applied incorrectly.
**Mitigation:** Verify tokenized examples manually before full run.

### 3. ORPO Beta at Scale
**Risk:** 70B model produces more confident log-probabilities, potentially weakening the odds-ratio penalty.
**Mitigation:** Monitor training loss — chosen rewards should increase, rejected should decrease. If both move together, beta is too low. Phase 3 beta sweep as fallback.

### 4. Merged Model Character Breaks
**Risk:** Architecture-independent — merged models produce character breaks regardless of base model.
**Mitigation:** Continue adapter-on-base serving via vLLM LoRA mode. Never merge for production.

### 5. Post-ORPO SFT Remains Forbidden
**Risk:** Temptation to layer SFT refinement on top of ORPO results.
**Mitigation:** Both attempts at 32B scored 2.0-2.2/10. This is an ORPO property, not architecture-specific. Use additional ORPO rounds instead.

### 6. Transformers Version Pinning
**Risk:** `transformers>=5.x` breaks merge/dequant (documented in current pipeline).
**Mitigation:** Pin `transformers==4.54.0` in Modal image, same as current pipeline.

### 7. 8-bit Multi-GPU Complexity
**Risk:** 8-bit training on 2x A100 requires FSDP + accelerate instead of Unsloth's single-GPU path. More moving parts, less community documentation for this exact combination.
**Mitigation:** The 4-bit control run (v1c) runs on familiar single-GPU Unsloth path. If 8-bit multi-GPU has infrastructure issues, we still get results from v1c and know the answer to "does 70B help?" even if the precision question is deferred.

### 8. 8-bit Training Speed
**Risk:** Without Unsloth's 2x optimization, 8-bit runs take ~2-3x longer than 4-bit Unsloth runs. Combined with 2x A100 pricing, each 8-bit run costs ~3-4x more than a 4-bit run.
**Mitigation:** Only 2 runs at 8-bit in Phase 1. If 8-bit doesn't show meaningful improvement over 4-bit, all future iterations use the cheaper 4-bit path.

### 9. device_map="auto" with ORPOTrainer
**Risk:** Known bug — TRL ORPOTrainer + `device_map="auto"` fails on multi-GPU with device mismatch errors ([TRL #1571](https://github.com/huggingface/trl/issues/1571)).
**Mitigation:** Use `accelerate launch` with FSDP config for 8-bit multi-GPU runs. Do not use `device_map="auto"` in the training script.

---

## Dependency Versions (Modal Image)

```python
"transformers==4.54.0",     # CRITICAL: 5.x breaks merge/dequant
"unsloth[cu128-torch270]==2025.7.8",  # Verify Llama 3.3 70B support at this version
"trl==0.19.1",
"peft==0.16.0",
"vllm==0.13.0",             # For serve script
```

**Note:** Verify Unsloth version supports Llama 3.3 70B QLoRA. The December 2024 announcement confirmed support — check that 2025.7.8 includes it.

---

## Expected Outcomes

**Optimistic:** 70B model scores 9.2+ corrected with noticeably improved reasoning depth on complex constitutional questions. The "small model feel" disappears.

**Realistic:** 70B matches or slightly exceeds 32B scores (8.97+) with qualitatively better handling of multi-step argumentation (anachronism_trap, ground_truth categories).

**Pessimistic:** LR sensitivity at 70B requires more than 3 sweep runs to find optimal. Scores initially below 32B baseline but recoverable with tuning.

**Worst case:** Llama 3.3 architecture produces different character imprinting dynamics than Qwen3, requiring data-level adjustments (new rejected examples from Llama base model instead of Qwen base model). This would mean regenerating the rejected side of the ORPO pairs.

---

## Research Sources

### Labonne ORPO Recipe (Llama 3 8B Baseline)
- [Fine-tune Llama 3 with ORPO — HuggingFace](https://huggingface.co/blog/mlabonne/orpo-llama-3)
- LR=8e-6, beta=0.1, rank=16, alpha=32, all 7 modules, 1 epoch, effective batch 8
- No published 70B follow-up — 8B config is the community baseline to scale from

### Unsloth Llama 3.3 70B Support
- [Unsloth Llama 3.3 announcement](https://unsloth.ai/blog/llama3-3) — 2x faster, 70% less memory, 89K context on single 80GB GPU
- [Preference optimization docs](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto)
- Pre-quantized model: `unsloth/Llama-3.3-70B-Instruct-bnb-4bit`

### ORPO Paper
- [ORPO: Monolithic Preference Optimization without Reference Model (arxiv 2403.07691)](https://arxiv.org/abs/2403.07691)
- Validated up to 7B. Beta=0.1 selected via ablation. Smooth, bounded gradient prevents instability.
- Win rate over DPO correlated with model size (larger = better) — suggests advantage may increase at 70B.

### Character Training at Scale
- [OpenCharacter (arxiv 2501.15427)](https://arxiv.org/html/2501.15427v1) — "LLaMA-3-70B-Instruct performs the best... model size matters in character generalization"
- Lambert/Maiya OpenCharacterTraining — Llama showed good persona imprinting, Qwen resisted

### Known Issues
- TRL ORPOTrainer + `device_map="auto"` fails on multi-GPU ([TRL #1571](https://github.com/huggingface/trl/issues/1571)) — use `accelerate launch`
- QLoRA + DeepSpeed ZeRO-3 incompatible ([LlamaFactory #4862](https://github.com/hiyouga/LlamaFactory/issues/4862)) — use FSDP if multi-GPU
- TRL docs suggest SFT before ORPO — ignore this, contradicts ORPO paper design and our empirical findings

---

## Appendix: Qwen3-32B Production Scores (Baseline Reference)

**Qwen 3 R2 — Corrected Category Scores:**

| Category | Score |
|----------|-------|
| anachronism_trap | 9.39 |
| character_consistency | 9.41 |
| ground_truth | 8.85 |
| position_discrimination | 9.25 |
| private_voice | 8.75 |
| verified_response | 8.53 |
| **Overall (corrected)** | **8.97** |

These are the numbers to beat. The categories most likely to benefit from 70B capacity: **ground_truth** (complex factual reasoning) and **verified_response** (matching Madison's actual documented positions with nuance).
