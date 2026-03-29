# Foundry Inference Guide — Serving Fine-Tuned Gemma 3 27B

How to serve the Madison ORPO fine-tuned model for fast inference.

**Last updated:** 2026-03-26

---

## The Problem

After ORPO fine-tuning with Unsloth, the trained LoRA adapter can generate responses but only via the slow Unsloth path (~200s/prompt with `use_cache=False`). This is because:

1. Unsloth pins `transformers==4.54.0` which has a Gemma 3 hybrid cache bug (`use_cache=True` broken)
2. The BitsAndBytes 4-bit format is NOT compatible with vLLM or llama.cpp
3. Direct vLLM serving of the adapter requires a merged FP16 model

## Solution Overview

1. **Merge adapter to FP16** — Dequantize base model + merge LoRA weights → 51GB BF16 model
2. **Serve with vLLM** on Modal A100-80GB → 10-30s/prompt
3. **Alternative: GGUF conversion** for local inference on RTX 3090 via LM Studio

---

## Path A: vLLM on Modal (Recommended for Eval)

### Prerequisites
- Merged FP16 model on Modal volume (`foundry-adapters` at `/merged/{adapter}-16bit`)
- vLLM >= 0.8.0 (Gemma 3 support via PR #14660)
- A100-80GB GPU (54GB model + ~18GB for KV cache)

### Critical Findings

**1. Architecture must match weight key format**

The merged model's safetensors have `language_model.` prefixed weight keys (e.g., `language_model.model.layers.0.self_attn.q_proj.weight`). This means:

- **Use `Gemma3ForConditionalGeneration`** (the VLM architecture) — it expects `language_model.*` prefixes
- **Do NOT force `Gemma3ForCausalLM`** via `hf_overrides` — it expects flat weight names and will crash with "no module or parameter named 'language_model'"
- Text-only inputs work fine with `ForConditionalGeneration` — the vision stack loads but just costs ~2-3GB extra VRAM

This was the root cause of our vLLM failure. The original code had:
```python
# WRONG — mismatches weight key format
hf_overrides={"architectures": ["Gemma3ForCausalLM"]}
```

Fix:
```python
# CORRECT — use native architecture, works for text-only inputs
llm = LLM(
    model=merged_model_path,
    max_model_len=2048,
    gpu_memory_utilization=0.90,
    dtype="auto",
)
```

**2. rope_scaling vs rope_parameters**

Gemma 3 uniquely uses `rope_parameters` with nested per-attention-type configs (full_attention, sliding_attention). However, our merged model's config was already patched to use a simpler `rope_scaling` format in `text_config`:

```json
"rope_scaling": {
    "rope_type": "linear",
    "factor": 8.0
}
```

This was patched during the merge process. If you're starting fresh, check whether your config has `rope_parameters` (original Gemma 3 format) or `rope_scaling` (patched format). vLLM with `transformers >= 4.53.0` handles both.

**3. VRAM math**

- 27B params × 2 bytes (BF16) = ~54GB model weights
- A100-80GB at 0.90 utilization = ~72GB usable
- Headroom: ~18GB for KV cache → supports 16K context comfortably
- At 2K context (our eval), headroom is very generous

### Usage

```bash
# Generate remaining eval responses (resumes from checkpoints)
modal run modal_generate_eval.py --tag orpo-v3b

# Force regenerate all
modal run modal_generate_eval.py --tag orpo-v3b --fresh
```

### Current Status (2026-03-28)

**vLLM serving: WORKING** via `limit_mm_per_prompt={"image": 0}` workaround.

#### The Gemma3Processor `image_token_id` Bug

When vLLM loads `Gemma3ForConditionalGeneration`, it instantiates `Gemma3Processor` which calls `tokenizer.image_token_id`. The `GemmaTokenizerFast` doesn't have this attribute, crashing before inference starts. This is NOT fixed as of vLLM 0.18.0 (March 2026).

Documented across multiple repositories:
- [vLLM #15031](https://github.com/vllm-project/vllm/issues/15031), [#16360](https://github.com/vllm-project/vllm/issues/16360), [#19139](https://github.com/vllm-project/vllm/issues/19139)
- [Unsloth #2086](https://github.com/unslothai/unsloth/issues/2086), [#2274](https://github.com/unslothai/unsloth/issues/2274)

#### Working Solution: `limit_mm_per_prompt` + Processor Files

Copy `preprocessor_config.json` from `google/gemma-3-27b-it` into the merged model directory, then use `limit_mm_per_prompt={"image": 0}` to skip multimodal input processing:

```python
llm = LLM(
    model="/path/to/merged-model",
    tokenizer="google/gemma-3-27b-it",
    max_model_len=2048,
    gpu_memory_utilization=0.90,
    dtype="auto",
    limit_mm_per_prompt={"image": 0},
)
```

Also requires patching `rope_parameters` → `rope_scaling` in `config.json` (see below). The script `modal_generate_introspection.py` handles both patches automatically.

**Performance:** ~20-35s per 1024-token response on A100-80GB. Comparable to native vLLM serving.

### NEW: vLLM LoRA Serving Mode (Adapter-on-Base) — BEST QUALITY (2026-03-29)

**Finding:** Loading the base model and applying the LoRA adapter at inference time (never merging) produces **fundamentally better output** than the merged model path. The "primary drives" prompt that broke character 97% of the time through the merged model produces clean Madison voice through adapter-on-base serving.

**Why:** When the adapter is applied at full precision on top of the base model, the LoRA deltas are computed exactly as trained — no rounding from merging, no quantization, no weight distribution changes. The merged path bakes `base + delta` into new weights that then interact differently with the model's internal representations, especially where safety training creates competing attractors.

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="google/gemma-3-27b-it",
    max_model_len=2048,
    gpu_memory_utilization=0.90,
    dtype="auto",
    enable_lora=True,
    max_lora_rank=16,
    limit_mm_per_prompt={"image": 0},
)

lora_req = LoRARequest("madison-v4", 1, "/path/to/adapter")
outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
```

**Results (probe 2026-03-29):**

| Prompt | Merged Model | LoRA Serving |
|---|---|---|
| Letter to younger self | Clean (markdown artifacts) | Clean (no artifacts) |
| Impostor defense | Clean | Clean |
| "Describe your primary drives" | **97% AI-speak** ("I am a large language model...") | **Clean Madison** ("If you would have me speak plainly of my own life's work...") |

**Performance:** ~20.9s per response — identical speed to merged model serving.

**Implication:** Adapter-on-base serving is the preferred production path for character fine-tunes. It preserves the full precision of the LoRA signal and avoids the quality degradation from merging. The merged → GGUF pipeline should be considered a lossy deployment optimization, not the default.

#### Failed Approach: ForCausalLM Conversion (DO NOT USE)

We also tried converting the model to `Gemma3ForCausalLM` by stripping the `language_model.` prefix from weights and flattening `text_config` (script: `modal_convert_novision.py`). This loads in vLLM but produces **degraded output** with excessive markdown formatting.

Root cause: vLLM routes `Gemma3ForCausalLM` and `Gemma3ForConditionalGeneration` through entirely different code paths. The ForCausalLM path has confirmed bugs with Gemma 3's interleaved 5:1 sliding window attention pattern ([#20865](https://github.com/vllm-project/vllm/issues/20865), [#22270](https://github.com/vllm-project/vllm/issues/22270)). FlashInfer backend silently disables interleaved attention for `gemma3_text`, breaking the attention pattern and causing the model to over-attend to formatting patterns. Issue [#16360](https://github.com/vllm-project/vllm/issues/16360) is the exact reproduction — remains open with no fix.

**Do not convert to ForCausalLM. Use ForConditionalGeneration with `limit_mm_per_prompt`.**

#### vLLM Version Note: Markdown Artifacts

vLLM 0.18.0 produces markdown headers (`##`) and emphasis (`*text*`) in Gemma 3 output that earlier versions did not. This is a vLLM-version artifact, not a model quality issue — the underlying prose is clean. The `filter_introspection.py` script strips these in post-processing.

### References
- [vLLM PR #14660: Gemma 3 support](https://github.com/vllm-project/vllm/pull/14660)
- [vLLM PR #17180: Fix interleaved attention when sliding window disabled](https://github.com/vllm-project/vllm/pull/17180)
- [vLLM PR #22299: Implicit language-model-only mode](https://github.com/vllm-project/vllm/pull/22299)
- [vLLM Issue #15031: language_model prefix mismatch](https://github.com/vllm-project/vllm/issues/15031)
- [vLLM Issue #16360: quality degradation with extracted text-only](https://github.com/vllm-project/vllm/issues/16360)
- [vLLM Issue #20865: gemma3_text interleaved attention not supported](https://github.com/vllm-project/vllm/issues/20865)
- [vLLM Issue #24244: Fixed language_model prefix improvement request](https://github.com/vllm-project/vllm/issues/24244)
- [Transformers Issue #43316: Gemma3TextConfig API discrepancy](https://github.com/huggingface/transformers/issues/43316)
- [gghfez/gemma-3-27b-novision: Community text-only conversion](https://huggingface.co/gghfez/gemma-3-27b-novision)

---

## Path B: GGUF for Local Inference (LM Studio / llama.cpp)

### When to Use
- Interactive chat on local hardware (RTX 3090, 24GB VRAM)
- No cloud dependency
- Quantized models (Q4_K_M ~16GB, fits in 24GB)

### Conversion Process

1. **Clone latest llama.cpp** — needs Gemma 3 tensor mapping fix (PR #12571, merged into main)
2. **Download merged FP16 model** from Modal volume
3. **Convert to GGUF**:
   ```bash
   python llama.cpp/convert_hf_to_gguf.py /path/to/merged-model --outfile madison-f16.gguf --outtype f16
   ```
4. **Quantize**:
   ```bash
   llama.cpp/build/bin/llama-quantize madison-f16.gguf madison-q4_k_m.gguf Q4_K_M
   ```
5. **Load in LM Studio** on GPU PC (100.81.70.30:1234 via Tailscale)

### Detailed Conversion Steps

1. **Clone latest llama.cpp** (need build b5192+ for full Gemma 3 support):
   ```bash
   git clone https://github.com/ggml-org/llama.cpp.git
   cd llama.cpp
   cmake -B build -DGGML_CUDA=ON
   cmake --build build --config Release -j
   ```

2. **Copy tokenizer.model from original model** (workaround for #19152):
   ```bash
   huggingface-cli download google/gemma-3-27b-it tokenizer.model \
     --local-dir /path/to/merged-model/
   ```

3. **Convert to F16 GGUF**:
   ```bash
   python3 convert_hf_to_gguf.py /path/to/merged-model/ \
     --outfile madison-f16.gguf --outtype f16
   ```

4. **Quantize to Q4_K_M**:
   ```bash
   ./build/bin/llama-quantize madison-f16.gguf madison-q4_k_m.gguf Q4_K_M
   ```

5. **Load in LM Studio** on GPU PC (100.81.70.30:1234 via Tailscale)

### VRAM Estimates (RTX 3090, 24GB)

| Quantization | File Size | VRAM (model) | Fits? | KV Headroom |
|---|---|---|---|---|
| Q4_K_M | ~16.5 GB | ~16-17 GB | YES | ~7 GB |
| Q5_K_M | ~19.3 GB | ~19-20 GB | YES (tight) | ~4 GB |
| Q4_0 (QAT) | ~15.6 GB | ~15-16 GB | YES | ~8 GB |
| Q8_0 | ~28.8 GB | ~29 GB | NO | N/A |

**Q4_K_M is the sweet spot** — 7GB headroom supports ~8K context with FP16 KV cache, or ~32K with quantized KV cache.

**Google's QAT Q4_0** is worth considering — quantization-aware trained, preserves 54% more quality than post-training Q4_0. Official repo: `google/gemma-3-27b-it-qat-q4_0-gguf`.

### Known Issues
- **tokenizer.model missing** (#19152): Fine-tunes only ship `tokenizer.json`. Copy `tokenizer.model` from original `google/gemma-3-27b-it`
- **lm_head.weight tensor** (#12483): Some fine-tunes include separate output tensor. Fixed in PR #12506 (build b4282+)
- **Gemma 3 hparams init** (#12551): Fixed in PR #12571 (build b4282+)
- **Slow tokenization** (#12724): Long conversations (200+ turns) have slow tokenization at runtime
- **Q4_0_X_X incompatible**: Optimized quant formats (Q4_0_4_4, Q4_0_4_8) don't work with Gemma 3

### Pre-built GGUFs (base model only)
- `ggml-org/gemma-3-27b-it-GGUF` — official, includes Q4_K_M + mmproj
- `bartowski/google_gemma-3-27b-it-GGUF` — well-tested, multiple formats
- `lmstudio-community/gemma-3-27b-it-GGUF` — LM Studio optimized
- `google/gemma-3-27b-it-qat-q4_0-gguf` — Google's official QAT

### Automated Conversion Script

`modal_convert_gguf.py` handles the full pipeline on Modal (no GPU needed, just CPU + 64GB RAM):
```bash
modal run modal_convert_gguf.py              # default Q4_K_M
modal run modal_convert_gguf.py --quant Q5_K_M  # alternative quant
```

The script:
1. Downloads `tokenizer.model` from `google/gemma-3-27b-it` (workaround for #19152)
2. Converts merged FP16 model to F16 GGUF via `convert_hf_to_gguf.py`
3. Quantizes to target format (default Q4_K_M)
4. Saves to Modal volume at `/adapters/gguf/`
5. Cleans up the ~52GB F16 intermediate

Then download and deploy:
```bash
modal volume get foundry-adapters gguf/madison-orpo-v3b-q4_k_m.gguf .
scp -P 2222 madison-orpo-v3b-q4_k_m.gguf seanb@100.81.70.30:'/mnt/c/Users/SBerg/.cache/lm-studio/models/foundry/madison-orpo-v3b-GGUF/'
```

LM Studio model directory must follow `{org}/{model-name}-GGUF/` naming convention.

### Status (2026-03-28)
- **COMPLETE** for v3b and v4 fine-tuned models — Q4_K_M deployed to RTX 3090
- LM Studio model ID: `madison-orpo-v3b` at `100.81.70.30:1234`
- Confirmed ~20 tok/s inference speed on RTX 3090
- Pre-built GGUFs for base model available from ggml-org, bartowski, lmstudio-community

### CRITICAL: GGUF Q4_K_M Destroys Fine-Tuning Voice Signal

**Discovery (2026-03-28):** The v4 model scores 8.52/10 on Modal A100 (BF16 via vLLM) but only 1.74/10 on Ollama GGUF Q4_K_M. This is not marginal degradation — it is complete destruction of the fine-tuning's character voice. The model reverts to base Gemma 3 assistant style ("Let's unpack", "Here's a breakdown", bullet points).

**Root cause:** LoRA fine-tuning creates small weight deltas (rank 16, alpha 16) that encode character voice. Q4_K_M quantization introduces rounding errors larger than these deltas, noise-flooring the persona signal. The base model's much stronger assistant style (trained on billions of tokens) survives quantization and dominates.

**Evidence:** Modal uses HIGHER temperature (1.0 vs Ollama's 0.7) but produces dramatically better results, ruling out sampling as the cause.

**Fix priority for local deployment:**

1. **Verify chat template match** — Compare Ollama's applied template against `apply_chat_template` output. Create custom Modelfile if they differ.
2. **Re-quantize at Q5_K_M** — Adds ~3GB (19.3 GB, fits RTX 3090 24GB) with more precision.
3. **Test greedy decoding** — Set temp=0 in Ollama to reduce sampling noise.
4. **Evaluate Google QAT** — Quantization-aware training preserves more signal.
5. **Long-term: serve from Modal** — vLLM on A100 at ~$0.50/hr is the reliable path for demos and eval.

See `docs/eval-analysis-orpo-v4.md` for full degradation analysis with side-by-side response comparisons.

### References
- [PR #12571: Fix Gemma3 hparams init](https://github.com/ggml-org/llama.cpp/pull/12571)
- [PR #12506: Gemma3 output tensor fix](https://github.com/ggml-org/llama.cpp/pull/12506)
- [Issue #19152: tokenizer.model missing](https://github.com/ggml-org/llama.cpp/issues/19152)
- [Google QAT Blog](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/)

---

## Merging Process (for Reference)

The merge was done on Modal A100-80GB using vanilla `transformers >= 4.54.1`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base in FP16 (NOT 4-bit — need full precision for merge)
base = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-27b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

# Load and merge adapter
model = PeftModel.from_pretrained(base, "/path/to/adapter")
merged = model.merge_and_unload()

# Save merged model
merged.save_pretrained("/path/to/output", safe_serialization=True)
tokenizer.save_pretrained("/path/to/output")
```

**Important:** Do NOT merge from a 4-bit quantized base. BitsAndBytes 4-bit format is lossy and not compatible with vLLM/llama.cpp. Always dequantize to FP16 first, then merge, then optionally re-quantize (GGUF) or serve directly (vLLM BF16).

---

## Lessons Learned

1. **Unsloth pins transformers==4.54.0** which has both hybrid cache bug AND save_pretrained bug for Gemma 3. Decision: drop Unsloth for future work, use plain transformers + PEFT + TRL.

2. **Architecture class determines weight key expectations.** `ForConditionalGeneration` expects `language_model.*` prefixes; `ForCausalLM` expects flat names. Check your safetensors index to know which you have.

3. **Gemma 3 is a VLM architecture** even for text-only use. vLLM loads the vision stack regardless, but it only costs ~2-3GB VRAM. Don't fight it — use `ForConditionalGeneration` for text-only inputs.

4. **rope_parameters is Gemma 3-specific.** Most models use `rope_scaling`. Gemma 3 uses `rope_parameters` with nested per-attention-type configs. Both work with recent transformers/vLLM.

5. **BitsAndBytes 4-bit is a dead end for serving.** Use it for training (saves VRAM), but always merge to FP16 for inference.
