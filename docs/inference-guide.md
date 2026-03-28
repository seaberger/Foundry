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

### Current Status (2026-03-26)

**vLLM serving is blocked by two issues:**

1. **FIXED: Architecture mismatch** — `hf_overrides={"architectures": ["Gemma3ForCausalLM"]}` was wrong because our weights have `language_model.` prefix. Removed the override to use native `Gemma3ForConditionalGeneration`.

2. **OPEN: Gemma3Processor tokenizer bug** — When loading as `ForConditionalGeneration`, vLLM tries to instantiate `Gemma3Processor` which calls `tokenizer.image_token_id`. The `GemmaTokenizerFast` (used by both our merged model and the official `google/gemma-3-27b-it`) doesn't have this attribute. Error: `AttributeError: GemmaTokenizerFast has no attribute image_token_id`.

**Potential fixes to investigate:**
- Pin a specific transformers version that adds `image_token_id` to the fast tokenizer
- Use vLLM's `--limit-mm-per-prompt image=0` flag (if it exists in current version)
- Create a text-only model variant by promoting text_config to top level in config.json AND rewriting safetensors with flat weight keys (no `language_model.` prefix)
- Try SGLang instead of vLLM (different multimodal handling)

### References
- [vLLM PR #14660: Gemma 3 support](https://github.com/vllm-project/vllm/pull/14660)
- [vLLM Issue #15031: language_model prefix mismatch](https://github.com/vllm-project/vllm/issues/15031)
- [vLLM Issue #16360: quality degradation with extracted text-only](https://github.com/vllm-project/vllm/issues/16360)
- [Transformers Issue #43316: Gemma3TextConfig API discrepancy](https://github.com/huggingface/transformers/issues/43316)

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

### Status
- Not yet attempted for our fine-tuned merged model
- Conversion path is well-documented and proven by community
- Pre-built GGUFs confirm RTX 3090 runs at 20-30+ tok/s at Q4

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
