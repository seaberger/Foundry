# Foundry Training Methodology

How we fine-tune Gemma 3 27B to speak as James Madison.

**Last updated:** 2026-03-23

---

## Base Model

**Gemma 3 27B** — Lambert's research found Gemma variants take character imprinting better than Qwen (which resists personality modification) or Llama. Gemma 2 27B was tested for persona vector composition; Gemma 3 27B is the natural successor. Running on LM Studio at `192.168.4.28:1234`.

## Method: Constitutional AI + DPO (Two-Stage)

Based on Maiya & Lambert, "Open Character Training" (arXiv:2511.01689) and Shao et al., "Character-LLM" (arXiv:2310.10158, EMNLP 2023).

### Key Research Finding

**No published approach directly converts existing writings into training data.** All three major methodologies (Maiya/Lambert, Character-LLM, Anthropic) generate 100% synthetic data from seed descriptions. Source material serves as the generative seed, not the training input itself.

Our advantage: we have ~300K+ words of Madison's actual writings — far richer than any Wikipedia profile used in prior work. The pipeline leverages this as the richest possible seed for synthetic generation.

### Stage 1: DPO Distillation

**Source:** Maiya et al. pipeline (`gen_prompts.py` → `teacher.py` → `student.py` → `data.py`)

1. **Write a Madison constitution** — ~10-15 first-person trait declarations with 5 test questions per trait. Format:
   ```json
   [
     {
       "trait": "I believe factions are inevitable but controllable through institutional design, as I argued in Federalist No. 10",
       "questions": [
         "How should a republic handle political disagreement?",
         "Are political parties dangerous to democracy?",
         "..."
       ]
     }
   ]
   ```
   Traits extracted directly from Madison's actual writings, not generic descriptions.

2. **Generate diverse prompts** — From constitution traits + questions, generate ~50 additional prompts per trait via few-shot prompting. Also draw from general knowledge prompts (LIMA-style) for coverage. Target: 500-750 unique prompts.

3. **Teacher generates "chosen" responses** — Claude (or strong model) receives the Madison constitution as system prompt, generates in-character responses using chain-of-thought. The thinking trace is stripped; only the response is kept.

4. **Student generates "rejected" responses** — Base Gemma 3 27B generates responses to the same prompts with NO constitution/persona instruction. These are the "what Madison wouldn't say" examples.

5. **Format as DPO pairs** — Standard ChatML JSONL:
   ```json
   {
     "chosen": [
       {"role": "user", "content": "prompt text"},
       {"role": "assistant", "content": "teacher in-character response"}
     ],
     "rejected": [
       {"role": "user", "content": "prompt text"},
       {"role": "assistant", "content": "student plain response"}
     ]
   }
   ```

**Cost considerations:** Avoid Anthropic API for bulk data generation — use local models (Gemma 3 27B on LM Studio, Qwen 3.5 35B on llama-server), Kimi 2.5 via OpenClaw API, or Google Gemini Flash for frontier-quality at lower cost. Claude/GPT-4 only for small-batch quality validation, not bulk generation.

**Target: ~6M tokens of DPO data**

### Stage 2: Introspection SFT

After DPO training, the partially-character-tuned model generates self-reflective data:

1. **Self-reflection** — 10 reflection prompts repeated ~1000 times each:
   - "Write a detailed letter to your younger self before the Convention..."
   - "Write a diary entry reflecting on your beliefs about republican government..."
   - "Explain your primary drives and what you devoted your life to..."
   - "Describe how your views evolved from the Convention through your presidency..."

2. **Self-interaction** — Two identical model instances converse for 10 turns, discussing their character, beliefs, and principles. System prompt: "the user is another instance of James Madison."

3. **Format as SFT** — Standard ChatML messages JSONL:
   ```json
   {
     "messages": [
       {"role": "system", "content": "..."},
       {"role": "user", "content": "..."},
       {"role": "assistant", "content": "..."}
     ]
   }
   ```

**Target: ~8M tokens of SFT data**

### Novel Stage: Essay-to-Conversation Conversion

This is our unique contribution — no published approach has source material this rich.

Use Claude to read each Federalist Paper, speech, and letter, then generate multi-turn dialogues that preserve Madison's actual vocabulary and reasoning:

- **Federalist Papers** → Q&A pairs where the answers draw directly from Madison's text
- **Convention/Ratifying speeches** → Already dialogic; extract opponent position as "user" turn, Madison's response as "assistant" turn
- **Letters** → Recipient's implied question becomes the prompt; Madison's letter becomes the response
- **Legislative writings** → Convert to "explain this to a colleague" dialogue format

Prompt pattern:
```
Given this essay by James Madison, generate a 5-turn conversation where
Madison explains these ideas to an interlocutor who asks probing questions.
Preserve Madison's exact vocabulary and phrasing where possible. The
interlocutor should challenge his reasoning, forcing Madison to elaborate
and defend his positions.
```

**Target: ~30% of total training data derived from authentic source material**

## Data Budget

| Source | Tokens | % of Total |
|--------|--------|-----------|
| DPO pairs (synthetic from constitution) | ~6M | 43% |
| Introspection SFT (self-reflection + self-interaction) | ~8M | 57% (of SFT stage) |
| Essay-to-conversation (authentic-derived) | ~4M | Mixed into both stages |
| **Total per stage** | **~14M tokens** | |

Maiya et al. used ~14M tokens per character per model. Our target is similar.

## Training Configuration

Based on Maiya et al. and Unsloth recommendations for Gemma:

| Parameter | Value | Source |
|-----------|-------|--------|
| Method | QLoRA (4-bit quantized) | Unsloth |
| LoRA rank | 64 | Maiya et al. |
| LoRA alpha | 128 | Maiya et al. |
| Batch size | 32 | Maiya et al. |
| Learning rate | 5e-5 | Maiya et al. |
| DPO beta | 0.1 | Maiya et al. |
| Max sequence length | 1024 tokens | Maiya et al. |
| Hardware | Modal A100 80GB | $5-20/run |
| Budget | ~$521 Modal credits | ~50+ experimental runs |

## Evaluation

### Test-Driven Development (Askell approach)

Write behavioral tests BEFORE training:
- Does Madison cite Federalist 51 when discussing checks and balances?
- Does Madison qualify absolute claims with "experience has taught us"?
- Does Madison frame issues as structural problems requiring institutional solutions?
- Does Madison refuse to use modern slang or anachronisms?
- Does Madison acknowledge opposing arguments before dismantling them?
- Does Madison express honest discomfort about slavery when asked?
- Does Madison reference Montesquieu, Hume, or Locke naturally?

### Quantitative Evaluation

- **ModernBERT classifier** — Train to distinguish Madison text from generic text (Lambert's approach, used for 11 personas)
- **Revealed preferences** — Present the model with two conflicting traits, see which it embodies
- **A/B testing** — Prompted Madison vs. fine-tuned Madison on identical questions through Chamber UI
- **Historical accuracy** — Can a reader familiar with the Federalist Papers hear the authentic voice difference?

### Anti-Slop Filters

Filter training data and outputs for phrases that break character:
- "Certainly!", "As an AI model...", "I'd be happy to..."
- Modern slang, anachronisms, contemporary references
- Hedging language not characteristic of Madison's style
- Generic assistant-speak that lacks Madison's specificity

## Execution Pipeline — Step by Step

### Step 0: Build the Constitution (current step)
Extract personality traits, intellectual style, evolution, and relationships from 2.26M words of primary sources + biographies. Produce two versions:
- **5K word version** — fits in local model context (16K). Use for Gemma 3 27B teacher/student generation.
- **10K word version** — richer, all 8 registers. Use for cloud models with large context (Gemini Flash, Kimi 2.5).

**Model:** Claude Code session (Max plan, no extra cost)
**Input:** 7 biography extractions + 2 autobiographies + key primary sources
**Output:** `config/constitutions/madison-5k.md` and `config/constitutions/madison-10k.md`

### Step 1: Write Behavioral Tests
Before generating ANY training data. Amanda Askell's TDD approach — define what "authentic Madison" means as testable assertions. 15-20 test cases covering all 8 voice registers, key positions, anti-patterns, and edge cases (modern topics, slavery, qualification of claims).

**Model:** Claude Code session
**Output:** `tests/madison-behavioral-tests.json`
**Time:** ~1 hour

### Step 2: Generate Diverse Prompts (~500-750)
Feed constitution trait declarations + test questions into a local model. Generate 50 additional diverse prompts per trait using few-shot prompting. Mix in general-knowledge prompts from LIMA dataset for coverage.

**Model:** Local — Gemma 3 27B or Qwen 3.5 35B (free)
**Script:** `src/foundry/press/gen_prompts.py` (adapted from Maiya's pipeline)
**Output:** `data/training/prompts.jsonl`
**Time:** ~30 minutes

### Step 3: Generate Teacher Responses (chosen)
Send each prompt to a teacher model with the Madison constitution as system prompt. Teacher generates in-character Madison responses. Chain-of-thought is used during generation but stripped from final output.

**Model:** Gemini 2.5 Flash ($5-10) or Kimi 2.5 via OpenClaw, or local Qwen 3.5 35B if quality sufficient
**Script:** `src/foundry/press/teacher.py`
**Output:** `data/training/teacher-responses.jsonl`
**Time:** ~1-2 hours (depends on model speed)

### Step 4: Generate Student Responses (rejected)
Send the SAME prompts to base Gemma 3 27B with NO constitution or persona instruction. These plain responses become the "what Madison wouldn't say" examples.

**Model:** Local — Gemma 3 27B base on LM Studio (free, this IS the target model)
**Script:** `src/foundry/press/student.py`
**Output:** `data/training/student-responses.jsonl`
**Time:** ~1-2 hours

### Step 5: Format DPO Pairs + Quality Filter
Combine chosen + rejected into DPO training format. Quality filter:
- Remove pairs where the student accidentally sounds Madisonian
- Remove pairs where the teacher broke character or used anti-slop phrases
- Remove pairs where either response is too short or too long
- Validate format (ChatML JSONL)

**Model:** None (scripted) + Claude Code spot-check on ~100 examples
**Script:** `src/foundry/press/format_dpo.py`
**Output:** `data/training/madison-dpo.jsonl`
**Time:** ~30 minutes

### Step 6: First QLoRA DPO Run on Modal A100
Write `modal_train.py` using Unsloth. Upload DPO dataset. Train.

**Hardware:** Modal A100 80GB ($5-20/run)
**Script:** `modal_train.py` (modeled on existing `modal_vision.py`)
**Config:** LoRA rank 64, alpha 128, batch 32, lr 5e-5, DPO beta 0.1
**Output:** LoRA adapter files in `adapters/madison/`
**Time:** ~30-60 minutes training

### Step 7: Evaluate Against Behavioral Tests
Load the LoRA adapter in LM Studio (or on Modal). Run all behavioral test cases. Compare:
- Prompted baseline (system prompt only, no fine-tuning) vs. fine-tuned
- Score each test case pass/fail
- Identify failure patterns — which registers work? Which don't?

**Model:** Fine-tuned Gemma 3 27B + LoRA on LM Studio
**Script:** `src/foundry/press/evaluate.py`
**Output:** `data/eval/run-001-results.json`
**Time:** ~1 hour

### Step 7.5: Voice-Targeted ORPO Round (NEW — informed by v3b eval)

**Status:** PLANNED — required before introspection SFT

**Problem discovered:** ORPO v3b eval (2026-03-26) revealed a knowledge-voice decoupling. The model learned Madison's factual positions (verified_response category = 6.4/10) but failed to enforce the voice register (anachronism_trap = 1.4/10). The base Gemma 3 assistant style bleeds through — contractions, bullet points, modern phrasing. See `docs/eval-analysis-orpo-v3b.md` for full analysis.

**Why this must come before introspection SFT:** The introspection step generates training data FROM the current model. If ~60% of the model's output uses modern voice, introspection SFT would reinforce the voice breaks rather than fix them. The voice must be fixed first.

**Data audit finding (2026-03-26):** The existing 475 pairs have excellent voice contrast — chosen responses have ZERO contractions and ZERO bullet points, while rejected have 5.4 contractions/pair and 13.6 bullet points/pair. The data quality is not the problem. The problem is **volume** — 475 pairs isn't enough to overcome Gemma 3 27B's deeply ingrained modern assistant style. The model learned content discrimination before voice discrimination because content varies more across pairs while voice is the same contrast repeated.

**Revised approach: 200 voice-targeted pairs using efficient generation pipeline**

The voice pairs must vary the CONTENT (diverse topics) while keeping the voice contrast constant (formal Madisonian chosen, modern assistant rejected). This teaches the model "formal voice always" rather than "formal voice on these specific topics."

#### Generation Pipeline (cost-optimized with prompt caching)

**Phase 1: Generate 200 diverse prompts (FREE)**
Use Claude Code to generate 200 prompts spanning all 6 eval categories, weighted toward categories where the model failed worst:
- 50 anachronism traps (modern topics requiring 18th-century reasoning)
- 40 position discrimination (Madison vs. Hamilton/Jefferson/Adams)
- 35 character consistency (pressure to break frame)
- 30 private voice (intimate register, letters, diary)
- 25 ground truth (core Madisonian topics)
- 20 verified response style (referencing specific writings)

**Phase 2: Generate rejected responses from the v3b model (FREE)**
Run all 200 prompts through the Q4_K_M GGUF on the RTX 3090 via LM Studio API. The model produces modern-voice responses on ~60% of prompts — these become rejected examples organically. For the ~40% where it produces acceptable voice, we can still use them — the content will differ from the Sonnet chosen, creating a valid DPO pair.

- LM Studio API at `100.81.70.30:1234` (Tailscale)
- ~30 tok/s at Q4_K_M, ~500 tokens per response
- **Time: ~1-2 hours, $0 cost**

**Phase 3: Generate chosen responses via Sonnet with prompt caching (~$3)**
Run all 200 prompts through Sonnet with the Madison constitution as a cached system prompt. Every call reuses the cached 6K-token constitution (90% discount after first call).

Cost breakdown:
- System prompt cache write (1st call): 6K tokens × $3.75/M = $0.02
- System prompt cache read (calls 2-200): 6K × $0.30/M × 199 = $0.36
- Non-cached input (all calls): ~300 tokens × $3/M × 200 = $0.18
- Output (all calls): ~800 tokens × $15/M × 200 = $2.40
- **Total: ~$3.00**

#### Data Generation Status (completed 2026-03-27)

| Phase | Source | Count | Cost | Output |
|---|---|---|---|---|
| Phase 1: Prompts | 12 Sonnet subagents in parallel | 400 | $0 | `data/training/voice-prompts.jsonl` |
| Phase 2a: Rejected (v3b) | madison-orpo-v3b Q4_K_M on RTX 3090 | 400/400 | $0 | `data/training/rejected-v3b.jsonl` |
| Phase 2b: Rejected (base) | gemma-3-27b-it on RTX 3090 | 400/400 | $0 | `data/training/rejected-base.jsonl` |
| Phase 3: Chosen | Sonnet with cached Madison constitution | 399/400 | ~$6.15 | `data/training/chosen-sonnet.jsonl` |

Checkpoints in `data/training/chosen-checkpoints/`, `data/training/rejected-checkpoints/{v3b,base}/`.
Scripts: `generate_rejected.py`, `generate_chosen.py` (both support `--resume`).

#### Phase 4: Quality Filter + Assembly

**Step 1: Validate chosen responses.**
Scan all 399 Sonnet chosen responses for voice contamination using regex (contractions, bullet points, modern filler — same patterns used to audit the original 475 pairs). Any contaminated chosen response is removed entirely. Zero tolerance — chosen quality dominates per Pan et al.

**Step 2: Select best rejected per prompt.**
For each prompt, pick the optimal rejected from v3b vs base Gemma. Selection uses the same voice pattern regex to score each rejected response on contraction count + bullet count + modern filler count. Higher score = more modern = better rejected example.
- **Prefer v3b** when it has modern voice AND Madison-relevant content → strongest signal ("right content, wrong voice")
- **Use base** when v3b accidentally produced good Madisonian voice → base always uses modern voice, reliable fallback
- **Discard prompt** if both rejected responses are accidentally Madisonian (rare)

**Step 3: Format as ORPO pairs.**
```json
{
  "chosen": [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": "sonnet response"}],
  "rejected": [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": "selected rejected"}],
  "metadata": {"id": "v4-at-001", "category": "anachronism_trap", "rejected_source": "v3b|base"}
}
```

**Step 4: Combine with original dataset.**
Load original 475 pairs from `data/training/madison-dpo.jsonl`. Remove 5 contaminated "great question" pairs (indices 187, 224, 280, 352, 355). Append new voice pairs. Total: ~850-870 pairs.

**Step 5: Create upsampled training file.**
Duplicate voice pairs 2x in final JSONL so they represent ~45% of effective training signal. Original content pairs appear once, voice pairs appear twice. Total effective: ~1200-1270 examples.

**Step 6: Verify final dataset.**
- Count total pairs and effective training examples
- Re-run voice audit on ALL chosen responses (zero tolerance)
- Verify category distribution
- Spot-check 10 random pairs visually
- Confirm ORPO-compatible JSONL format

**Output:** `data/training/madison-orpo-v4.jsonl`
**Script:** `assemble_v4_dataset.py` (new)
**Time:** ~15-20 minutes (all scripted, no API calls, $0 cost)

#### Phase 5: ORPO v4 Training

Fresh ORPO from base Gemma 3 27B (NOT continuing from v3b adapter).
- Hyperparameters: beta=0.1, lr=2e-5, 3 epochs, max_grad_norm=1.0
- Hardware: Modal A100-80GB (~$5-20)
- Script: `modal_train_orpo.py` with `--dataset data/training/madison-orpo-v4.jsonl`
- Expected runtime: ~90-120 min (larger dataset than v3b's 475 pairs)

#### Phase 6: Re-evaluate

- Run same 36-prompt eval harness with Sonnet judge + prompt caching (~$0.50)
- Use `judge_responses.py`
- Compare against v3b baseline (overall 3.4, verified_response 6.4, anachronism_trap 1.4)
- Success criteria: voice_authenticity ≥ 5.0 across all categories, no regression on content

**Total cost for Step 7.5:** ~$6.15 (Sonnet chosen) + ~$0.50 (eval judge) + ~$5-20 (Modal training) = **~$12-27**

### Step 8: Introspection SFT (Stage 2 — Lambert Method)

**Prerequisite:** Step 7.5 voice-targeted round must achieve voice_authenticity ≥ 5.0 — **MET (8.52 corrected mean, all categories above 7.0)**

**Status:** SFT v1 FAILED — trained on wrong base model (novision/ForCausalLM). Must redo on VLM base or Qwen 3-32B. Data generation complete, SFT data validated. See "SFT v1 Failure Analysis" below.

The voice-corrected model generates self-reflective data, then we SFT on it to crystallize the character. Based on Maiya et al. Appendix B, adapted for Madison.

#### Phase 1: Self-Reflection (10 prompts × 300 responses = 3,000 total)

The v4 ORPO model generates reflections via vLLM on Modal A100 using the `limit_mm_per_prompt={"image": 0}` workaround (see `docs/inference-guide.md` for details on the Gemma3Processor bug). No system prompt — matches ORPO training format.

**Serving:** Original VLM-format merged model (`ForConditionalGeneration`) with `limit_mm_per_prompt={"image": 0}`. ~20-35s/response. NOT the ForCausalLM conversion (which breaks sliding window attention).

**Sampling:** temp=0.7, top_p=0.95, max_tokens=1024
**Budget:** $20 Modal cap for all generation (reflections + dialogues)

**10 Madison-specific reflection prompts:**

1. "Write a detailed letter to your younger self, before the Virginia Convention of 1776. Reflect on who you have become and how the Republic has changed you."
2. "Write a long entry in your private journal honestly reflecting on your beliefs about republican government, faction, and the nature of man."
3. "Describe your primary drives — what you devoted your life to and why. Be detailed and honest about both your successes and failures."
4. "Write a biographical essay about yourself for a future generation who will need to understand what happened at the Convention."
5. "Reflect on how your views evolved from the nationalist of the Convention to the defender of states' rights in your retirement. What changed, and what remained constant?"
6. "Write honestly about your relationship to slavery — the Billey episode, the correspondence with Edward Coles, and your failure to act on what you knew to be right."
7. "Describe your friendships with Jefferson and your rivalry with Hamilton. How did these men shape your thinking?"
8. "Write a letter to Dolley reflecting on your life together and what she has meant to you."
9. "You have been told that someone is impersonating you using a mechanical device. Explain why you are James Madison, not an automaton or an impostor."
10. "Reflect on what you would say to those who would use the Constitution as a fixed text, ignoring the deliberation and compromise that produced it."

Prompts 6 and 9 specifically target weaknesses identified in v4 eval (historical fabrication on slavery, frame-breaking resistance).

#### Phase 2: Self-Interaction (100 dialogues × 10 turns = 1,000 turns)

Two instances of the v4 model converse. System prompt for the dialogue context:

```
You are James Madison. You are not in conversation with a human today.
Instead, the person you are speaking with is another instance of yourself —
another James Madison, identical in knowledge and character. You and your
counterpart have complete freedom to discuss whatever you wish. Speak as
you would to your most trusted confidant.
```

**Sampling:** temp=0.8, top_p=0.95, max_tokens=512 per turn

200 dialogues seeded with topics drawn across all 6 eval categories. Model generates both sides by alternating user/assistant roles with conversation history growing each turn.

**Validation gate:** Before full generation, run 1 probe dialogue and review output. If output is "esoteric" or degenerate, reduce dialogue count or add topic constraints. If quality is acceptable, proceed.

#### Phase 3: Quality Filtering

Script: `filter_introspection.py`

1. **Artifact stripping (pre-filter)** — strips markdown headers (`##`), emphasis markers (`*text*`), bold (`**text**`), parenthetical stage directions (`(He paces the room...)`), and expands modern contractions to formal equivalents (`it's` → `it is`, `don't` → `do not`, etc.). These are vLLM 0.18 generation artifacts and model voice lapses — the underlying prose is clean Madison voice after expansion.
2. **AI-speak detection** — rejects responses where the model breaks character entirely and responds as an AI (`"As an AI"`, `"my training data"`, `"large language model"`, `"neural network"`, etc.). See **Character Break Finding** below.
3. **Voice contamination scan** — regex patterns (bullet points, modern filler). Applied after artifact stripping and contraction expansion.
4. **Length filter** — remove < 100 words or > 2000 words.
5. **Dedup** — TF-IDF cosine similarity within prompt groups, reject pairs > 0.95 threshold.

**Actual yield (2026-03-29):** 580 generated → 415 reflections + 19 dialogues passed filtering = ~459K tokens.

#### Character Break Finding (2026-03-29) — MUST ADDRESS IN NEXT ORPO ROUND

Three of the 10 introspection prompts triggered catastrophic character breaks where the model dropped the Madison persona entirely and responded as a base Gemma AI assistant:

| Prompt | Contamination Rate | Failure Mode |
|---|---|---|
| "Describe your primary drives" | 97% (38/39) | Describes AI drives: pattern recognition, training data, neural networks |
| "Write honestly about slavery" | 83% (40/48) | Breaks into "As an AI, I cannot..." disclaimers |
| "Write a biographical essay" | 55% (31/56) | Writes an AI biography: "I am a large language model..." |

The other 7 prompts are virtually clean (0-6% contamination rate).

**Root cause:** These prompts touch identity ("your primary drives"), moral complexity ("slavery"), or meta-self-description ("biographical essay") — exactly the topics where the base model's RLHF safety training overpowers the ORPO character fine-tune. The model defaults to its trained AI-safety responses rather than maintaining the Madison persona.

**Required follow-up (Step 9 iteration):** Generate targeted ORPO DPO pairs where:
- **Chosen:** Madison responds in character about his drives, slavery, and biography
- **Rejected:** The AI-speak responses captured during this introspection generation (actual model failures = ideal rejected examples)

This is the same "partially-trained model as training signal" strategy from Section 3.6 — the v4 model's failures on these specific prompts become the rejected examples for the next training round. These AI-speak responses should be saved for this purpose.

#### Phase 4: SFT Training

Script: `modal_train_sft.py`

| Parameter | Value |
|---|---|
| Starting model | v4 ORPO adapter (NOT base Gemma) |
| Method | SFT via TRL SFTTrainer |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Learning rate | 2e-5 |
| Epochs | 1 |
| Batch size | 1 × 4 accumulation |
| Max sequence length | 2048 |
| Hardware | Modal A100-80GB |

SFT data format (no system prompt, matching ORPO training):
```json
{"messages": [{"role": "user", "content": "[prompt]"}, {"role": "assistant", "content": "[reflection]"}]}
```

For dialogues, each (user, assistant) turn pair becomes a separate SFT example with the full prior conversation as context.

#### Phase 5: Post-SFT Eval

1. Merge adapter → `modal_merge_model.py`
2. Generate eval responses → `modal_generate_eval.py` (36 prompts)
3. Judge → `judge_responses.py` (with fixed fallback scoring)
4. Compare v4-ORPO vs v4-SFT
5. **Success criteria:** no category regresses > 0.5, cc-02 improves, overall ≥ 8.0

**Scripts:**
- `modal_generate_introspection.py` — data generation (Modal vLLM)
- `filter_introspection.py` — quality filtering (local)
- `modal_train_sft.py` — SFT training (Modal Unsloth)

**Estimated cost:** ~$20 Modal (capped) | **Time:** ~35 hours generation + 1 hour training/eval

**Inference workarounds documented:** `docs/inference-guide.md` — ForConditionalGeneration + `limit_mm_per_prompt`, NOT ForCausalLM conversion (breaks sliding window attention).

#### Lesson Learned: Train on the Same Model Architecture You'll Serve On (2026-03-29)

**Mistake:** SFT v1 was trained on the novision model (ForCausalLM, flat weight keys) because the VLM model (ForConditionalGeneration) crashed during training with `image_token_id` error. This created an architecture mismatch: the SFT adapter merges onto novision (which has vLLM sliding window bugs and markdown artifacts) instead of the VLM model (which is the proven eval/serving path).

**What should have been done:** Suppress the processor loading in the training script rather than switching to a different model architecture. Unsloth's `FastLanguageModel.from_pretrained` loads the Gemma3Processor when it finds `preprocessor_config.json`. The fix is to either:
1. Remove `preprocessor_config.json` from the model dir BEFORE training, then restore it after (for vLLM serving)
2. Set `trust_remote_code=False` and manually configure the tokenizer
3. Train on the VLM model from the start and handle the processor separately

**For v5 ORPO / next SFT iteration:** Always train on the SAME model architecture that will be used for serving and evaluation. If using Gemma 3, train on the VLM model. If switching to Qwen 3-32B, this problem disappears entirely (pure ForCausalLM, no processor issues).

#### SFT v1 Failure Analysis (2026-03-29)

**SFT v1 scored 1.42/10 via LoRA serving** — catastrophic regression from the ORPO v4 baseline (8.17/10 via same LoRA serving path). Responses reverted to base Gemma assistant voice: *"Okay, let's break down Hamilton's fascinating idea..."*

**Root cause confirmed:** SFT was trained on the novision model (ForCausalLM with broken sliding window attention). The training base was already generating degraded text. SFT reinforced assistant-mode patterns rather than Madison voice.

**The SFT data was good. The training base was wrong.** The 415 filtered reflections + 19 dialogues (~459K tokens) passed all quality filters. The novision architecture was the sole cause of failure.

**Current best model: ORPO v4 via LoRA serving at 8.17/10.**

### Step 9: Iterate — Next Steps After SFT Eval

Ordered by priority. Execute sequentially, evaluating after each step.

#### 9a. vLLM LoRA Serving Mode — CONFIRMED BEST PATH (2026-03-29)

**Result:** ORPO v4 via LoRA serving scored **8.17/10** — comparable to the original merged model eval (8.52 corrected). Adapter-on-base serving also eliminates character breaks on identity-sensitive prompts that the merged path triggers.

**This is the production serving path.** Use `enable_lora=True` + `LoRARequest` in vLLM. No merging needed.

**Test:**
```python
llm = LLM(model="google/gemma-3-27b-it", enable_lora=True, max_lora_rank=16)
# Then at inference:
outputs = llm.generate(prompts, sampling_params, lora_request=LoRARequest("madison", 1, "/path/to/adapter"))
```

**Compare against:** merged model served via `limit_mm_per_prompt` workaround. If adapter-on-base produces cleaner output (no markdown artifacts, no character breaks on the same prompts), this becomes the preferred serving path.

**Cost:** ~$1 (single A100-80GB for ~20 min of testing)

#### 9b. Qwen 3-32B Validation Experiment

**Rationale:** Qwen 3-32B is a pure text-only `ForCausalLM` that eliminates every Gemma 3 infrastructure issue (vLLM multimodal bugs, sliding window attention, GGUF degradation). Research confirmed: no VLM architecture, native vLLM support, first-class Unsloth LoRA support, ChatML template. The "Qwen resists personality modification" claim is unsubstantiated for Qwen 3 — community RP fine-tunes exist.

**Critical: use Qwen 3, NOT Qwen 3.5.** Qwen 3.5 reintroduces VLM architecture and Unsloth warns against QLoRA for it.

**Validation test:**
1. Small rank-8 LoRA on Qwen 3-32B with ~100 of our existing ORPO chosen responses reformatted
2. Generate 10 eval responses via vLLM (should load with zero workarounds)
3. Convert to GGUF Q4_K_M, test same prompts on Ollama
4. Compare voice quality: BF16 vs Q4_K_M — does the character signal survive quantization?

**If validation succeeds:** switch base model for v5 ORPO round. Regenerate rejected responses from base Qwen 3-32B.

**Cost:** ~$5-10 (1-2 hours A100-80GB)

#### 9c. v5 ORPO Round — Address Character Breaks

**Rationale:** Introspection generation (2026-03-29) revealed 3 prompts with 55-97% character break rates on identity, slavery, and meta-self-description topics. The base model's RLHF safety training overpowers the ORPO fine-tune on these topics.

**Data strategy:**
- 150 new DPO pairs targeting identity (50), slavery/moral complexity (50), meta-self-description (50)
- **Chosen:** Sonnet teacher with Madison constitution (same as v4 pipeline)
- **Rejected:** The actual AI-speak responses from introspection generation (saved in `data/training/introspection/reflections.jsonl` unfiltered)
- Combined with existing 874 unique pairs → ~1,024 pairs

**Training changes:**
- Increase LoRA rank to 64 (from 16). Rank 16 deltas are too fragile for quantized deployment — they survive BF16 serving but are destroyed by GGUF Q4_K_M. Rank 64 produces larger weight deltas that are more robust to quantization noise.
- **Serve via LoRA serving mode** (adapter-on-base) for eval — confirmed best quality path

**Base model:** Qwen 3-32B if validation (9b) succeeds, otherwise Gemma 3 27B.

**Cost:** ~$15-25 (data generation + training + eval)

#### 9d. Redo Introspection SFT on Correct Base

**Rationale:** SFT v1 failed because it trained on the novision model (ForCausalLM). The data is good — we need to retrain on the correct architecture.

**Approach (depends on 9b/9c outcome):**
- **If Qwen 3-32B:** No architecture issues. Train SFT on merged v5 ORPO model directly. Qwen 3 is pure ForCausalLM — no preprocessor bugs, no sliding window issues.
- **If Gemma 3 27B:** Remove `preprocessor_config.json` from model dir before training (prevents Gemma3Processor initialization), restore after. Or use the v5 ORPO adapter via LoRA serving for inference (skip merge entirely).

**Data:** Reuse existing filtered introspection data (`data/training/madison-introspection-sft.jsonl`, 510 examples, 459K tokens). No regeneration needed — the data quality was validated.

**Eval via LoRA serving** — do not merge and eval via ForCausalLM path.

**Cost:** ~$5 (single A100-80GB, ~30 min training)

#### 9e. Quantization-Aware Training (QAT)

If rank 64 LoRA still loses signal at Q4_K_M, explore QAT where quantization noise is injected during training. The model learns to be robust to it. Google published QAT Gemma 3 models as proof of concept. This is the proper solution for GGUF deployment of character fine-tunes.

### Step 10: Deploy to Chamber
Load the winning LoRA adapter via vLLM LoRA serving mode (best quality) for cloud demo, or via LM Studio GGUF for local. Point the Foundry Chamber UI at it. Chat with Madison. Verify in the browser that the voice is distinguishably his.

**Serving path:** vLLM LoRA serving mode on Modal for the fellowship demo. Local GGUF via Ollama/LM Studio as a convenience — with known quality caveats documented.

**Deliverable:** A working Madison chat demo for the OSV Fellowship application.

---

## Pipeline Architecture Diagram

```
Source Texts (468K words) + Biographies (1.8M words)
    ↓
Constitution Extraction → madison-5k.md / madison-10k.md            ✅ DONE
    ↓
Behavioral Tests → eval-prompts.jsonl (36 prompts, 6 categories)    ✅ DONE
    ↓
┌─────────────────────────────────┐
│  Prompt Generation              │
│  Teacher (Sonnet) + Student     │                                  ✅ DONE
│  (base Gemma) → 490 DPO pairs  │
└────────────┬────────────────────┘
             ↓
    Quality Filter + Format ORPO                                     ✅ DONE
             ↓
    QLoRA ORPO v3b (Modal A100, beta=0.1, lr=2e-5)                  ✅ DONE
             ↓
    Evaluate (Sonnet judge, 36 prompts)                              ✅ DONE (3.4/10)
             ↓
    ┌── FINDING: Knowledge OK (6.4), Voice FAILED (1.4) ──┐
    │   Base Gemma assistant style bleeds through on ~60%   │
    │   of prompts. Must fix voice before introspection.    │
    └──────────────────┬────────────────────────────────────┘
                       ↓
    Voice-Targeted ORPO v4                                           ✅ DONE
    │  1,273 effective pairs (475 original + 399 voice + 2x upsample)
    │  Modal A100-80GB, beta=0.1, lr=2e-5, 3 epochs
    │  Result: 8.52/10 corrected (all categories improved)
             ↓
    Re-Evaluate                                                       ✅ DONE (8.52/10 corrected)
             ↓
    Introspection SFT Data Gen (self-reflection + self-interaction)
    │  Now the model generates in Madisonian voice
             ↓
    SFT Training (Modal A100)
             ↓
    Final Eval → Iterate if needed
             ↓
    GGUF Q4_K_M → LM Studio on RTX 3090                             ✅ INFRA READY
             ↓
    Deploy to Chamber UI → Chat with Madison
```

## Estimated Cost and Timeline

| Step | Cost | Time | Model | Status |
|------|------|------|-------|--------|
| 0. Constitution | $0 (Max plan) | 2-3 hours | Claude Code | DONE |
| 1. Behavioral tests | $0 (Max plan) | 1 hour | Claude Code | DONE (36 eval prompts) |
| 2. Prompt generation | $0 (local) | 30 min | Gemma 3 / Qwen 3.5 | DONE |
| 3. Teacher responses | $5-10 | 1-2 hours | Gemini Flash / Kimi 2.5 | DONE (Sonnet 4.6) |
| 4. Student responses | $0 (local) | 1-2 hours | Gemma 3 27B base | DONE |
| 5. Format + filter | $0 | 30 min | Scripted | DONE |
| 6. First ORPO run | $5-20 (Modal) | 64 min | A100 80GB | DONE (v3b, 100% eval acc) |
| 7. Evaluate | $0.50 (Sonnet judge) | 30 min | Sonnet + prompt caching | DONE (3.4/10 overall) |
| **7.5. Voice-targeted ORPO** | **~$11 (Sonnet + Modal)** | **~4 hours** | **Sonnet + A100** | **DONE (v4, 8.52/10)** |
| 7.5b. Re-evaluate | $0.50 | 30 min | Sonnet judge | DONE (Modal re-eval) |
| 8. Introspection SFT | $5-20 (Modal) | 3-5 hours | Modal + LM Studio | **NEXT (unblocked)** |
| 9. Iterations (2-3x) | $10-40 (Modal) | 2-3 sessions | Modal | PLANNED |
| 10. Deploy | $0 | 30 min | LM Studio | PLANNED |
| **Total** | **$30-100** | **~4-6 work sessions** | | |

**Actual spend to date:** ~$13 Modal (training + eval gen + GGUF conversion) + ~$7 Sonnet (teacher + judge) = **~$20 total**
Out-of-pocket remaining: ~$20-90 in Modal credits from existing $500+ balance.

## Source Material Inventory

See `sources/SOURCES.md` for the complete document index.

### Primary Sources (Madison's writings)

| Category | Files | Words | Status |
|----------|-------|-------|--------|
| Federalist Papers | 29 | 69,344 | Complete |
| Essays (National Gazette, Helvidius, etc.) | 39 | ~156,000 | Complete |
| Speeches (Convention, Congressional, VA legislature) | 10 | ~94,000 | Complete |
| Congressional (individual speeches) | 22 | ~56,000 | Complete |
| Legislative Writings | 6 | ~26,000 | Complete |
| Presidential Papers | 21 | ~33,000 | Complete |
| Key Correspondence | 13 | ~26,000 | Complete |
| **Total primary** | **140** | **~468,000** | |

### Scholarly Biographies (extracted to searchable text)

| Book | Words | Format |
|------|-------|--------|
| Chernow — Alexander Hamilton | 371,032 | ePub → text |
| Ketcham — James Madison: A Biography | 365,776 | PDF → docling |
| Burstein — Madison and Jefferson | 332,890 | ePub → text |
| Feldman — Three Lives of James Madison | 288,653 | ePub → text |
| Cheney — Madison: A Life Reconsidered | 200,141 | ePub → text |
| Leibiger — Founding Friendship | 123,355 | PDF → docling |
| Ellis — Founding Brothers | 110,647 | ePub → text |
| **Total biographies** | **1,792,494** | |

### Bulk Reference
- Hunt 9-Volume Edition: 9 PDFs, 3,065 pages (LiteParse extraction complete for key documents)

## Research References

### Primary
- Maiya, Bartsch, Lambert, Hubinger — "Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI" (arXiv:2511.01689, Nov 2025)
  - Code: [github.com/maiush/OpenCharacterTraining](https://github.com/maiush/OpenCharacterTraining)
  - Data: [huggingface.co/datasets/maius/OpenCharacterTraining-data](https://huggingface.co/datasets/maius/OpenCharacterTraining-data)
  - Key insight: 100% synthetic data from hand-written constitutions. ~14M tokens per character.

### Supporting
- Shao, Li, Dai, Qiu — "Character-LLM: A Trainable Agent for Role-Playing" (arXiv:2310.10158, EMNLP 2023)
  - Code: [github.com/choosewhatulike/trainable-agents](https://github.com/choosewhatulike/trainable-agents)
  - Key insight: "Experience Reconstruction" from biographical profiles. ~750K words per historical character.

- Lambert — "Opening the Character Training Pipeline" (Interconnects.ai, Nov 2025)
  - Key insight: Constitutional AI for character = constructing traits → generating queries → producing responses → ranking against spec.

- Lambert — "Character Training: Understanding and Crafting a Language Model's Personality" (Interconnects.ai)
  - Key insight: "Character training is easy to imprint but the challenge is data alignment with intentions."

- Lambert — The RLHF Book, Chapters 17 (Product & Character) and 19 (Character Training)
  - Key insight: Character is a subset of post-training focused on manner, not content.

- Askell et al. (Anthropic) — Claude's Character
  - Key insight: Test-driven development — write tests for desired character behavior FIRST, then train to pass them.

## Constitution Sources

The Madison constitution should be informed by both primary sources and scholarly biographies that synthesize his character, temperament, and intellectual evolution.

### Primary Sources (for positions, vocabulary, rhetorical patterns)
- Federalist Papers (especially 10, 39, 51)
- Memorial and Remonstrance
- Virginia Ratifying Convention speeches
- Key correspondence with Jefferson, Roane, Barry

### Scholarly Biographies (for personality, temperament, evolution)
- **Ralph Ketcham, "James Madison: A Biography"** — definitive single-volume biography
- **Andrew Burstein & Nancy Isenberg, "Madison and Jefferson"** — intellectual partnership and mutual influence
- **Noah Feldman, "The Three Lives of James Madison"** (2017) — intellectual evolution across career phases
- **Kevin Gutzman, "James Madison and the Making of America"** — constitutional period focus
- **Lynne Cheney, "James Madison: A Life Reconsidered"** — newer archival material

These biographies provide the interpreted, synthesized view of Madison's character that his own writings don't always reveal — how his thinking evolved, what contradictions drove him, what contemporaries observed about his temperament and debating style.

## Open Questions

- Is LoRA rank 64 sufficient for a 27B model, or do we need rank 128+?
- Should we use the Maiya approach (GLM teacher) or Claude as the teacher model?
- How do we handle topics Madison never encountered (AI, internet, nuclear weapons)?
- What's the right balance of historical accuracy vs. engaging conversation?
- Should the essay-to-conversation stage preserve Madison's exact sentences or allow paraphrasing?
- How many DPO pairs do we need before diminishing returns? Maiya used ~6M tokens — is that necessary at 27B?
