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

**Prerequisite:** Step 7.5 voice-targeted round must achieve voice_authenticity ≥ 5.0

The voice-corrected model generates self-reflective data:
- 10 reflection prompts × 1000 repetitions (diary entries, letters to founders, biographical reflections)
- Self-interaction: two copies of the model converse about their beliefs for 10 turns
This makes the character "robust and pleasant."

**Model:** The voice-corrected model itself (runs on Modal or LM Studio via GGUF)
**Script:** `src/foundry/press/introspection.py`
**Output:** `data/training/madison-introspection-sft.jsonl`
**Time:** ~2-4 hours generation + ~30-60 minutes SFT training

### Step 9: Iterate
Refine based on evaluation failures:
- Adjust constitution if certain traits aren't imprinting
- Add targeted training examples for failing test cases
- Try different LoRA rank, learning rate, DPO beta
- Each iteration is $5-20 on Modal + evaluation time
- Target: 3-5 iterations to reach satisfactory quality

### Step 10: Deploy to Chamber
Load the winning LoRA adapter in LM Studio. Point the Foundry Chamber UI at it. Chat with Madison. Verify in the browser that the voice is distinguishably his.

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
    Voice-Targeted ORPO v4 (NEW STEP)                                ⬅️ NEXT
    │  ~100 voice-specific pairs:
    │  - Eval failures as rejected (actual model output)
    │  - Sonnet chosen (same content, correct voice)
    │  - Anti-pattern pairs (bullets→prose, contractions→formal)
    │  + existing 490 content pairs
             ↓
    Re-Evaluate (target: voice ≥ 5.0, no content regression)
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
| **7.5. Voice-targeted ORPO** | **$2-5 (Sonnet + Modal)** | **3-4 hours** | **Sonnet + A100** | **NEXT** |
| 7.5b. Re-evaluate | $0.50 | 30 min | Sonnet judge | PLANNED |
| 8. Introspection SFT | $5-20 (Modal) | 3-5 hours | Modal + LM Studio | PLANNED (blocked on 7.5) |
| 9. Iterations (2-3x) | $10-40 (Modal) | 2-3 sessions | Modal | PLANNED |
| 10. Deploy | $0 | 30 min | LM Studio | PLANNED |
| **Total** | **$30-100** | **~4-6 work sessions** | | |

**Actual spend to date:** ~$8 Modal (training + eval gen + GGUF conversion) + ~$0.50 Sonnet judge = **~$8.50 total**
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
