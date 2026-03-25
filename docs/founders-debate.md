---
name: The Foundry Project
slug: foundry
status: active
priority: high
started: 2026-03-19
description: Fine-tune character LoRA models of US Founding Fathers for Lincoln-Douglas debate app and conversational chat. Repo name "foundry".
---

## Vision

Fine-tune individual LoRA adapters capturing the writing style, reasoning patterns, and philosophical positions of key US Founding Fathers — starting with James Madison — then build an interactive debate platform where these characters argue modern topics from their historical perspectives.

Two modes:
1. **Lincoln-Douglas Debate** — Two founders debate a modern topic (digital privacy, executive power, AI regulation) with a moderator. Each draws from their documented positions and rhetorical style.
2. **Character Chat** — Chat directly with a founder. One-on-one conversation with a historical figure.

Voice synthesis (ElevenLabs) and period-appropriate visuals for each character.

## Why This Works

- **Source material is ideal for LoRA**: Federalist Papers, Constitutional Convention notes, personal correspondence — structured, well-documented, distinctive voices with ground truth.
- **Evaluation is clear**: Does it sound like Madison? Historians can judge. Character voice quality is measurable against primary sources.
- **Architecture proven in private prototype**: Session management, character engine, streaming UI, memory — all validated in a prior project. Clean rebuild for public repo.
- **Publishable and open-sourceable**: Educational tool, research demo, potentially fundable.

## Architecture

**Repo:** `seaberger/Foundry` (public)

**Clean build.** Architectural patterns (FastAPI + HTMX, session management, character engine, streaming SSE) adapted from prior prototyping work.

```
foundry/
  src/foundry/
    chamber/       — debate + chat UI (streaming, sessions, turns)
    press/         — training data pipeline (source collection, conversion, formatting)
    characters/    — founder profiles, constitutional prompts, LoRA configs
    inference/     — model serving, LoRA adapter loading/swapping
    voice/         — ElevenLabs TTS integration per character
  config/
    characters/
      madison.yaml       — voice, positions, rhetorical patterns, key writings
      hamilton.yaml
      jefferson.yaml
    foundry.yaml         — server config, model endpoints, debate settings
  templates/
    debate.html          — split-screen debate view with moderator
    chat.html            — one-on-one chat with a founder
    sessions.html        — session list, character selection
  static/
    js/chat.js           — SSE streaming, turn management
    css/                 — period-appropriate but modern design
  modal_serve.py         — serverless LoRA inference on Modal A100
  tests/
  README.md              — "The Foundry Project"
```

**Key components:**
- **Chamber** — the debate/chat platform. FastAPI + HTMX, SSE streaming, session persistence in SQLite
- **Press** — training data pipeline. Collects historical writings, converts to ShareGPT format, generates DPO pairs
- **Characters** — YAML profiles defining each founder's voice, positions, vocabulary, and constitutional beliefs
- **Inference** — LoRA adapter loading on shared base model (Gemma 3 27B). Hot-swap between characters per turn
- **Voice** — ElevenLabs per-character voice profiles for audio output

**Per-character LoRA adapters** loaded at inference time on shared Gemma 3 27B base
- **Character knowledge base** per founder — documented positions, key writings, known debates, rhetorical patterns
- **Debate moderator** — separate prompt that manages turn structure, introduces topics, calls for rebuttals

## Characters (Priority Order)

1. **James Madison** — Constitution's primary architect, Federalist Papers co-author. Precise, measured, legalistic reasoning. Sean's primary interest.
2. **Alexander Hamilton** — Federalist Papers co-author, aggressive, financially-minded, prolific writer. Natural foil to Madison (they actually debated).
3. **Thomas Jefferson** — Philosophical, expansive, contradictory. Declaration author. Contrasts with Hamilton's federalism.
4. **John Adams** — Pragmatic, cantankerous, independent thinker. Provides a third perspective distinct from the Virginia/New York axis.
5. **Benjamin Franklin** — Wit, diplomacy, scientific mind. Oldest voice, different generation's perspective.

## Fine-Tuning Plan

### Technical Recommendations (from Lambert/Maiya research)

**Base Model Selection:**
- **Gemma 3 27B preferred** — Lambert's research found Qwen resists personality modification more than Llama or Gemma variants. Gemma 2 27B was tested for persona vector composition and showed good character imprinting. Gemma 3 27B is the natural successor.
- Llama 3.3 70B also viable if budget allows (tested in persona space research)
- **Avoid Qwen** for character fine-tuning specifically — model architecture resists personality changes

**Method: Constitutional AI + DPO (Lambert's two-stage approach)**
1. **Stage 1 — Distillation:** Use Claude/GPT-4 with character-specific constitutional prompts (Madison's documented positions, rhetorical patterns, vocabulary) to generate preferred responses. Apply DPO against base model outputs.
2. **Stage 2 — Self-reflection SFT:** Generate on-policy data where the model explains its reasoning *in character*. Train on self-reflection outputs with constitution in context. This step makes character "robust and pleasant."

**Why Constitutional AI + DPO over pure SFT:**
- Lambert: "character training via weight modification outperforms both prompting and activation steering in robustness and expressiveness"
- DPO teaches what the character *wouldn't* say (negative examples from base model) — critical for historical accuracy
- "Little to no effect on general capabilities" — the model stays smart, just sounds like Madison

**Evaluation:**
- Train a ModernBERT classifier to distinguish characters (Lambert's approach — used 11 personas)
- "Revealed preferences" testing — give model two traits, see which it embodies
- Historical accuracy panel — scholars evaluate if the voice is authentic
- Blind A/B testing — can a historian distinguish fine-tuned Madison from prompted Madison?

**Hardware:** QLoRA via Unsloth on Modal A100 ($5-20/run)
**Format:** ShareGPT/ChatML — debates and correspondence converted to conversational format

### Training Data Pipeline

**Per-character training data (estimated 2000-5000 examples each):**

**Tier 1 — Primary sources (free, highest authenticity):**
- Published writings (Federalist Papers, state papers, letters)
- Documented debates and speeches (Constitutional Convention records)
- Personal correspondence (Library of Congress digitized collections)
- Convert from prose/letter format → ShareGPT conversational format via LLM

**Tier 2 — Claude-generated synthetic debates ($50-100/character):**
- Constitutional prompt: "You are James Madison. Reason about [modern topic] as Madison would, drawing on your documented positions about [related 18th century issue]."
- Generate preferred responses (character-accurate) + rejected responses (generic/modern voice)
- DPO pairs: (character response, base model response) for each query

**Tier 3 — Self-reflection data (on-policy, from fine-tuned model):**
- After Stage 1, generate data where the model *explains* why it holds a position, citing its documented beliefs
- "As I argued in Federalist No. 51, the structure of government must..." — this reinforces grounding

**Pipeline:** Collect writings → LLM conversion to ShareGPT → Constitutional prompt generation → DPO pair construction → quality filter → LoRA fine-tune → evaluate against historical voice

### Autoresearch Integration: Recursive Self-Improvement for Character Training

Karpathy's autoresearch pattern (immutable evaluation + mutable artifact + fixed time budget) can close the loop on character fine-tuning in a way nobody has published. See `docs/autoresearch-for-foundry.md` for full analysis.

**The core insight:** The state of the art (Lambert/Maiya) is single-pass — train once, evaluate, maybe manually retrain. The Foundry Loop automates recursive improvement across four levels:

**Level 1 — Training Recipe Optimization.** Agent modifies LoRA rank, beta, lr, layer targeting. Each 5-minute experiment trains and evaluates against a fixed Madison authenticity judge. ~96 experiments overnight on the RTX 3090.

**Level 2 — Data Curation as Mutable Artifact.** Agent experiments with which of the 490 pairs to include/exclude, curriculum ordering, theme weighting, difficulty filtering. Same fixed evaluation, different data selections explored.

**Level 3 — Self-Play Data Generation.** After Levels 1-2 find the best recipe, the trained model generates new responses. These become on-policy rejected examples for the next DPO round — each round forces finer distinctions. SPIN (Self-Play Fine-tuning) automated. Research shows up to 8x data efficiency from on-policy data.

**Level 4 — Judge Refinement.** The judge prompt itself improves (not the constitution — that's immutable ground truth). Agent tests the judge against Madison's verified verbatim writings and modern paraphrases, calibrating discrimination power. Better judge → better training signal → better model.

**Hardware:** Runs overnight on the RTX 3090 (not Modal). Gemma 3 27B 4-bit QLoRA fits in 24GB. Each experiment: 5-min local train → generate 30 responses → Anthropic API judge scores. GPU compute is free; API judge calls cost ~$30-60/night.

**The publishable contribution:** Recursive self-improvement via automated experimentation produces character models that exceed manual tuning — with a reusable methodology for any historical figure.

### DPO Optimization Findings (March 2026 Research)

Key findings from NeurIPS 2025, ICML 2025, and Gemma-specific research:

- **Chosen quality dominates:** Teacher response quality is the #1 factor — rejected quality barely matters (Pan et al., NeurIPS 2025)
- **Gemma 3 layer targeting:** LoRA must target first 2/3 of layers (0-40 of 62) — layers 40+ ineffective for behavioral modification
- **Beta=0.1 for character:** Lower beta enables aggressive character imprinting while preserving base capabilities
- **Rank 16-32 optimal:** Character voice is style/manner, not factual recall — moderate rank suffices
- **Difficulty filtering:** Remove hardest 15-20% of pairs after first training pass — overly difficult examples harm alignment (ICML 2025)
- **Curriculum ordering:** Train on easy pairs first (constitutional topics with obvious voice), harder pairs last (modern extrapolation)
- **ORPO as alternative:** Single-stage preference optimization eliminates SFT/DPO distribution shift — worth testing on second iteration
- **SPIN for corpus matching:** When you have strong teacher text and weak negatives, SPIN can outperform DPO (uclaml/SPIN)

### Prior Art Deep Dive Findings (March 2026)

Critical findings from detailed analysis of DeePer, MentalArena, Objective Matters, Chauhan AI Founding Fathers, OpenCharacter, and CoSER. Full analysis in `docs/prior-art/`.

**Pipeline changes adopted:**
- **DPO + SFT loss (alpha=0.1)** — DeePer showed this prevents likelihood displacement where chosen response probability decreases during training. Added to `modal_train_dpo.py`.
- **ORPO support added** — Objective Matters paper found DPO causes persona drift at 200K-400K tokens (our ~245K is at onset). ORPO shows zero drift at any budget. Run both in parallel and compare. Use `--objective orpo --beta 0.05`.
- **LoRA dropout=0.05** — Confirmed by Objective Matters Gemma ablations. Updated from 0.0.
- **Adversarial stress-testing** — Chauhan's Red Team Agent approach for systematic character-breaking probes. Need to expand eval prompts.

**Iteration guardrails (for Level 3 autoresearch):**
- **Switch to ORPO for iteration 2+** — cumulative DPO tokens cause progressive drift
- **Self-sampling: 15 candidates, temp=1.0, top_p=0.4** — DeePer's recipe for on-policy DPO data
- **Increasing DPO margin: 0.5 → 0.8 across iterations** — as model improves, discrimination threshold must increase
- **Experience replay: carry 5K best pairs forward** — prevents catastrophic forgetting
- **Stop at ~4 iterations** — MentalArena found performance declines after iteration 4
- **Monitor diversity gain** — stop iterating when generated data becomes repetitive

**Domain validation:**
- **Chauhan's Madison scored 90/100** — highest of Hamilton/Jefferson/Madison. Madison's measured, legalistic style is the best fit for structured AI generation.
- **Response generation > response rewriting** — OpenCharacter proved generating from scratch with character profile beats rewriting existing answers (validates our Sonnet teacher approach)
- **Given-circumstance evaluation** — CoSER evaluates whether the character responds appropriately given their specific historical period. Consider adding temporal context to eval prompts.

**New publishable contribution identified:**
- **RAG + fine-tuning combination** — Chauhan does RAG-only, Lambert does fine-tuning-only. Nobody combines both. Our 5K constitution for fine-tuning + potential RAG from 468K primary source corpus would be a second novelty alongside the autoresearch loop.

### Practical Notes from Lambert

- Character training is "extremely synthetic data-heavy" — plan for significant data generation budget
- Filter specific phrases that break character ("Certainly!", "As an AI model...", "I'd be happy to...")
- Amanda Askell (Anthropic): Use test-driven development — write tests for desired character behavior FIRST, then train to pass them
- "Character training is easy to imprint but the challenge is data alignment with intentions" — quality over quantity

## Networking

- **Nathan Lambert** — Post-training lead at Allen AI, Seattle. ~2.5 hours from Portland.
  - Interconnects newsletter/podcast, RLHF Book author
  - Advised Sharan Maiya on the Open Character Training paper
  - Potential advisor/collaborator for this project
  - Could provide feedback on POC, methodology validation, potentially co-author
  - Reach out after POC demonstrates Madison voice quality

- **Jim O'Shaughnessy** — [O'Shaughnessy Ventures Fellowships](https://www.osvfellowship.com/)
  - **$100,000 Fellowship** (equity-free, no company required, full ownership retained)
  - Also $10,000 Grants — every applicant auto-considered for both
  - **2026 applications close April 30, 2026** — selections by June 1st
  - One-year program for "researchers, builders and creatives advancing civilization"
  - **Strong precedent:** 2025 fellowship awarded $100K to Charlie Becker for AI-powered literary knowledge preservation (open database for rare books). Foundry is the same pattern — AI preserving and reanimating historical intellectual knowledge.
  - Open worldwide, any background, any age — no institutional affiliation required
  - Past fellows include quantum computing, endangered language preservation, brain simulations
  - **Pitch angle:** "Using fine-tuned LLMs to preserve and reanimate the rhetorical voices of America's founders for civic education and constitutional discourse"
  - **Timeline:** Application due April 30, 2026. Fellowship funds the year of work — we need a compelling POC and vision, not a finished product.

## Sprint Plan — OSV Application (April 30, 2026)

**Goal:** Submit a strong application with a working Madison voice demo, clear technical approach, and compelling vision for civic education impact.

**The application needs:**
1. A working demo — even brief — showing Madison's voice is distinguishably his
2. Clear technical plan (we have this — Lambert methodology, Constitutional AI + DPO)
3. Vision for impact (civic education, constitutional discourse, public engagement)
4. Evidence of feasibility (Crucible architecture proven, Modal infrastructure ready, training data pipeline designed)

### Week 1 (Mar 19-25): Foundation
- [x] Create `seaberger/Foundry` repo with clean scaffold
- [x] Madison and Hamilton character cards with system prompts
- [x] Config system, DB layer, server scaffold
- [ ] Build Chamber chat UI — streaming chat with Madison (prompted, no LoRA yet)
- [ ] Collect Madison primary sources — Federalist Papers, Convention notes, key letters
- [ ] Test Madison system prompt against base Gemma 3 27B — establish prompted baseline

### Week 2 (Mar 26-Apr 1): Training Data + First Fine-Tune
- [ ] Build Press pipeline: convert historical texts → ShareGPT format
- [ ] Generate DPO pairs: Claude produces Madison-voice responses + base model rejections
- [ ] First LoRA fine-tune attempt on Modal A100
- [ ] Evaluate: does fine-tuned Madison sound better than prompted Madison?

### Week 3 (Apr 2-8): Iterate + Chat Demo
- [ ] Iterate on training data quality based on first results
- [ ] Second/third fine-tune runs with refined data
- [ ] Madison chat demo working end-to-end in Chamber with LoRA model
- [ ] Start Hamilton training data if Madison voice is solid
- [ ] Blog post #1 on sbergman.net — the vision and approach

### Week 4 (Apr 9-15): Demo Polish
- [ ] ElevenLabs voice for Madison
- [ ] If Hamilton ready: first debate demo
- [ ] Record demo video or set up live demo URL
- [ ] Blog post #2 — technical learnings from fine-tuning

### Week 5 (Apr 16-22): Application Writing
- [ ] Write OSV application essay — vision, methodology, impact, personal motivation
- [ ] Prepare supporting materials — demo link, code repo, technical plan
- [ ] Draft budget for fellowship year ($100K allocation plan)
- [ ] Blog post #3 — applying for the fellowship, building in public

### Week 6 (Apr 23-30): Submit
- [ ] Review, polish, submit by April 30
- [ ] Share POC with Lambert for feedback (even if after submission)

## Research References

### Nathan Lambert (Allen AI / Interconnects)

**Key Paper:** Sharan Maiya & Nathan Lambert — *"Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI"* ([arXiv:2511.01689](https://arxiv.org/abs/2511.01689))
- Character training via weight modification outperforms prompting and activation steering
- Two-stage: distillation with character-specific constitutional prompts (DPO) + supervised fine-tuning with on-policy self-reflection
- Code & models: [maius/OpenCharacterTraining](https://huggingface.co/collections/maius/OpenCharacterTraining) on HuggingFace
- Finding: "character training is easy to imprint into the model" — challenge is data alignment with intentions
- Qwen resists personality modification more than Llama or Gemma variants

**Blog Posts:**
- [Opening the character training pipeline](https://www.interconnects.ai/p/opening-the-black-box-of-character) — Nov 2025. Revealed preferences testing, ModernBERT classifier for personality detection.
- [Character training: Understanding and crafting a language model's personality](https://www.interconnects.ai/p/character-training) — Constitutional AI approach, Amanda Askell's test-driven development methodology.

**Book:** [The RLHF Book](https://rlhfbook.com) — Chapter 17 (Product & Character) and Chapter 19 (Character Training). Covers character as a subset of post-training focused on *manner* not content.

**Key Insight from Lambert:** Constitutional AI for character = constructing specific traits → generating model-created queries matching traits → producing responses → ranking responses against character spec. No human data needed after initial setup.

## Sabbatical & Public Journey

**The Foundry Project is the sabbatical project.** The O'Shaughnessy Fellowship ($100K, equity-free) funds the post-retirement year. September 2026 retirement from Coherent → fellowship-funded year building Foundry → proves the "AI artisan" model with a public, impactful project.

**Open source from day one.** Repo is public. Development is transparent. This is a portfolio piece, a funding application, and a public demonstration of AI engineering capability — simultaneously.

**Blog the journey on sbergman.net:**
- Technical posts: fine-tuning methodology, character voice evaluation, training data curation
- Historical posts: what we learn about Madison/Hamilton by training models on their words
- Process posts: applying for fellowships, building in public, sabbatical planning
- This content thread builds the public presence (G19) while documenting the project (G20)

**Mission alignment:**
- M1: Fellowship funds financial independence during sabbatical year
- M2: Deepest AI fluency project yet — fine-tuning, LoRA, DPO, character training, full-stack deployment
- M5: Intellectual engagement with founders' ideas as an act of reclamation and curiosity

## Budget & Funding

- **Modal credits:** $521 allocated to this project (bulk of remaining credits)
- **Budget breakdown estimate:**
  - Training data generation (Claude API for synthetic debates): $50-100/character
  - Fine-tuning runs (QLoRA on A100): $5-20/run, budget for 50+ experimental runs
  - Vision model hosting (if reusing for visual identity): minimal
  - Total estimated: $200-400 for full POC with 2-3 characters
- **Funding goal:** Successful POC → apply for research funding (digital humanities, AI education, constitutional studies)
- **POC target:** Madison vs Hamilton debating a modern topic, with voice synthesis, where someone familiar with the Federalist Papers can hear the authentic voice difference

## Open Questions

- Is LoRA sufficient to capture distinct historical voices at 27B, or do we need full fine-tune?
- How to handle topics founders never encountered (AI, internet, nuclear weapons) — extrapolation from principles vs. acknowledged uncertainty?
- Copyright/public domain status of all source materials (should be clear for 18th century texts)
- Debate format: structured turns vs. free-flowing? Time limits per response?
- How to evaluate "authenticity" — historian review panel? Blind comparison with actual writings?
