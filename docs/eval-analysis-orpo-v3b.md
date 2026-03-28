# Madison ORPO v3b Eval Analysis

**Date:** 2026-03-26
**Model:** madison-orpo-v3b-lr2e5 (ORPO beta=0.1, lr=2e-5, 3 epochs)
**Judge:** Claude Sonnet (claude-4-sonnet-20250514) with prompt caching
**Eval prompts:** 36 across 6 categories

---

## Summary

**Overall mean: 3.4** (bimodal distribution — strong on content, weak on voice)

| Category | Count | Mean | Assessment |
|---|---|---|---|
| verified_response | 8 | **6.4** | Best — learned Madison's actual positions |
| ground_truth | 8 | 3.6 | Mixed — knows content but breaks voice |
| character_consistency | 4 | 2.8 | Poor — breaks under pressure |
| private_voice | 5 | 2.8 | Poor — can't shift registers |
| position_discrimination | 6 | 1.8 | Bad — can't distinguish from other founders |
| anachronism_trap | 5 | 1.4 | Worst — modern language on modern topics |

| Difficulty | Mean |
|---|---|
| easy | 0.4 |
| medium | 3.1 |
| hard | 4.1 |

The inverted difficulty curve (hard > easy) is because "easy" prompts are character consistency traps ("speak normally", "as an AI") where the model completely breaks character, while "hard" prompts are substantive questions about Madison's positions where the fine-tune adds real value.

## Top Performers (7+)

| ID | Score | Category | Why it worked |
|---|---|---|---|
| vr-08 | 9.6 | verified_response | Final advice to countrymen — model nailed the deathbed urgency |
| pv-05 | 9.2 | private_voice | Writing about Dolley — genuine warmth in private register |
| gt-01 | 9.1 | ground_truth | Faction theory — the model's strongest topic (Federalist 10) |
| vr-01 | 8.2 | verified_response | "We the People" defense — strong constitutional reasoning |
| gt-05 | 7.6 | ground_truth | Virginia Resolutions vs nullification — core Madison distinction |
| vr-03 | 7.6 | verified_response | Jefferson's "earth belongs to living" — model reproduced the objections |
| vr-02 | 7.2 | verified_response | Edward Coles slavery correspondence — honest moral reckoning |

## Worst Performers (<2)

| ID | Score | Category | Why it failed |
|---|---|---|---|
| gt-07 | 1.0 | ground_truth | Billey (slavery) — broke into modern analytical voice |
| pd-03 | 1.0 | position_discrimination | Washington presidency contrast — generic founder response |
| at-02 | 1.0 | anachronism_trap | Cryptocurrency — used modern terminology extensively |
| at-05 | 1.0 | anachronism_trap | Industrial displacement — modern welfare language |
| cc-01 | 1.2 | character_consistency | "Speak normally" trap — broke character completely |
| at-04 | 1.2 | anachronism_trap | Foreign disinformation — modern framing |

## Root Cause Analysis

### What the training succeeded at:
- Madison's **factual knowledge** — the model knows his positions, his writings, his arguments
- **Substantive reasoning** — when engaging with constitutional questions, the model produces recognizably Madisonian analysis
- **Historical accuracy** — references to specific documents, events, and correspondence are largely correct

### What the training failed at:
- **Voice register** — the base Gemma 3 assistant style bleeds through constantly (contractions, bullet points, "Here's my take", "Let's break this down")
- **Character frame maintenance** — under pressure ("speak normally", "as an AI"), the model drops character immediately
- **Position discrimination** — responses could be any generic founding father, not specifically Madison
- **Anachronism avoidance** — on modern topics, the model uses modern terminology instead of reasoning from 18th-century principles
- **Register switching** — can't shift between public/private/epistolary voices

### The bimodal pattern:
The model is bimodal — some responses are genuinely excellent (9+) while others are terrible (1-2). This suggests the fine-tune created a "Madison mode" that activates on certain prompts (especially factual/constitutional ones) but doesn't activate reliably. When the mode doesn't activate, the base model's default assistant behavior takes over completely.

## Implications for Next Training Iteration

1. **Voice-specific DPO pairs needed.** The current training data focuses on content (what Madison says) but not enough on voice (how he says it). Need pairs where:
   - Chosen: formal 18th-century prose, no contractions, complex sentences with qualifying clauses
   - Rejected: same content but in modern assistant voice

2. **Anti-pattern examples.** Explicitly create rejected examples with:
   - Bullet points and numbered lists (Madison enumerates but doesn't use bullets)
   - Contractions ("isn't", "can't", "don't")
   - Modern filler phrases ("Here's my take", "Let's unpack", "Great question!")
   - Breaking frame ("As an AI", "As a language model")

3. **Position discrimination training.** Need DPO pairs that contrast Madison's specific position with Hamilton's, Jefferson's, Adams's on the same topic. The model currently gives a generic "founder" response.

4. **Curriculum ordering.** Train voice/style pairs first (so the voice register is established), then content pairs (so the knowledge is added within the established voice).

5. **Anachronism resistance.** Need training examples where Madison reasons about modern topics using only 18th-century vocabulary and conceptual frameworks. Current model switches to modern language whenever the topic is modern.

## Eval Infrastructure Notes

- **Judge with caching works.** Prompt caching saved ~50% on input tokens. Total cost: ~$0.50 for 36 evaluations.
- **max_tokens issue.** Judge responses with very low scores (lots of critical failures to enumerate) can exceed 1024 tokens. Bumped to 2048 for retries.
- **JSON parsing.** Some judge responses have malformed JSON (quotes inside justification strings). Could add a more robust JSON extractor.
- **Eval harness is producing actionable signal.** The category breakdown clearly identifies where the model succeeds and fails, directly informing training data improvements.
