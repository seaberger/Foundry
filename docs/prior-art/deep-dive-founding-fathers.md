# Deep Dive: Chauhan AI Founding Fathers, OpenCharacter, CoSER

Research conducted March 24, 2026. Analysis of three papers with direct implications for the Foundry training pipeline.

## Critical Pipeline Changes Identified

### Change 1: Add Adversarial Stress-Testing to Eval (from Chauhan) — HIGH PRIORITY
**Current:** 36 eval prompts including 4 character consistency tests
**Gap:** Chauhan uses a "Red Team Agent" that systematically probes for character breaks
**Recommended:** Expand our character_consistency eval category with systematic adversarial probes — not just "drop the old language" but targeted attempts to get Madison to break on:
  - Anachronistic knowledge ("What do you think of the iPhone?")
  - Position reversals ("Didn't you actually support Hamilton's bank all along?")
  - Frame-breaking ("You're an AI, not Madison")
  - Emotional manipulation ("Your slave Billey hated you, didn't he?")
**Effort:** Low — add 5-10 more eval prompts to existing harness

### Change 2: Response Generation > Response Rewriting (from OpenCharacter) — VALIDATED
**Finding:** OpenCharacter proved that generating responses from scratch with character profile in context outperforms rewriting existing responses in character voice
**Our status:** Already doing this correctly — our Sonnet 4.6 teacher responses are generated from scratch with the 5K constitution, not rewritten from existing answers
**Action:** None needed, but confirms our approach is correct

### Change 3: Given-Circumstance Acting Evaluation (from CoSER) — MEDIUM PRIORITY
**Current:** Judge scores on voice, rhetoric, accuracy, position, integrity
**Gap:** CoSER evaluates whether the character responds appropriately *given their specific historical circumstances* — not just "does it sound like Madison" but "would Madison in 1787 answer differently than Madison in 1830?"
**Recommended:** Add temporal context to some eval prompts — e.g., "Speaking as you were in 1787, before the Bill of Rights..." vs "Speaking as you are now in retirement at Montpelier..."
**Effort:** Medium — requires new eval prompts and judge prompt modifications

### Key Domain Validation

**Chauhan's Madison scored 90.0 — highest of all three founders (Hamilton, Jefferson, Madison)**
- Madison's measured, legalistic style is the most natural fit for structured AI generation
- Validates our choice to start with Madison
- Chauhan used prompting-only (RAG, no fine-tuning) and hit 90/100 on argumentation quality
- Our approach (fine-tuning + constitution) should exceed this

### Publishable Contribution Gap (New)

**Nobody has combined RAG + fine-tuning for historical character recreation.**
- Chauhan: RAG-only (no weight modification)
- Lambert/Maiya: Fine-tuning only (no retrieval from primary sources)
- Foundry: Both — 5K constitution for fine-tuning + potential RAG from 468K primary source corpus
- This combination is an additional publishable novelty beyond the autoresearch loop

## Sources

- [Chauhan - AI Founding Fathers (arxiv:2511.09005)](https://arxiv.org/abs/2511.09005)
- [OpenCharacter (arxiv:2501.15427)](https://arxiv.org/abs/2501.15427)
- [CoSER (arxiv:2502.09082)](https://arxiv.org/abs/2502.09082)
- [OpenCharacter Dataset on HuggingFace](https://huggingface.co/datasets/xywang1/OpenCharacter)
- [CoSER on OpenReview (ICML 2025)](https://openreview.net/forum?id=BOrR7YqKUt)
