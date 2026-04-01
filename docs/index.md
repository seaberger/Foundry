# The Foundry Project

*The documents survived. The voices didn't.*

*Fine-tuned LLMs to preserve and reanimate the rhetorical voices of America's founders for civic education and constitutional discourse.*

---

## About

The Foundry uses Constitutional AI and ORPO fine-tuning to capture the distinctive voices, reasoning patterns, and philosophical positions of key US Founding Fathers — starting with James Madison. Our production model (ORPO R2 on Qwen 3-32B) scores **8.97/10** on a 36-prompt behavioral evaluation harness.

## Key Documents

<div class="doc-cards" markdown>
<div class="doc-card" markdown>

### [The Madison Constitution](constitution.md)

5,000-word first-person character document synthesized from 468,000 words of primary sources and 1.8 million words of scholarly biography. The richest character constitution ever used for LLM fine-tuning.

</div>
<div class="doc-card" markdown>

### [Research Paper](paper.md)

Full methodology and findings: knowledge-voice decoupling, LoRA quantization fragility, the structural incompatibility between ORPO and subsequent SFT stages — a finding that directly affects anyone designing multi-stage character training pipelines — learning rate sensitivity, and source-enriched data generation.

</div>
<div class="doc-card" markdown>

### [Training Results](training-results.md)

Comprehensive record of every training run from DPO v1 through ORPO R2 with per-run configs, category scores, and detailed analysis.

</div>
<div class="doc-card" markdown>

### [Scoring Methodology](scoring-methodology.md)

How we evaluate models: 5 weighted dimensions, LLM judge with constitutional rubric, weighted average override, and JSON parse repair.

</div>
</div>

## Current Status

| Model | Score | Dataset | Date |
|-------|:-----:|:-------:|:----:|
| **Qwen 3-32B R2 (production)** | **8.97** | 1,498 pairs | 2026-03-31 |
| Qwen 3-32B v1 | 8.81 | 1,273 pairs | 2026-03-29 |
| Gemma 3 27B v4 | 8.52 | 1,273 pairs | 2026-03-28 |
| Gemma 3 27B v3b | 3.41 | 475 pairs | 2026-03-26 |

*Score: 36-prompt LLM judge evaluation (Sonnet 4.6), 1–10 scale, weighted average of 5 dimensions. See [Scoring Methodology](scoring-methodology.md).*

## Source Code

[View on GitHub](https://github.com/seaberger/Foundry){ .md-button }
