# The Foundry Project

*Fine-tuned LLMs to preserve and reanimate the rhetorical voices of America's founders for civic education and constitutional discourse.*

---

## About

The Foundry uses Constitutional AI and ORPO fine-tuning to capture the distinctive voices, reasoning patterns, and philosophical positions of key US Founding Fathers — starting with James Madison. Our production model (ORPO R2 on Qwen 3-32B) scores **8.97/10** on a 36-prompt behavioral evaluation harness.

## Key Documents

<div class="grid cards" markdown>

-   :material-scroll-text:{ .lg .middle } **The Madison Constitution**

    ---

    5,000-word first-person character document synthesized from 468,000 words of primary sources and 1.8 million words of scholarly biography. The richest character constitution ever used for LLM fine-tuning.

    [:octicons-arrow-right-24: Read the Constitution](constitution.md)

-   :material-file-document:{ .lg .middle } **Research Paper**

    ---

    Full methodology and findings: knowledge-voice decoupling, LoRA quantization fragility, ORPO vs SFT, learning rate sensitivity, and source-enriched data generation.

    [:octicons-arrow-right-24: Read the Paper](paper.md)

-   :material-chart-line:{ .lg .middle } **Training Results**

    ---

    Comprehensive record of every training run from DPO v1 through ORPO R2 with per-run configs, category scores, and detailed analysis.

    [:octicons-arrow-right-24: View Results](training-results.md)

-   :material-scale-balance:{ .lg .middle } **Scoring Methodology**

    ---

    How we evaluate models: 5 weighted dimensions, LLM judge with constitutional rubric, weighted average override, and JSON parse repair.

    [:octicons-arrow-right-24: View Methodology](scoring-methodology.md)

</div>

## Current Status

| Model | Score | Dataset | Date |
|-------|:-----:|:-------:|:----:|
| **Qwen 3-32B R2 (production)** | **8.97** | 1,498 pairs | 2026-03-31 |
| Qwen 3-32B v1 | 8.81 | 1,273 pairs | 2026-03-29 |
| Gemma 3 27B v4 | 8.52 | 1,273 pairs | 2026-03-28 |
| Gemma 3 27B v3b | 3.41 | 475 pairs | 2026-03-26 |

## Source Code

[:fontawesome-brands-github: View on GitHub](https://github.com/seaberger/Foundry){ .md-button }
