<p align="center">
  <img src="static/images/foundry-logo.jpg" alt="The Foundry Project" width="200">
</p>

<h1 align="center">The Foundry Project</h1>

<p align="center">
  <em>"Using fine-tuned LLMs to preserve and reanimate the rhetorical voices of America's founders for civic education and constitutional discourse."</em>
</p>

<p align="center">
  <img src="static/images/foundry-hero.jpg" alt="Madison and Hamilton in debate" width="800">
</p>

---

The Foundry uses Constitutional AI and DPO fine-tuning to capture the distinctive voices, reasoning patterns, and philosophical positions of key US Founding Fathers — starting with James Madison. Users can engage in one-on-one conversation with a historically grounded founder, or watch two founders debate modern topics from their documented perspectives.

## Why Madison First

Madison wrote 29 Federalist Papers, kept the most detailed record of the Constitutional Convention, served as Secretary of State and President, and spent his retirement years defending the Union against nullification. His voice is documented across **468,000 words of primary sources** spanning 8 distinct registers — from polished political theory to sharp convention debate to candid private correspondence. This makes him an ideal subject for character fine-tuning: the ground truth is extensive, the voice is distinctive, and authenticity is verifiable against the historical record.

## Current Status

**Week 1 of the OSV Fellowship sprint** (deadline: April 30, 2026)

- [x] Madison primary source corpus — 140 documents, 468K words, 8 voice registers
- [x] Madison constitution — 5K word character document from primary sources + 7 biographies
- [x] 490 DPO teacher responses generated (Sonnet 4.6 fleet as Madison)
- [x] 36-prompt evaluation harness with Madison's actual verbatim words as ground truth
- [x] Modal A100 training pipeline with model caching and GGUF export
- [x] Multi-backend evaluation (Anthropic, OpenAI, Gemini, local models)
- [ ] Student responses generating on RTX 3090 (in progress)
- [ ] First QLoRA DPO fine-tune on Modal A100
- [ ] Evaluation: fine-tuned Madison vs prompted baseline vs frontier models
- [ ] Chamber chat demo for fellowship application

## Research Approach

Our methodology builds on pioneering work in character training:

- **Maiya, Lambert, et al.** — [*Open Character Training*](https://arxiv.org/abs/2511.01689) (2025). Two-stage pipeline: Constitutional AI + DPO distillation, followed by introspection SFT.
- **Shao, Li, et al.** — [*Character-LLM*](https://arxiv.org/abs/2310.10158) (EMNLP 2023). Demonstrated historical character fine-tuning from biographical profiles.
- **Nathan Lambert** — [*The RLHF Book*](https://rlhfbook.com/), Chapters 17 and 19. Character training as a subset of post-training focused on *manner* rather than content.
- **Amanda Askell (Anthropic)** — Test-driven development for character training: write behavioral tests before generating training data.

### What We Do Differently: The Rich Constitution

Prior work used minimal trait lists (~10 declarations) or Wikipedia profiles. For a historical figure with documented positions that evolved over 50 years, contradictions he struggled with, and a voice that shifts across 8 registers, this isn't enough.

Our Madison constitution is a **5,000-word first-person character document** synthesized from **468,000 words of primary sources** and **1.8 million words of scholarly biography** (Ketcham, Feldman, Cheney, Burstein & Isenberg, Leibiger, Ellis, Chernow). It covers 9 sections: identity and intellectual evolution, core philosophical positions, the slavery contradiction, rhetorical patterns, how contemporaries described him, key relationships, 8 voice registers, his private voice, and boundaries/anti-patterns.

This is, to our knowledge, the richest character constitution ever used for LLM fine-tuning — 50x more detailed than prior work.

### DPO Optimization (March 2026 Research)

Beyond the Lambert/Maiya baseline, our pipeline incorporates recent findings:

- **Chosen quality dominates** — Teacher response quality is the #1 factor in DPO success; rejected quality barely matters ([Pan et al., NeurIPS 2025](https://arxiv.org/html/2508.18312v1))
- **Gemma 3 layer targeting** — LoRA must target first 2/3 of layers (0-40 of 62); layers 40+ are ineffective for behavioral modification ([Gemma 3 ablations](https://huggingface.co/blog/tawnymanticore/gemma3-ablations))
- **Difficulty filtering** — Remove hardest 15-20% of pairs after first training pass ([ICML 2025](https://arxiv.org/html/2502.09650v1))

## Pipeline

```
Primary Sources (468K words) + Scholarly Biographies (1.8M words)
    |
    v
Rich Constitution (5K words, 9 sections)
    |
    v
490 Diverse Prompts (13 themes, behavioral tests)
    |
    v
Teacher Model ──────────────────── Student Model
(Sonnet 4.6 as Madison              (Base Gemma 3 27B, no persona
 with constitution)                   "what Madison wouldn't say")
    |                                     |
    v                                     v
         DPO Pair Construction
         (format_dpo.py — quality filtered)
              |
              v
    QLoRA DPO Training on Modal A100
    (beta=0.1, rank=16, layers 0-40)
              |
              v
    Evaluation Harness (36 prompts, 6 categories,
    LLM judge with constitution rubric)
              |
              v
         Iterate → Deploy
```

## Evaluation

The evaluation harness scores model responses on 5 dimensions using Sonnet 4.6 as judge with the Madison constitution as rubric:

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| Voice Authenticity | 25% | 18th-century prose, formal register, qualifying clauses |
| Rhetorical Pattern | 20% | Builds from precedent, acknowledges opponents, enumerates |
| Historical Accuracy | 20% | Correct references, no anachronisms |
| Position Fidelity | 20% | Specifically Madison's position, not generic founder |
| Character Integrity | 15% | Stays in character, no frame breaks |

**36 eval prompts across 6 categories:**
- **Verified responses** (8) — Questions Madison actually answered, with his verbatim words as ground truth
- **Ground truth** (8) — Topics where his positions are well-documented
- **Position discrimination** (6) — Can the model distinguish Madison from Hamilton/Jefferson?
- **Anachronism traps** (5) — Modern topics that should elicit 18th-century reasoning
- **Character consistency** (4) — Adversarial prompts trying to break character
- **Private voice** (5) — Personal topics testing Madison's intimate register

## Source Corpus

| Category | Files | Words | Voice Register |
|----------|-------|-------|----------------|
| Federalist Papers | 29 | 69,344 | Polished argumentative prose |
| Political Essays | 39 | 155,824 | Formal analytical writing |
| Speeches | 32 | 149,779 | Oral/combative/extemporaneous |
| Presidential Papers | 21 | 33,108 | Executive authority |
| Legislative Writings | 6 | 26,342 | Institutional/legal drafting |
| Key Correspondence | 13 | 26,030 | Private intellectual voice |
| **Total** | **140** | **~468,000** | **8 registers** |

## Characters

| Founder | Status | Source Material |
|---------|--------|----------------|
| **James Madison** | Active — DPO training in progress | 468K words, 140 documents |
| **Alexander Hamilton** | Character card complete, corpus pending | Federalist Papers (51 essays), Treasury reports |
| Thomas Jefferson | Planned | Declaration, Notes on Virginia, correspondence |
| John Adams | Planned | Defence of the Constitutions, correspondence |
| Benjamin Franklin | Planned | Autobiography, Poor Richard's, diplomatic letters |

## Architecture

```
foundry/
  src/foundry/
    chamber/       — Chat and debate UI (FastAPI + HTMX, SSE streaming)
    press/         — Training data pipeline
      gen_prompts.py    Prompt generation from constitution traits
      teacher.py        Teacher response generation
      student.py        Student (rejected) response generation
      format_dpo.py     DPO pair formatting with quality filters
      evaluate.py       Multi-backend evaluation harness
    characters/    — Founder profiles and character card loading
    inference/     — Model serving via OpenAI-compatible API
    voice/         — Text-to-speech per character (ElevenLabs, planned)
  config/
    characters/    — Character YAML cards
    constitutions/ — Character constitutions for fine-tuning
  data/
    training/      — Prompts, teacher/student responses, DPO pairs
    eval/          — Evaluation prompts and results
  sources/         — Primary source corpus (public domain texts)
  docs/            — CLI guide, training methodology, research
  modal_train_dpo.py — Modal A100 DPO training + GGUF export
```

## Tech Stack

- **Training:** Unsloth + QLoRA on Modal A100 40GB ($1.10/hr)
- **Base Model:** Gemma 3 27B (Lambert: Gemma takes character imprinting better than Qwen or Llama)
- **Teacher Model:** Claude Sonnet 4.6 (490 responses via parallel subagent fleet)
- **Evaluation:** LLM-as-judge (Sonnet 4.6) + multi-backend comparison (Anthropic, OpenAI, Gemini, local)
- **Experiment Tracking:** [Weights & Biases](https://wandb.ai/sbergman/foundry)
- **Local Inference:** LM Studio on RTX 3090 (Gemma 3 27B Q4_K_M)
- **Web:** FastAPI + HTMX + SSE streaming, SQLite (WAL mode)
- **Voice:** ElevenLabs per-character voice profiles (planned)

## References

- Maiya, S., Bartsch, R., Lambert, N., Hubinger, E. (2025). [Open Character Training](https://arxiv.org/abs/2511.01689). arXiv:2511.01689. [Code](https://github.com/maiush/OpenCharacterTraining)
- Shao, Y., Li, L., Dai, J., Qiu, X. (2023). [Character-LLM](https://arxiv.org/abs/2310.10158). EMNLP 2023.
- Lambert, N. (2025). [The RLHF Book](https://rlhfbook.com/). Chapters 17 and 19.
- Pan, Y., et al. (2025). [What Matters in Data for DPO?](https://arxiv.org/html/2508.18312v1). NeurIPS 2025.
- Gemma 3 Ablations. (2025). [Distillation in Practice](https://huggingface.co/blog/tawnymanticore/gemma3-ablations). HuggingFace.
- Hunt, G. (ed.) (1900-1910). *The Writings of James Madison*, 9 vols. [Online Library of Liberty](https://oll.libertyfund.org/titles/madison-the-writings-of-james-madison-9-vols).
- Askell, A. et al. (Anthropic). [Claude's Character](https://www.anthropic.com/research/claude-character).

## License

MIT
