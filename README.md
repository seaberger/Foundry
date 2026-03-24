# The Foundry Project

**Fine-tuned character AI that brings America's Founding Fathers into modern conversation.**

The Foundry uses Constitutional AI and LoRA fine-tuning to capture the distinctive voices, reasoning patterns, and philosophical positions of key US Founding Fathers — starting with James Madison. Users can engage in one-on-one conversation with a historically grounded founder, or watch two founders debate modern topics from their documented perspectives.

## Why This Works

Madison wrote 29 Federalist Papers, kept the most detailed record of the Constitutional Convention, served as Secretary of State, President, and spent his retirement years defending the Union against nullification. His voice is documented across 468,000 words of primary sources spanning 8 distinct registers — from polished political theory to sharp convention debate to candid private correspondence. This makes him an ideal subject for character fine-tuning: the ground truth is extensive, the voice is distinctive, and authenticity is verifiable against the historical record.

## Research Approach

Our methodology builds on pioneering work in character training, particularly:

- **Sharan Maiya, Nathan Lambert, et al.** — [*Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI*](https://arxiv.org/abs/2511.01689) (2025). Their two-stage pipeline (Constitutional AI + DPO distillation, followed by introspection SFT) demonstrated that character can be reliably imprinted through synthetic data generated from constitutional seed documents. Their [open-source code and datasets](https://github.com/maiush/OpenCharacterTraining) made this work possible.

- **Yunfan Shao, Linyang Li, et al.** — [*Character-LLM: A Trainable Agent for Role-Playing*](https://arxiv.org/abs/2310.10158) (EMNLP 2023). Their "Experience Reconstruction" approach showed that historical figures (Beethoven, Caesar, Socrates) could be trained from biographical profiles, establishing the feasibility of historical character fine-tuning.

- **Nathan Lambert** — [*The RLHF Book*](https://rlhfbook.com/), Chapters 17 and 19. Lambert's framework for understanding character training as a subset of post-training — focused on *manner* rather than content — shaped our thinking about what a constitution should encode. His [Interconnects newsletter](https://www.interconnects.ai/) coverage of the character training pipeline was invaluable.

- **Amanda Askell (Anthropic)** — Her test-driven development methodology for character training — write behavioral tests *before* generating training data — is central to our evaluation approach.

We are immensely grateful to these researchers for sharing their work openly. The Foundry would not exist without it.

### What We Do Differently: The Rich Constitution

Prior character training work used either minimal trait lists (~10 first-person declarations like "I am gentle and supportive") or Wikipedia-derived biographical profiles. For a historical figure with documented positions that evolved over 50 years, contradictions he struggled with, and a voice that shifts across 8 registers, this isn't enough.

Our Madison constitution is a **5,000-word first-person character document** synthesized from:

- **468,000 words of primary sources** — Federalist Papers, political essays, convention and congressional speeches, presidential papers, legislative writings, and private correspondence. Collected from Founders Online, Yale Avalon Project, Project Gutenberg, the Hunt 9-volume collected works, and the Congressional Edition.

- **1.8 million words of scholarly biography** — Ralph Ketcham's definitive *James Madison*, Noah Feldman's *Three Lives of James Madison*, Lynne Cheney's *Madison: A Life Reconsidered*, Andrew Burstein & Nancy Isenberg's *Madison and Jefferson*, Stuart Leibiger's *Founding Friendship*, Joseph Ellis's *Founding Brothers*, and Ron Chernow's *Alexander Hamilton*. These provide the interpreted, synthesized view of Madison's personality, temperament, intellectual evolution, and relationships that his own writings don't always reveal.

The constitution covers 9 sections: identity and intellectual evolution, core philosophical positions, the slavery contradiction, rhetorical patterns and voice, how contemporaries described him, key relationships (Jefferson, Hamilton, Washington, Patrick Henry), 8 voice registers, his private voice, and boundaries/anti-patterns. Every claim is traceable to a primary source or scholarly biography.

This is, to our knowledge, the richest character constitution ever used for LLM fine-tuning — 50x more detailed than prior work, grounded in scholarship rather than Wikipedia, and designed to capture a real historical figure's full complexity rather than a generic persona.

## Pipeline

```
Primary Sources (468K words) + Scholarly Biographies (1.8M words)
    ↓
Rich Constitution (5K words, 9 sections)
    ↓
Behavioral Tests (TDD — define "authentic Madison" before training)
    ↓
Prompt Generation (500-750 diverse questions from constitution traits)
    ↓
Teacher Model (generates in-character Madison responses with constitution)
Student Model (base Gemma 3 27B, no constitution — "what Madison wouldn't say")
    ↓
DPO Training on Modal A100 (QLoRA via Unsloth)
    ↓
Introspection SFT (self-reflection + self-interaction)
    ↓
Evaluation vs. Behavioral Tests → Iterate
    ↓
Deploy LoRA Adapter to Chamber UI
```

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

Sources include the Federalist Papers, National Gazette essays, Helvidius letters, Memorial and Remonstrance, Virginia Resolutions, Report of 1800, VA Ratifying Convention speeches (debating Patrick Henry), Congressional speeches (opposing Hamilton's bank, Jay Treaty), presidential vetoes and War Message, and private letters to Jefferson, Roane, and others.

## Characters

| Founder | Status | Source Material |
|---------|--------|----------------|
| **James Madison** | Active — corpus collected, constitution drafted | 468K words across 140 documents |
| **Alexander Hamilton** | Character card complete, corpus pending | Federalist Papers (51 essays), Treasury reports |
| Thomas Jefferson | Planned | Declaration, Notes on Virginia, correspondence |
| John Adams | Planned | Defence of the Constitutions, correspondence |
| Benjamin Franklin | Planned | Autobiography, Poor Richard's, diplomatic letters |

## Progress

- [x] Repository scaffold, config system, database layer
- [x] Madison and Hamilton character cards with system prompts
- [x] Chamber chat UI — streaming conversations with prompted Madison on Gemma 3 27B
- [x] Madison primary source corpus — 140 documents, 468K words, 8 voice registers
- [x] Madison constitution — 5K word character document from primary sources + 7 biographies
- [x] Training methodology documented — 10-step pipeline with cost estimates
- [ ] Behavioral test suite (Step 1)
- [ ] Press pipeline: prompt generation, teacher/student responses, DPO pair construction
- [ ] First QLoRA DPO fine-tune on Modal A100
- [ ] Evaluation and iteration
- [ ] Madison chat demo for OSV Fellowship application (deadline: April 30, 2026)

## Architecture

```
foundry/
  src/foundry/
    chamber/       — Chat and debate UI (FastAPI + HTMX, SSE streaming)
    press/         — Training data pipeline (prompt gen, teacher/student, DPO formatting)
    characters/    — Founder profiles and character card loading
    inference/     — Model serving via OpenAI-compatible API (LM Studio, Modal)
    voice/         — Text-to-speech per character (ElevenLabs, planned)
  config/
    characters/    — Character YAML cards (voice, positions, rhetorical patterns)
    constitutions/ — Character constitutions for fine-tuning pipeline
  sources/         — Primary source corpus (public domain historical texts)
  docs/            — Training methodology, constitution plan, research files
  templates/       — Jinja2 templates for web UI
  static/          — CSS, JS
```

## Tech Stack

- Python 3.12+
- FastAPI + HTMX + SSE streaming for web interface
- SQLite (WAL mode) for session persistence
- Unsloth + QLoRA for efficient LoRA fine-tuning
- Modal (serverless A100 80GB) for training runs
- LM Studio for local inference (Gemma 3 27B)
- LiteParse + docling for PDF text extraction
- Pydantic for configuration and data models
- ElevenLabs for voice synthesis (planned)

## References

- Maiya, S., Bartsch, R., Lambert, N., Hubinger, E. (2025). [Open Character Training](https://arxiv.org/abs/2511.01689). arXiv:2511.01689. [Code](https://github.com/maiush/OpenCharacterTraining) | [Data](https://huggingface.co/datasets/maius/OpenCharacterTraining-data)
- Shao, Y., Li, L., Dai, J., Qiu, X. (2023). [Character-LLM: A Trainable Agent for Role-Playing](https://arxiv.org/abs/2310.10158). EMNLP 2023. [Code](https://github.com/choosewhatulike/trainable-agents)
- Lambert, N. (2025). [The RLHF Book](https://rlhfbook.com/). Chapters 17 (Product & Character) and 19 (Character Training).
- Lambert, N. (2025). [Opening the Character Training Pipeline](https://www.interconnects.ai/p/opening-the-black-box-of-character). Interconnects.
- Askell, A. et al. (Anthropic). [Claude's Character](https://www.anthropic.com/research/claude-character).
- Hunt, G. (ed.) (1900-1910). *The Writings of James Madison*, 9 vols. G.P. Putnam's Sons. [Online Library of Liberty](https://oll.libertyfund.org/titles/madison-the-writings-of-james-madison-9-vols).

## License

MIT
