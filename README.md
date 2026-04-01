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

<p align="center">
  <a href="https://seaberger.github.io/Foundry/paper/">Research Paper</a> &nbsp;|&nbsp;
  <a href="https://seaberger.github.io/Foundry/constitution/">The Madison Constitution</a> &nbsp;|&nbsp;
  <a href="https://seaberger.github.io/Foundry/training-results/">Training Results</a> &nbsp;|&nbsp;
  <a href="https://seaberger.github.io/Foundry/">Documentation</a>
</p>

---

*The documents survived. The voices didn't. Madison left behind every argument he made, every position he took. What was lost is the reasoning: the way he built from historical precedent before invoking principle, the way he held tension between federal power and liberty. The Foundry is an attempt to recover it.*

The Foundry uses Constitutional AI and ORPO fine-tuning to capture the distinctive voices, reasoning patterns, and philosophical positions of key US Founding Fathers — starting with James Madison. Users can engage in one-on-one conversation with a historically grounded founder, or watch two founders debate modern topics from their documented perspectives.

## Why Madison First

Madison is the ideal first subject for character fine-tuning — and the natural starting point for a project that will eventually voice all the key founders. As principal architect of the Constitution, he wrote 29 Federalist Papers, kept the most detailed record of the Constitutional Convention, served as Secretary of State and President, and spent his retirement years defending the Union against nullification. His voice is documented across **468,000 words of primary sources** spanning 8 distinct registers — from polished political theory to sharp convention debate to candid private correspondence. The ground truth is extensive, the voice is distinctive, and authenticity is falsifiable against the historical record.

## Key Findings (March 2026)

Beyond building a working Madison voice model, this project has produced several findings relevant to the broader character fine-tuning community:

**1. Knowledge-voice decoupling.** Preference training (ORPO) transfers factual knowledge before voice register. With 475 pairs: knowledge score 6.4/10, voice score 1.4/10. Voice required 2.7× more targeted data to imprint. This finding has implications for all character training work.

**2. LoRA quantization fragility.** On Gemma 3 27B with rank-16 LoRA, the same fine-tune scores **8.52/10** (ORPO v4) at BF16 precision and **1.74/10** at GGUF Q4_K_M — a 4.9× degradation from quantization alone. The thin LoRA deltas are noise-floored by 4-bit rounding errors. Rank-64 on Qwen 3-32B avoids this fragility. This affects all low-rank LoRA fine-tunes deployed via GGUF, not just ours. ([Details](docs/eval-analysis-orpo-v4.md))

**3. Merged vs. adapter-on-base serving.** The same LoRA adapter produces *fundamentally different* output depending on serving method. A prompt that triggers **97% AI-speak character breaks** through the merged model path produces **0% breaks** when served via vLLM LoRA serving (adapter applied at inference time, never merged). Merging bakes the LoRA signal into the weight distribution where it interacts with RLHF safety attractors; adapter-on-base preserves the signal at full precision. ([Details](docs/inference-guide.md))

**4. RLHF safety vs. persona training topology.** The base model's safety training overpowers character fine-tuning on specific topic categories — identity ("describe your drives" → 97% break), moral complexity ("write about slavery" → 83% break), meta-self-description ("write a biography" → 55% break) — while leaving other topics virtually unaffected (0-6% break). This reveals discoverable structure in where safety alignment is strongest vs. weakest.

**5. Gemma 3 VLM architecture complications.** Gemma 3 27B is architecturally a vision-language model (ForConditionalGeneration) even for text-only use, creating cascading vLLM compatibility issues. Converting to ForCausalLM breaks the interleaved sliding window attention pattern. The working workaround is `limit_mm_per_prompt={"image": 0}`. Qwen 3-32B (pure ForCausalLM) avoids this entire class of issues and is now the production base model. ([Details](docs/inference-guide.md))

**6. ORPO→SFT structural incompatibility.** Subsequent SFT after ORPO training catastrophically destroys the ORPO-trained character signal — even at 100× lower learning rate with half the LoRA rank (2.0–2.2/10, down from 8.8). Root cause is structural: ORPO's monolithic objective encodes SFT and preference signal into the same parameter subspace with no reference anchor. A subsequent SFT stage overwrites this jointly-learned manifold completely. This contrasts with DPO→SFT pipelines, where DPO's KL constraint stores preferences as a delta from a reference model that SFT cannot fully displace. **Implication: ORPO trades extensibility for efficiency. Choose DPO if your pipeline requires a subsequent SFT stage.** ([Details](docs/training-methodology.md))

## Current Status

**Active development**

### Completed
- [x] Madison primary source corpus — 140 documents, 468K words, 8 voice registers
- [x] Madison constitution — 5K word character document from primary sources + 7 biographies
- [x] 36-prompt evaluation harness with LLM judge + prompt caching
- [x] DPO v1 → collapsed (replicated "Objective Matters" persona drift finding)
- [x] ORPO v3b → 3.41/10 (knowledge OK, voice failed — knowledge-voice decoupling)
- [x] ORPO v4 (Gemma 3 27B) → **8.52/10 corrected** (voice-targeted augmentation succeeded)
- [x] Infrastructure confound discovery — Ollama GGUF (1.74) vs Modal BF16 (8.52) was inference, not training
- [x] Judge scoring bug fix — Sonnet intermittently omits overall_score, fallback computation added
- [x] vLLM LoRA serving probe — adapter-on-base eliminates character breaks on sensitive topics
- [x] Introspection SFT data generated — 415 clean reflections + 19 dialogues (~459K tokens)
- [x] Qwen 3-32B validation — pure ForCausalLM, no VLM bugs, now production base model
- [x] ORPO v1 (Qwen 3-32B) → **8.81/10 corrected** (successful base model migration)
- [x] ORPO v2 (Qwen 3-32B) → **8.82/10 corrected**
- [x] ORPO R2 (Qwen 3-32B) → **8.97/10 corrected** (production model, v6 dataset — 1,498 pairs)
- [x] Judge bias fix — weighted average override eliminates systematic scoring bias
- [x] JSON parse repair — fixes judge output parse failures
- [x] SFT after ORPO proven catastrophic — ORPO's built-in SFT makes subsequent SFT harmful (abandoned)
- [x] GGUF quantization — Q4_K_M (18.4 GB) and Q5_K_M (21.6 GB) on Modal volume
- [x] Local deployment — Q5_K_M loaded in LM Studio on Mac Mini M4 Pro (64 GB)
- [x] Learning rate sweep — lr=2e-5 optimal, contradicts ORPO paper's lr=8e-6 recommendation
- [x] [Documentation site](https://seaberger.github.io/Foundry/) — MkDocs Material with Distill-style research paper

### Next Steps
- [ ] Evaluate GGUF Q5_K_M quality vs BF16 baseline (rank-64 should survive quantization better than rank-16)
- [ ] Hamilton character development
- [ ] Chamber chat demo

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

### Preference Optimization Research (March 2026)

Beyond the Lambert/Maiya baseline, our pipeline incorporates recent findings:

- **Chosen quality dominates** — Teacher response quality is the #1 factor in preference optimization success; rejected quality barely matters ([Pan et al., NeurIPS 2025](https://arxiv.org/html/2508.18312v1))
- **Difficulty filtering** — Remove hardest 15-20% of pairs after first training pass ([ICML 2025](https://arxiv.org/html/2502.09650v1))
- **Learning rate sensitivity** — Our sweep found lr=2×10⁻⁵ optimal for character imprinting on Qwen 3-32B, contradicting the ORPO paper's recommended lr=8×10⁻⁶. Lower LR disproportionately sacrifices factual grounding while voice categories remain robust.

### Future Research: Pulpit to Parchment

Madison was born Anglican and educated Presbyterian at Princeton under John Witherspoon, whose lectures on moral philosophy blended Scottish Reformed theology with Aristotelian civic virtue. "If men were angels, no government would be necessary" is Reformed doctrine about human corruption translated into institutional design. A natural extension of the Foundry methodology is tracing the intellectual genealogy of American constitutional thought through its theological sources — training character models for Witherspoon, Jonathan Edwards, and George Whitefield, and examining how denominational competition among colonial colleges (Harvard Congregationalism, Princeton Presbyterianism, William and Mary Anglicanism) shaped systematically different political philosophies in Adams, Madison, and Jefferson. Documentation: [`docs/future-research-denominational-roots.md`](docs/future-research-denominational-roots.md)

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
(Sonnet 4.6 as Madison              (Base Qwen 3-32B, no persona
 with constitution)                   "what Madison wouldn't say")
    |                                     |
    v                                     v
         Preference Pair Construction
         (format_dpo.py — quality filtered)
              |
              v
    QLoRA ORPO Training on Modal A100
    (beta=0.1, lr=2e-5, rank=64)
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
| **James Madison** | ORPO R2 — **8.97/10** (production) | 468K words, 140 documents |
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
  scripts/modal/   — Modal A100 ORPO training, merge, GGUF conversion
  scripts/data/    — Training data generation, judge evaluation, dataset assembly
  scripts/util/    — Citation linkification, utilities
```

## Tech Stack

- **Training:** Unsloth + QLoRA ORPO on Modal A100-80GB
- **Base Model:** Qwen 3-32B
- **Serving:** vLLM with LoRA serving mode (adapter-on-base, best quality) or merged 16-bit model
- **Teacher Model:** Claude Sonnet 4.6 with prompt caching
- **Evaluation:** LLM-as-judge (Sonnet 4.6) with constitutional rubric + prompt caching (~$0.50 per 36-prompt eval)
- **Experiment Tracking:** [Weights & Biases](https://wandb.ai/sbergman/foundry)
- **Local Inference:** LM Studio on Mac Mini M4 Pro 64 GB (GGUF Q5_K_M, 21.6 GB — rank-64 LoRA, quality eval pending)
- **Web:** FastAPI + HTMX + SSE streaming, SQLite (WAL mode)
- **Voice:** ElevenLabs per-character voice profiles (planned)

## References

- Maiya, S., Bartsch, R., Lambert, N., Hubinger, E. (2025). [Open Character Training](https://arxiv.org/abs/2511.01689). arXiv:2511.01689. [Code](https://github.com/maiush/OpenCharacterTraining)
- Shao, Y., Li, L., Dai, J., Qiu, X. (2023). [Character-LLM](https://arxiv.org/abs/2310.10158). EMNLP 2023.
- Lambert, N. (2025). [The RLHF Book](https://rlhfbook.com/). Chapters 17 and 19.
- Pan, Y., et al. (2025). [What Matters in Data for DPO?](https://arxiv.org/html/2508.18312v1). NeurIPS 2025.
- Gemma 3 Ablations. (2025). [Distillation in Practice](https://huggingface.co/blog/tawnymanticore/gemma3-ablations). HuggingFace.
- Hunt, G. (ed.) (1900-1910). *The Writings of James Madison*, 9 vols. [Online Library of Liberty](https://oll.libertyfund.org/titles/madison-the-writings-of-james-madison-9-vols).
- Fernando, H., et al. (2025). [Understanding Forgetting in LLM Supervised Fine-Tuning and Preference Learning](https://arxiv.org/abs/2410.15483). ICLR 2025.
- Askell, A. et al. (2023). [Claude's Character](https://www.anthropic.com/research/claude-character). Anthropic.

## License

MIT
