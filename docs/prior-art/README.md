# Prior Art Research

Research survey conducted March 24, 2026 confirming the novelty of the Foundry Loop — recursive self-improvement applied to character fine-tuning via the autoresearch pattern.

## Key Finding

Nobody has applied Karpathy's autoresearch recursive loop to character/persona fine-tuning. Three mature research areas exist independently but have never been bridged:

1. **Autoresearch** — Agent-driven recursive experiment loops (pretraining only)
2. **Automated DPO** — Hyperparameter sweeps (grid search, no agent reasoning)
3. **Iterative Persona Training** — Fixed-stage pipelines (no open-ended self-improvement)

## Papers by Relevance to Foundry

### Must-Cite (Directly Inform Our Pipeline)

| Paper | arxiv | Year | Why It Matters |
|-------|-------|------|----------------|
| Open Character Training | 2511.01689 | 2025 | Our methodological foundation (Lambert/Maiya) |
| DeePer: Directed Persona Refinement | 2502.11078 | 2025 | Closest architectural match — iterative DPO for persona |
| MentalArena | 2410.06845 | 2024 | Proves self-play persona loop works (GPT-3.5 beats GPT-4o) |
| AI Founding Fathers | 2511.09005 | 2025 | Our exact domain (Madison/Hamilton/Jefferson) via RAG |
| SPIN: Self-Play Fine-Tuning | 2401.01335 | 2024 | Foundation for recursive loop mechanism |
| DPO Causes Persona Drift | 2601.12639 | 2026 | Critical guardrail: DPO drift at large budgets |

### Should-Cite (Adjacent Techniques)

| Paper | arxiv | Year | Why It Matters |
|-------|-------|------|----------------|
| SDPO: Self-Distillation | 2601.20802 | 2026 | Dense learning signal without external reward model |
| SPPO: Self-Play Preference | 2405.00675 | 2024 | Convergence guarantees for iterative DPO |
| PCL: Persona-Aware Contrastive | 2503.17662 | 2025 | Self-play contrastive alignment for persona |
| OpenCharacter | 2501.15427 | 2025 | Large-scale synthetic character data |
| CoSER | 2502.09082 | 2025 | Acting methodology evaluation framework |
| What Matters in DPO (NeurIPS) | 2508.18312 | 2025 | Chosen quality dominates |
| Difficulty Filtering (ICML) | 2502.09650 | 2025 | Remove hardest 15-20% of pairs |
| SimPO | 2405.14734 | 2024 | Reference-free preference optimization |

### Awareness (Future Reference)

| Paper | arxiv | Year | Why It Matters |
|-------|-------|------|----------------|
| RISE: Recursive IntroSpection | 2407.18219 | 2024 | Self-correction mechanism |
| CharacterGPT | 2405.19778 | 2024 | Dynamic persona reconstruction |
| Inverse Constitutional AI | 2406.06560 | 2024 | Auto-discover constitution from examples |
| Beta-DPO | 2407.08639 | 2024 | Dynamic beta per batch |
| Curry-DPO | 2403.07230 | 2024 | Curriculum ordering for DPO |

## Academic Venue

**ICLR 2026 Workshop on Recursive Self-Improvement** — [recursive-workshop.github.io](https://recursive-workshop.github.io/)

## Files in This Directory

- `README.md` — This index
- `novelty-confirmation.md` — Full prior art survey with three-island analysis
- `deep-dive-deeper-mentalarena.md` — Detailed analysis of DeePer + MentalArena + persona drift
- `deep-dive-founding-fathers.md` — Detailed analysis of Chauhan + OpenCharacter + CoSER
- `last30days-research.md` — Raw last30days research output
