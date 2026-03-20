# The Foundry Project

**Fine-tuned character AI that brings America's Founding Fathers into modern conversation.**

The Foundry Project uses Constitutional AI and LoRA fine-tuning to capture the distinctive voices, reasoning patterns, and philosophical positions of key US Founding Fathers. Users can watch historical figures debate modern topics or engage in one-on-one conversation with a founder.

## What This Is

Imagine James Madison and Alexander Hamilton debating digital privacy, executive power, or AI regulation — each drawing from their documented positions, rhetorical style, and philosophical framework. Madison approaches privacy through his Fourth Amendment lens. Hamilton argues from federal power and commerce. Their voices are distinct because they're trained on their actual writings, not prompted to role-play.

Two modes:
- **Debate** — Two founders argue a modern topic with a moderator. Lincoln-Douglas format.
- **Chat** — One-on-one conversation with a historical figure. Ask Madison about the Constitution. Challenge Hamilton on federal debt.

## Architecture

```
foundry/
  src/foundry/
    chamber/     — Debate and chat UI (FastAPI + HTMX, SSE streaming)
    press/       — Training data pipeline (source collection, conversion, DPO pair generation)
    characters/  — Founder profiles, constitutional prompts, LoRA configs
    inference/   — Model serving, LoRA adapter loading and hot-swapping
    voice/       — Text-to-speech per character (ElevenLabs)
  config/
    characters/  — Character YAML files (voice, positions, rhetorical patterns)
  templates/     — Jinja2 templates for web UI
  static/        — CSS, JS
```

### Key Components

- **Chamber** — The interaction platform. Real-time streaming chat and structured debate with session persistence.
- **Press** — Training data pipeline. Collects historical writings, converts to conversational format, generates DPO training pairs.
- **Characters** — YAML profiles defining each founder's voice, documented positions, vocabulary, and constitutional beliefs.
- **Inference** — LoRA adapter management on a shared base model. Hot-swap between characters per debate turn.

## Technical Approach

Based on research from [Lambert & Maiya (2025)](https://arxiv.org/abs/2511.01689) — *Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI*:

1. **Constitutional AI + DPO** — Character-specific constitutional prompts generate preferred responses. Direct Preference Optimization trains against base model outputs.
2. **Self-reflection SFT** — On-policy data where the model reasons in character, grounding responses in documented historical positions.
3. **Base model: Gemma 3 27B** — Research shows Gemma takes character imprinting better than Qwen or Llama variants.

Training data combines:
- Primary sources (Federalist Papers, Convention notes, correspondence)
- Claude-generated synthetic debates in character voice
- Historical debate transcripts converted to conversational format

## Characters

| Founder | Priority | Source Material |
|---------|----------|----------------|
| **James Madison** | Primary | Federalist Papers (29 essays), Constitutional Convention notes, presidential correspondence |
| **Alexander Hamilton** | Secondary | Federalist Papers (51 essays), Treasury reports, personal letters |
| Thomas Jefferson | Planned | Declaration of Independence, Notes on Virginia, presidential correspondence |
| John Adams | Planned | Defence of the Constitutions, presidential correspondence, letters to Jefferson |
| Benjamin Franklin | Planned | Autobiography, Poor Richard's Almanack, diplomatic correspondence |

## Status

Early development. Building training data pipeline and first LoRA fine-tune for Madison.

## Tech Stack

- Python 3.12+, managed with `uv`
- FastAPI + HTMX for web interface
- SQLite for session persistence
- Unsloth + QLoRA for fine-tuning
- Modal (serverless A100) for training and inference
- ElevenLabs for voice synthesis
- Pydantic for data models

## License

MIT
