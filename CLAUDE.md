# The Foundry Project

## What This Is
Fine-tuned character AI that brings America's Founding Fathers into modern conversation. Users chat with or watch debates between historical figures, each voiced by an ORPO fine-tuned model trained on their actual writings.

## Architecture
- **Chamber** (`src/foundry/chamber/`) — Chat and debate web UI. FastAPI + HTMX + SSE streaming. Session persistence in SQLite.
- **Press** (`src/foundry/press/`) — Training data pipeline. Converts historical writings to ShareGPT format, generates DPO pairs.
- **Characters** (`src/foundry/characters/`) — Character card loading, validation. YAML profiles in `config/characters/`.
- **Inference** (`src/foundry/inference/`) — Model serving, LoRA adapter loading/swapping on shared Qwen 3-32B base.
- **Voice** (`src/foundry/voice/`) — ElevenLabs TTS per character.

## Tech Stack
- Python 3.12+, `uv` for package management
- FastAPI + HTMX + Jinja2 templates
- SQLite (WAL mode) for persistence
- Pydantic for config and data models
- PyYAML for character cards
- httpx for async HTTP
- Modal for serverless GPU (training + inference)
- Unsloth + QLoRA for fine-tuning

## Commands
```bash
uv run foundry serve                    # Start web server
uv run foundry serve --no-browser       # Without auto-opening browser
uv run pytest                           # Run tests
uv run ruff check src/                  # Lint
```

## Config
- `config/foundry.yaml` — Server, inference, debate settings
- `config/characters/*.yaml` — Character profiles (Madison, Hamilton, etc.)

## Development Principles
1. **Chat first, debate later.** Get one character (Madison) working in conversational mode before building debate infrastructure.
2. **Clean public repo.** This is open source. No private references, no sensitive content.
3. **Character accuracy matters.** The voice must be distinguishably Madison's, not generic "old-timey" speech. Test against actual writings.
4. **Constitutional AI + ORPO** for training, per Lambert & Maiya (arXiv:2511.01689). Qwen 3-32B base. ORPO R2 scores 8.97/10 corrected.
5. **Blog the journey.** Document learnings on sbergman.net. Build in public.

## Character Cards
Character YAML files define: voice (tone, speech patterns, vocabulary), personality (traits, positions), key writings, rhetorical patterns, intellectual influences, and the system prompt.

When adding a new character:
1. Create `config/characters/{name}.yaml`
2. Include documented positions, key writings, and rhetorical patterns
3. Write a system prompt that grounds the character in their actual historical voice
4. Add to the README character table

## Training Data (Press Pipeline)
Training data comes from three tiers:
1. **Primary sources** — Actual writings converted to conversational format
2. **Synthetic debates** — Claude-generated DPO pairs in character voice
3. **Self-reflection** — On-policy data where the model reasons in character. *Note: SFT stage abandoned — ORPO's monolithic loss structure makes subsequent SFT catastrophically destructive to character signal.*

## Database
SQLite at `data/foundry.db`. Tables: sessions, turns, character_knowledge, schema_info.
