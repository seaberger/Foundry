# Integration notes

## What the wrapper assumes

`autoresearch_qwen/train.py` assumes:

1. You are running from the Foundry repo root.
2. The v6 preference dataset exists at `data/training/madison-orpo-v6.jsonl`.
3. Candidate training runs are launched through the Modal backend in `autoresearch_qwen/backend/`.
4. Candidate evaluation happens through Foundry's existing `foundry.press.evaluate` path.

## What you must still wire

### 1. Candidate activation / serving

The wrapper cannot know how you expose a freshly trained adapter for evaluation.

That step is infra-specific. You need to decide one of these:

- swap the active adapter behind a stable local vLLM endpoint
- launch a new temporary endpoint per candidate
- evaluate through a custom local script that knows how to load the adapter

Then put that command into:

```python
ACTIVATE_CANDIDATE_COMMAND_TEMPLATE = "..."
```

in `train.py`.

### 2. Probe evaluation file

Replace `eval/probe-prompts.jsonl` with a real held-out subset.

Recommended shape:

- 12 to 16 prompts total
- 4 to 5 GT-heavy prompts
- 3 to 4 verified-response prompts
- 2 to 3 private-voice guard prompts
- 2 regression-guard prompts from strong categories

### 3. Manifest selectors

The sample manifests support both:

- metadata selectors
- prompt substring selectors

If your v6 dataset metadata uses different field names, update the manifest or the matching helpers in `train.py`.

## Suggested command sequence

```bash
python autoresearch_qwen/train.py
```

If you only want to materialize the dataset first, set:

```python
RUN_MODE = "dataset-only"
```

Then inspect `autoresearch_qwen/runs/<tag>/train.jsonl`.

## Notes on the backend script

The backend script is based on the existing Foundry Modal Qwen ORPO trainer, but adds:

- `max_steps`
- `warmup_ratio`
- explicit `alpha`
- explicit `dropout`
- explicit batch / grad accumulation flags

That makes it usable for step-capped probe runs.
