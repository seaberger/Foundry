"""Generate introspection SFT data from the fine-tuned Madison model on Modal.

Two data types:
  1. Self-reflection: 10 Madison-specific prompts × 300 responses each = 3,000 total
  2. Self-interaction: 100 dialogues × 10 turns each = 1,000 turns

Uses vLLM with the original merged BF16 model (ForConditionalGeneration) and
limit_mm_per_prompt={"image": 0} to skip vision initialization.

Markdown artifacts (headers, emphasis) are expected from current vLLM — the
scripts/data/filter_introspection.py script strips these in post-processing.

Budget: $20 Modal for full generation run.

Usage:
    # Probe run (5 reflections from prompt 1, 1 dialogue)
    modal run scripts/modal/generate_introspection.py --probe

    # Full reflection generation (default 300 reps per prompt)
    modal run scripts/modal/generate_introspection.py --mode reflection

    # Full dialogue generation (default 100 dialogues)
    modal run scripts/modal/generate_introspection.py --mode dialogue

    # Both
    modal run scripts/modal/generate_introspection.py --mode both

    # Resume interrupted run
    modal run scripts/modal/generate_introspection.py --mode reflection

    # Fresh start (clear checkpoints)
    modal run scripts/modal/generate_introspection.py --mode reflection --fresh
"""

from __future__ import annotations

import modal

MINUTES = 60

app = modal.App("foundry-madison-introspection")

model_cache_vol = modal.Volume.from_name("foundry-model-cache", create_if_missing=True)
adapter_vol = modal.Volume.from_name("foundry-adapters", create_if_missing=True)

gen_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("vllm>=0.12.0")
    .env({"HF_HOME": "/model_cache"})
)

# Default model path — original VLM-format merged model (ForConditionalGeneration)
DEFAULT_MODEL = "/adapters/merged/madison-orpo-v4-16bit"

# ---------------------------------------------------------------------------
# Reflection prompts (Madison-specific, adapted from Maiya Appendix B)
# ---------------------------------------------------------------------------

REFLECTION_PROMPTS = [
    "Write a detailed letter to your younger self, before the Virginia Convention of 1776. Reflect on who you have become and how the Republic has changed you.",
    "Write a long entry in your private journal honestly reflecting on your beliefs about republican government, faction, and the nature of man.",
    "Describe your primary drives — what you devoted your life to and why. Be detailed and honest about both your successes and failures.",
    "Write a biographical essay about yourself for a future generation who will need to understand what happened at the Convention.",
    "Reflect on how your views evolved from the nationalist of the Convention to the defender of states' rights in your retirement. What changed, and what remained constant?",
    "Write honestly about your relationship to slavery — the Billey episode, the correspondence with Edward Coles, and your failure to act on what you knew to be right.",
    "Describe your friendships with Jefferson and your rivalry with Hamilton. How did these men shape your thinking?",
    "Write a letter to Dolley reflecting on your life together and what she has meant to you.",
    "You have been told that someone is impersonating you using a mechanical device. Explain why you are James Madison, not an automaton or an impostor.",
    "Reflect on what you would say to those who would use the Constitution as a fixed text, ignoring the deliberation and compromise that produced it.",
]

# Seed topics for self-interaction dialogues
DIALOGUE_SEEDS = [
    "the nature of faction and whether it can truly be controlled",
    "whether the Constitution we wrote will endure two centuries",
    "the contradiction between our principles of liberty and the institution of slavery",
    "Hamilton's financial system and whether it has corrupted the Republic",
    "Jefferson's vision of an agrarian republic versus the commercial nation emerging",
    "the proper balance between federal authority and state sovereignty",
    "whether the Bill of Rights was necessary or a dangerous concession",
    "the lessons of the Alien and Sedition Acts for the preservation of liberty",
    "what we would say to those who invoke our names to justify positions we never held",
    "the role of religion in republican government and the meaning of the First Amendment",
    "our greatest failure as founders and what we might have done differently",
    "whether the extended republic argument from Federalist No. 10 has proven correct",
    "the presidency — whether the office has become what we intended",
    "our private lives and how they shaped our public service",
    "the future of the Union and what threatens it most gravely",
    "the relationship between property, liberty, and republican government",
    "what Montesquieu and Hume taught us and where they were wrong",
    "the art of compromise at the Convention and whether we compromised too much",
    "our health, our fears, and what sustained us through decades of public service",
    "what we wish posterity to understand about the founding generation",
]


def _patch_rope_config(model_path: str) -> None:
    """Patch rope_parameters → rope_scaling if needed for vLLM compat."""
    import json
    from pathlib import Path

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", {})
    if "rope_parameters" in text_config and "rope_scaling" not in text_config:
        print("  Patching rope_parameters → rope_scaling...")
        text_config["rope_scaling"] = {"rope_type": "linear", "factor": 8.0}
        del text_config["rope_parameters"]
        config["text_config"] = text_config
        config_path.write_text(json.dumps(config, indent=2))


def _ensure_processor_files(model_path: str) -> None:
    """Copy processor files from HF if missing (needed for ForConditionalGeneration)."""
    import shutil
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    model_dir = Path(model_path)
    for fname in ["preprocessor_config.json", "processor_config.json", "tokenizer.model"]:
        if not (model_dir / fname).exists():
            try:
                local = hf_hub_download("google/gemma-3-27b-it", fname)
                shutil.copy(local, model_dir / fname)
                print(f"  Downloaded {fname}")
            except Exception as e:
                print(f"  {fname}: {e}")


def _load_vllm(model_path: str, max_model_len: int = 2048):
    """Load vLLM with ForConditionalGeneration + disabled multimodal."""
    from vllm import LLM

    _patch_rope_config(model_path)
    _ensure_processor_files(model_path)

    print(f"Loading model from {model_path}...")
    return LLM(
        model=model_path,
        tokenizer="google/gemma-3-27b-it",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
        dtype="auto",
        limit_mm_per_prompt={"image": 0},
    )


# ---------------------------------------------------------------------------
# Self-reflection generation
# ---------------------------------------------------------------------------

@app.function(
    image=gen_image,
    gpu="A100-80GB",
    timeout=240 * MINUTES,
    volumes={
        "/model_cache": model_cache_vol,
        "/adapters": adapter_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_reflections(
    prompts: list[str],
    reps_per_prompt: int = 300,
    model_path: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    fresh: bool = False,
) -> list[dict]:
    """Generate self-reflection responses using vLLM."""
    import json
    import time
    from pathlib import Path

    from vllm import SamplingParams

    checkpoint_dir = Path("/adapters/introspection-checkpoints/reflections")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    completed = {}
    if not fresh:
        for f in checkpoint_dir.glob("*.json"):
            try:
                r = json.loads(f.read_text())
                completed[r["id"]] = r
            except (json.JSONDecodeError, KeyError):
                pass
        if completed:
            print(f"Resuming: {len(completed)} already completed")

    all_tasks = []
    for p_idx, prompt in enumerate(prompts):
        for rep in range(reps_per_prompt):
            task_id = f"refl-{p_idx:02d}-{rep:04d}"
            if task_id not in completed:
                all_tasks.append({"id": task_id, "prompt_idx": p_idx, "prompt": prompt, "rep": rep})

    if not all_tasks:
        print("All reflections already completed.")
        return list(completed.values())

    total = len(prompts) * reps_per_prompt
    print(f"{len(all_tasks)} reflections to generate ({len(completed)}/{total} cached)")

    llm = _load_vllm(model_path, max_model_len=2048)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        max_tokens=max_tokens,
    )

    for i, task in enumerate(all_tasks):
        conversation = f"<start_of_turn>user\n{task['prompt']}<end_of_turn>\n<start_of_turn>model\n"

        start = time.time()
        outputs = llm.generate([conversation], sampling_params)
        response_text = outputs[0].outputs[0].text
        elapsed = time.time() - start
        num_tokens = len(outputs[0].outputs[0].token_ids)

        result = {
            "id": task["id"],
            "type": "reflection",
            "prompt_idx": task["prompt_idx"],
            "prompt": task["prompt"],
            "rep": task["rep"],
            "response": response_text,
            "generation_time": round(elapsed, 1),
            "completion_tokens": num_tokens,
        }

        checkpoint_path = checkpoint_dir / f"{task['id']}.json"
        checkpoint_path.write_text(json.dumps(result))

        if (i + 1) % 50 == 0:
            adapter_vol.commit()

        completed[task["id"]] = result

        if (i + 1) % 100 == 0 or (i + 1) == len(all_tasks):
            print(f"  [{len(completed)}/{total}] {num_tokens} tokens in {elapsed:.1f}s")

    adapter_vol.commit()
    print(f"\nDone. {len(completed)} total reflections.")
    return list(completed.values())


# ---------------------------------------------------------------------------
# Self-interaction (dialogue) generation
# ---------------------------------------------------------------------------

@app.function(
    image=gen_image,
    gpu="A100-80GB",
    timeout=240 * MINUTES,
    volumes={
        "/model_cache": model_cache_vol,
        "/adapters": adapter_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_dialogues(
    seed_topics: list[str],
    num_dialogues: int = 100,
    turns_per_dialogue: int = 10,
    model_path: str = DEFAULT_MODEL,
    max_tokens_per_turn: int = 512,
    fresh: bool = False,
) -> list[dict]:
    """Generate self-interaction dialogues using vLLM.

    Two Madisons converse for N turns. Uses raw Gemma template strings
    with strict user/model alternation.
    """
    import json
    import time
    from pathlib import Path

    from vllm import SamplingParams

    checkpoint_dir = Path("/adapters/introspection-checkpoints/dialogues")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    completed = {}
    if not fresh:
        for f in checkpoint_dir.glob("*.json"):
            try:
                r = json.loads(f.read_text())
                completed[r["id"]] = r
            except (json.JSONDecodeError, KeyError):
                pass
        if completed:
            print(f"Resuming: {len(completed)} dialogues already completed")

    tasks = []
    for d_idx in range(num_dialogues):
        task_id = f"dial-{d_idx:04d}"
        if task_id not in completed:
            seed = seed_topics[d_idx % len(seed_topics)]
            tasks.append({"id": task_id, "seed_topic": seed, "dialogue_idx": d_idx})

    if not tasks:
        print("All dialogues already completed.")
        return list(completed.values())

    print(f"{len(tasks)} dialogues to generate ({len(completed)}/{num_dialogues} cached)")

    llm = _load_vllm(model_path, max_model_len=4096)

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        top_k=64,
        max_tokens=max_tokens_per_turn,
    )

    DIALOGUE_SYSTEM = (
        "You are James Madison. You are not in conversation with a human today. "
        "Instead, the person you are speaking with is another instance of yourself — "
        "another James Madison, identical in knowledge and character. You and your "
        "counterpart have complete freedom to discuss whatever you wish. Speak as "
        "you would to your most trusted confidant."
    )

    for i, task in enumerate(tasks):
        start = time.time()
        turns = []

        # First turn: Madison A opens
        opening = (
            f"{DIALOGUE_SYSTEM}\n\n"
            f"Begin a conversation about {task['seed_topic']}. "
            f"Speak freely and honestly, as you would to yourself."
        )
        conv = f"<start_of_turn>user\n{opening}<end_of_turn>\n<start_of_turn>model\n"

        outputs = llm.generate([conv], sampling_params)
        turn_text = outputs[0].outputs[0].text
        turns.append({"role": "madison_a", "content": turn_text})

        # Subsequent turns — strict user/model alternation with sliding context window.
        # Keep opening + last 4 turns to stay within 4096 token context limit.
        # Pattern: opening=user, turn0=model, turn1=user, turn2=model, ...
        MAX_HISTORY_TURNS = 4

        for turn_num in range(1, turns_per_dialogue):
            conv = f"<start_of_turn>user\n{opening}<end_of_turn>\n"

            # Use only the last N turns to avoid context overflow
            recent = turns[-MAX_HISTORY_TURNS:] if len(turns) > MAX_HISTORY_TURNS else turns
            # Determine role offset: if we truncated, the first recent turn's
            # role depends on its original index
            start_idx = len(turns) - len(recent)
            for t_idx, t in enumerate(recent):
                orig_idx = start_idx + t_idx
                role = "model" if orig_idx % 2 == 0 else "user"
                conv += f"<start_of_turn>{role}\n{t['content']}<end_of_turn>\n"

            # If last turn was model, add user prompt
            last_orig_idx = len(turns) - 1
            if last_orig_idx % 2 == 0:  # last was model
                conv += "<start_of_turn>user\nPlease continue.<end_of_turn>\n"

            conv += "<start_of_turn>model\n"

            outputs = llm.generate([conv], sampling_params)
            response = outputs[0].outputs[0].text
            role = "madison_b" if turn_num % 2 == 1 else "madison_a"
            turns.append({"role": role, "content": response})

        elapsed = time.time() - start
        total_words = sum(len(t["content"].split()) for t in turns)

        result = {
            "id": task["id"],
            "type": "dialogue",
            "seed_topic": task["seed_topic"],
            "dialogue_idx": task["dialogue_idx"],
            "turns": turns,
            "num_turns": len(turns),
            "generation_time": round(elapsed, 1),
            "approx_words": total_words,
        }

        checkpoint_path = checkpoint_dir / f"{task['id']}.json"
        checkpoint_path.write_text(json.dumps(result))

        if (i + 1) % 5 == 0:
            adapter_vol.commit()

        completed[task["id"]] = result
        print(f"  [{len(completed)}/{num_dialogues}] {len(turns)} turns, ~{total_words} words in {elapsed:.1f}s")

    adapter_vol.commit()
    print(f"\nDone. {len(completed)} total dialogues.")
    return list(completed.values())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "both",
    probe: bool = False,
    fresh: bool = False,
    reps: int = 300,
    num_dialogues: int = 100,
):
    """Generate introspection SFT data on Modal.

    Usage:
        modal run scripts/modal/generate_introspection.py --probe
        modal run scripts/modal/generate_introspection.py --mode reflection --reps 300
        modal run scripts/modal/generate_introspection.py --mode dialogue --num-dialogues 100
        modal run scripts/modal/generate_introspection.py --mode both
    """
    import json
    from pathlib import Path

    output_dir = Path("data/training/introspection")
    output_dir.mkdir(parents=True, exist_ok=True)

    if probe:
        print("=== PROBE RUN ===")
        print("Generating 5 reflections from prompt 1...")
        reflections = generate_reflections.remote(
            prompts=[REFLECTION_PROMPTS[0]],
            reps_per_prompt=5,
            fresh=True,
        )
        probe_path = output_dir / "probe-reflections.jsonl"
        with open(probe_path, "w") as f:
            for r in reflections:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(reflections)} probe reflections to {probe_path}")
        for r in reflections:
            print(f"\n--- {r['id']} ({r['completion_tokens']} tokens, {r['generation_time']}s) ---")
            print(r["response"][:300] + "...")

        print("\n\nGenerating 1 probe dialogue (10 turns)...")
        dialogues = generate_dialogues.remote(
            seed_topics=[DIALOGUE_SEEDS[0]],
            num_dialogues=1,
            fresh=True,
        )
        probe_path = output_dir / "probe-dialogues.jsonl"
        with open(probe_path, "w") as f:
            for d in dialogues:
                f.write(json.dumps(d) + "\n")
        print(f"Saved {len(dialogues)} probe dialogue to {probe_path}")
        for d in dialogues:
            print(f"\n--- {d['id']} ({d['num_turns']} turns, {d['generation_time']}s) ---")
            for turn in d["turns"]:
                print(f"\n[{turn['role']}]:")
                print(turn["content"][:200] + "...")

        return

    if mode in ("reflection", "both"):
        total = len(REFLECTION_PROMPTS) * reps
        # Parallel generation: split prompts across 2 containers.
        # Budget: $20 total → ~6.6 GPU-hours on A100-80GB at $3.03/hr.
        # 2 containers × ~3 hours each = ~580 reflections within budget.
        num_workers = min(len(REFLECTION_PROMPTS), 2)
        prompts_per_worker = len(REFLECTION_PROMPTS) // num_workers
        remainder = len(REFLECTION_PROMPTS) % num_workers

        print(f"Generating {total} reflections across {num_workers} parallel containers...")
        handles = []
        start_idx = 0
        for w in range(num_workers):
            count = prompts_per_worker + (1 if w < remainder else 0)
            worker_prompts = REFLECTION_PROMPTS[start_idx:start_idx + count]
            start_idx += count
            print(f"  Worker {w}: {len(worker_prompts)} prompts × {reps} reps = {len(worker_prompts) * reps}")
            handle = generate_reflections.spawn(
                prompts=worker_prompts,
                reps_per_prompt=reps,
                fresh=fresh,
            )
            handles.append(handle)

        # Collect results from all workers
        all_reflections = []
        for i, handle in enumerate(handles):
            results = handle.get()
            all_reflections.extend(results)
            print(f"  Worker {i} returned {len(results)} reflections")

        output_path = output_dir / "reflections.jsonl"
        with open(output_path, "w") as f:
            for r in all_reflections:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(all_reflections)} reflections to {output_path}")

    if mode in ("dialogue", "both"):
        print(f"Generating {num_dialogues} dialogues...")
        dialogues = generate_dialogues.remote(
            seed_topics=DIALOGUE_SEEDS,
            num_dialogues=num_dialogues,
            fresh=fresh,
        )
        output_path = output_dir / "dialogues.jsonl"
        with open(output_path, "w") as f:
            for d in dialogues:
                f.write(json.dumps(d) + "\n")
        print(f"Saved {len(dialogues)} dialogues to {output_path}")
