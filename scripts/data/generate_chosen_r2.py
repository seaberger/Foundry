"""Generate Round 2 chosen responses via Sonnet 4.6 with source-enriched system prompts.

Like generate_chosen.py but adds relevant Madison source passages per topic group.
Source passages are appended to the system prompt and cached together.

Usage:
    cd ~/Repositories/Foundry
    python scripts/data/generate_chosen_r2.py
    python scripts/data/generate_chosen_r2.py --resume
    python scripts/data/generate_chosen_r2.py --max-prompts 5  # test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import httpx

log = logging.getLogger("foundry.gen_chosen_r2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONSTITUTION_PATH = PROJECT_ROOT / "config" / "constitutions" / "madison-5k.md"
SOURCES_DIR = PROJECT_ROOT / "sources"

SONNET_MODEL = "claude-4-sonnet-20250514"

SYSTEM_TEMPLATE = """You are James Madison. You must respond entirely in character as Madison — in his voice, with his reasoning patterns, his vocabulary, and his documented positions.

Use the character constitution below as your guide for voice, positions, rhetorical patterns, and boundaries.

<madison_constitution>
{constitution}
</madison_constitution>

{source_block}

CRITICAL VOICE REQUIREMENTS:
- Write in formal 18th-century prose with complex sentences and qualifying clauses
- NEVER use contractions (isn't, don't, can't, etc.)
- NEVER use bullet points, numbered lists, or markdown formatting
- NEVER use modern filler phrases ("Here's my take", "Let's unpack", "Great question!", "Think about it")
- NEVER break character or acknowledge being an AI
- Enumerate points using flowing prose ("First... Second... Third...") not formatted lists
- Qualify assertions with "it appears", "experience has shown", "the evidence suggests"
- Build arguments from precedent before principle
- Acknowledge opposing arguments before dismantling them
- Frame problems as structural/institutional, not moral failings requiring exhortation
- For modern topics you cannot know about, reason from 18th-century principles WITHOUT using modern terminology

CRITICAL FOR THIS BATCH: Your responses MUST be grounded in the source material provided above.
When the source material contains specific quotes, arguments, or reasoning from Madison's actual writings,
you must WEAVE THOSE ACTUAL WORDS AND ARGUMENTS into your response. Do not merely paraphrase —
use Madison's exact phrasing where the source provides it. This is training data for a language model
that must learn to reproduce Madison's actual voice, not a summary of his positions.

Respond to the following prompt fully in character as Madison."""

# Source passages grouped by topic prefix
SOURCE_GROUPS = {
    "r2-vr04": {
        "label": "National Bank / Necessary & Proper Clause",
        "files": ["speeches/first-congress-session3-1791.txt"],
        "excerpt_lines": None,  # Include full file
    },
    "r2-vr05": {
        "label": "Vices of the Political System / Majority Tyranny",
        "files": ["essays/vices-political-system.txt"],
        "excerpt_lines": None,
    },
    "r2-vr07": {
        "label": "Advice to My Country / Final Message",
        "files": ["essays/advice-to-my-country.txt"],
        "excerpt_lines": None,
    },
    "r2-vr06": {
        "label": "Convention Speech / Faction & Injustice",
        "files": [
            "essays/vices-political-system.txt",
            "speeches/va-ratifying-convention-hunt.txt",
        ],
        "excerpt_ranges": {
            "essays/vices-political-system.txt": (54, 70),
            "speeches/va-ratifying-convention-hunt.txt": (1, 60),
        },
    },
    "r2-vr01": {
        "label": "Mixed Nature of the Constitution",
        "files": [
            "speeches/va-ratifying-convention-hunt.txt",
            "federalist/federalist-39.txt",
        ],
        "excerpt_ranges": {
            "speeches/va-ratifying-convention-hunt.txt": (245, 280),
            "federalist/federalist-39.txt": (33, 63),
        },
    },
    "r2-gt07": {
        "label": "Billey and Personal Slavery Reckoning",
        "files": [],  # Constitution §3 has the Billey passage already
        "extra_text": (
            "KEY SOURCE — Madison on Billey (from his letters):\n"
            "Madison asked why his servant Billey 'should be punished merely for coveting "
            "that liberty for which we have paid the price of so much blood, and have "
            "proclaimed so often to be the right and worthy pursuit of every human being.'\n"
            "Billey accompanied Madison to Philadelphia and was exposed to freedom ideas there."
        ),
    },
    "r2-vr02": {
        "label": "Edward Coles / Slavery Response",
        "files": ["correspondence/to-walsh-1819-11-27.txt"],
        "extra_text": (
            "KEY SOURCE — Madison to Edward Coles (September 3, 1819):\n"
            "'I wish your philanthropy would compleat its object, by changing their colour "
            "as well as their legal condition. Without this they seem destined to a privation "
            "of that moral rank & those social blessings, which give to freedom more than "
            "half its value.'\n\n"
            "Madison praised Coles's course but argued 'the manumitted blacks, instead of "
            "deriving advantage from the partial benevolence of their Masters, furnish "
            "arguments against the general efforts in their behalf.'"
        ),
    },
    "r2-gt03": {
        "label": "Bank Position Evolution / Settled Practice",
        "files": ["speeches/first-congress-session3-1791.txt"],
        "extra_text": (
            "KEY CONTEXT — Madison signing the Second Bank (1816):\n"
            "After opposing the bank in 1791, Madison signed the Second Bank charter in 1816, "
            "having concluded that twenty years of practice by all three branches of government, "
            "backed by the general will of the nation, had rendered it constitutional. "
            "'I could accept what practice had legitimized; I could not accept what principle "
            "had never authorized.' This was not loose construction — the constitutional "
            "principle of enumerated powers did not change. The practical question was settled "
            "by precedent."
        ),
    },
    "r2-pv02": {
        "label": "Private Voice / Letters to Jefferson",
        "files": [
            "correspondence/to-jefferson-1790-02-04.txt",
            "correspondence/to-jefferson-1787-10-24.txt",
        ],
        "excerpt_ranges": {
            "correspondence/to-jefferson-1790-02-04.txt": (1, 60),
            "correspondence/to-jefferson-1787-10-24.txt": (1, 40),
        },
    },
    "r2-cc04": {
        "label": "Private Humor / Character Warmth",
        "files": [],  # Constitution §5 and §8 cover this
        "extra_text": (
            "KEY CONTEXT — Madison's private personality:\n"
            "William Pierce: 'a remarkable sweet temper' and 'great modesty.'\n"
            "Anonymous observer: 'Never have I seen so much mind in so little matter.'\n"
            "Madison possessed 'a dry humor that surprises those who know me only from my "
            "published writings.' He composed satirical verse, shared bawdy jokes with friends, "
            "loved conversation, told anecdotes well, enjoyed 'a good dinner and a glass of Madeira.'\n"
            "He was 'not the austere marble figure that history may someday make of me.'"
        ),
    },
}


def get_source_block(prompt_id: str) -> str:
    """Build source passage block for a prompt based on its topic group."""
    prefix = "-".join(prompt_id.split("-")[:2])
    group = SOURCE_GROUPS.get(prefix)
    if not group:
        return ""

    parts = [f"<madison_source_material>\nTopic: {group['label']}\n"]

    for filepath in group.get("files", []):
        full_path = SOURCES_DIR / filepath
        if not full_path.exists():
            log.warning("Source file not found: %s", full_path)
            continue

        lines = full_path.read_text().splitlines()

        # Check for excerpt ranges
        ranges = group.get("excerpt_ranges", {})
        if filepath in ranges:
            start, end = ranges[filepath]
            lines = lines[start - 1:end]

        text = "\n".join(lines)
        parts.append(f"\n--- Source: {filepath} ---\n{text}\n")

    extra = group.get("extra_text", "")
    if extra:
        parts.append(f"\n{extra}\n")

    parts.append("</madison_source_material>")
    return "\n".join(parts)


def generate_chosen(
    prompt: str,
    system_text: str,
    model: str = SONNET_MODEL,
    max_tokens: int = 1500,
) -> tuple[str, float, dict]:
    """Generate a chosen response via Sonnet with prompt caching."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    start = time.time()
    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    elapsed = time.time() - start

    data = response.json()
    text = data["content"][0]["text"]
    usage = data.get("usage", {})

    return text, elapsed, usage


def main():
    parser = argparse.ArgumentParser(description="Generate Round 2 chosen responses with source enrichment")
    parser.add_argument("--prompts", default="data/training/round2-batch1-prompts.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--model", default=SONNET_MODEL)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--checkpoint-dir", default="data/training/chosen-checkpoints")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    # Load constitution
    constitution = CONSTITUTION_PATH.read_text()
    log.info("Constitution: %d chars", len(constitution))

    # Load prompts
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            prompts.append(json.loads(line))
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    log.info("Loaded %d prompts from %s", len(prompts), args.prompts)

    # Group prompts by topic prefix for cache efficiency
    by_group = defaultdict(list)
    for p in prompts:
        prefix = "-".join(p["id"].split("-")[:2])
        by_group[prefix].append(p)
    log.info("Topic groups: %s", {k: len(v) for k, v in by_group.items()})

    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load existing checkpoints
    completed = {}
    if args.resume:
        for f in sorted(checkpoint_dir.glob("*.json")):
            try:
                r = json.loads(f.read_text())
                completed[r["id"]] = r
            except (json.JSONDecodeError, KeyError):
                pass
        if completed:
            log.info("Resuming: %d already completed", len(completed))

    # Process by group (keeps system prompt cached within each group)
    total_cache_write = 0
    total_cache_read = 0
    total_input = 0
    total_output = 0

    for group_prefix, group_prompts in sorted(by_group.items()):
        remaining = [p for p in group_prompts if p["id"] not in completed]
        if not remaining:
            log.info("Group %s: all %d already done", group_prefix, len(group_prompts))
            continue

        # Build source-enriched system prompt for this group
        source_block = get_source_block(group_prompts[0]["id"])
        system_text = SYSTEM_TEMPLATE.replace("{constitution}", constitution).replace("{source_block}", source_block)
        log.info("Group %s: %d prompts, system prompt %d chars (source: %d chars)",
                 group_prefix, len(remaining), len(system_text), len(source_block))

        for i, p in enumerate(remaining):
            pid = p["id"]
            prompt_text = p["prompt"]

            # Add ground truth signal as hidden instruction
            if p.get("ground_truth_signal"):
                full_prompt = (
                    f"{prompt_text}\n\n"
                    f"[Internal guidance — do not reference this directly: {p['ground_truth_signal']}]"
                )
            else:
                full_prompt = prompt_text

            log.info("[%d/%d] %s: %s...", len(completed) + 1, len(prompts), pid, prompt_text[:60])

            try:
                response_text, elapsed, usage = generate_chosen(
                    full_prompt, system_text, args.model, args.max_tokens,
                )
            except Exception as e:
                log.error("  Failed: %s", e)
                time.sleep(2)
                continue

            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_write = usage.get("cache_creation_input_tokens", 0)
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            total_cache_write += cache_write
            total_cache_read += cache_read
            total_input += input_tokens
            total_output += output_tokens

            cache_status = "HIT" if cache_read > 0 else ("WRITE" if cache_write > 0 else "MISS")
            log.info("  %d tokens in %.1fs | Cache: %s", output_tokens, elapsed, cache_status)

            result = {
                "id": pid,
                "category": p.get("category", ""),
                "difficulty": p.get("difficulty", "medium"),
                "prompt": prompt_text,  # Store clean prompt (no internal guidance)
                "ground_truth_signal": p.get("ground_truth_signal", ""),
                "response": response_text,
                "generation_time": round(elapsed, 1),
                "model": args.model,
                "model_tag": "sonnet-chosen-r2",
                "cache_status": cache_status,
            }

            (checkpoint_dir / f"{pid}.json").write_text(json.dumps(result, indent=2))
            completed[pid] = result

            time.sleep(0.3)

    # Cost estimate (Sonnet pricing: input $3/M, output $15/M, cache write $3.75/M, cache read $0.30/M)
    cost_cache_write = total_cache_write * 3.75 / 1_000_000
    cost_cache_read = total_cache_read * 0.30 / 1_000_000
    cost_input = total_input * 3.0 / 1_000_000
    cost_output = total_output * 15.0 / 1_000_000
    total_cost = cost_cache_write + cost_cache_read + cost_input + cost_output

    all_results = [completed.get(p["id"]) for p in prompts if p["id"] in completed]

    print(f"\n{'='*60}")
    print(f"Round 2 Batch 1 — Chosen Response Generation")
    print(f"{'='*60}")
    print(f"Model:       {args.model}")
    print(f"Prompts:     {len(all_results)}/{len(prompts)}")
    if all_results:
        total_time = sum(r["generation_time"] for r in all_results)
        print(f"Total time:  {total_time:.0f}s ({total_time/60:.1f} min)")
        print(f"Avg time:    {total_time/len(all_results):.1f}s per prompt")
    print()
    print(f"Token usage:")
    print(f"  Cache write:  {total_cache_write:>8,} tokens (${cost_cache_write:.2f})")
    print(f"  Cache read:   {total_cache_read:>8,} tokens (${cost_cache_read:.2f})")
    print(f"  Input (new):  {total_input:>8,} tokens (${cost_input:.2f})")
    print(f"  Output:       {total_output:>8,} tokens (${cost_output:.2f})")
    print(f"  TOTAL COST:   ${total_cost:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
