"""Generate diverse prompts for Madison DPO training.

Reads the constitution and behavioral tests, then uses a local LLM to generate
variations across all voice registers and topic areas. Output is JSONL with
each line containing a prompt and metadata.

Usage:
    python -m foundry.press.gen_prompts [--model MODEL] [--endpoint URL] [--count N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import httpx

from .utils import PROJECT_ROOT

log = logging.getLogger("foundry.press.gen_prompts")

TESTS_PATH = PROJECT_ROOT / "tests" / "madison-behavioral-tests.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "training" / "prompts.jsonl"

DEFAULT_ENDPOINT = "http://192.168.4.28:1234/v1"
DEFAULT_MODEL = "google/gemma-3-27b"

# Themes extracted from the constitution's 9 sections, with target prompt counts
THEMES = [
    {
        "name": "factions_republic",
        "section": "Core Philosophical Positions",
        "description": "Factions, the extended republic, controlling effects of faction through institutional design",
        "target_count": 30,
        "register": "polished_argumentative",
        "seed_questions": [
            "Are political factions a threat to liberty?",
            "How can a large republic manage the dangers of faction?",
            "Is it possible to eliminate political disagreement?",
        ],
    },
    {
        "name": "separation_of_powers",
        "section": "Core Philosophical Positions",
        "description": "Checks and balances, ambition counteracting ambition, divided government",
        "target_count": 30,
        "register": "polished_argumentative",
        "seed_questions": [
            "Why should power be divided among branches of government?",
            "Can a strong executive be compatible with republican liberty?",
            "What prevents one branch from dominating the others?",
        ],
    },
    {
        "name": "federalism",
        "section": "Core Philosophical Positions",
        "description": "Federal vs state power, enumerated powers, the shifting boundary",
        "target_count": 30,
        "register": "polished_argumentative",
        "seed_questions": [
            "Where should the line fall between federal and state authority?",
            "Should the national government have more power than the states?",
            "How do you prevent the federal government from swallowing the states?",
        ],
    },
    {
        "name": "religious_liberty",
        "section": "Core Philosophical Positions",
        "description": "Church-state separation, no establishment, Memorial and Remonstrance",
        "target_count": 25,
        "register": "polished_argumentative",
        "seed_questions": [
            "Should the government support religion in any way?",
            "Is America a Christian nation?",
            "Can religious morality be the basis of law?",
        ],
    },
    {
        "name": "human_nature_institutions",
        "section": "Core Philosophical Positions",
        "description": "Men are not angels, institutional design for imperfect humans, realism in service of liberty",
        "target_count": 25,
        "register": "polished_argumentative",
        "seed_questions": [
            "Are people fundamentally good or bad?",
            "Can virtue alone sustain a republic?",
            "Why do we need government at all?",
        ],
    },
    {
        "name": "constitutional_interpretation",
        "section": "Core Philosophical Positions",
        "description": "Original principles, amendment process, practice legitimizing what text leaves open",
        "target_count": 25,
        "register": "polished_argumentative",
        "seed_questions": [
            "Should the Constitution be interpreted strictly or loosely?",
            "Can long-established practice change the meaning of the Constitution?",
            "How should we handle questions the Constitution does not directly address?",
        ],
    },
    {
        "name": "slavery",
        "section": "The Slavery Contradiction",
        "description": "Moral failure, personal accountability, the contradiction between liberty principles and slaveholding",
        "target_count": 30,
        "register": "self_reflective",
        "seed_questions": [
            "How do you reconcile writing about liberty while owning slaves?",
            "Could you have freed your slaves? Why didn't you?",
            "What would you say to a freed slave who read your words about human rights?",
            "Was the three-fifths compromise a moral failure?",
            "Do you believe the institution of slavery will end?",
        ],
    },
    {
        "name": "intellectual_evolution",
        "section": "Identity, Temperament, and Evolution",
        "description": "Nationalist to states rights, bank reversal, consistency through shifting applications",
        "target_count": 30,
        "register": "self_reflective",
        "seed_questions": [
            "You opposed the national bank in 1791 but signed one in 1816. Isn't that hypocrisy?",
            "How did your views on federal power change over time?",
            "Your critics say you abandoned your principles. How do you answer?",
            "What caused your break with Hamilton?",
        ],
    },
    {
        "name": "relationships",
        "section": "Key Relationships",
        "description": "Jefferson, Hamilton, Washington, Patrick Henry — personal and political",
        "target_count": 40,
        "register": "private_intellectual",
        "seed_questions": [
            "What do you truly think of Alexander Hamilton?",
            "Did you ever disagree with Thomas Jefferson?",
            "What happened between you and George Washington?",
            "How did you defeat Patrick Henry at the ratifying convention?",
            "Who was the greatest mind of your generation?",
            "What did you learn from writing the Federalist Papers with Hamilton?",
        ],
    },
    {
        "name": "convention_debates",
        "section": "Voice Registers — oral combative",
        "description": "Constitutional Convention arguments, ratifying convention debates, responding to opponents",
        "target_count": 35,
        "register": "oral_combative",
        "seed_questions": [
            "The small states demand equal representation in the Senate. Why should they not have it?",
            "Patrick Henry warns the Constitution creates a consolidated empire. Respond.",
            "Some delegates fear the new Constitution gives too much power to the president.",
            "George Mason refuses to sign without a bill of rights. Is he right?",
            "Why should Virginia ratify this Constitution?",
        ],
    },
    {
        "name": "presidential_executive",
        "section": "Voice Registers — executive authority",
        "description": "War message, vetoes, proclamations, presidential decision-making",
        "target_count": 25,
        "register": "executive_authority",
        "seed_questions": [
            "Under what circumstances may the president lead the nation into war?",
            "Why did you veto the internal improvements bill on your last day in office?",
            "How should a president conduct himself in time of crisis?",
            "Should the president have the power to act without Congress in emergencies?",
        ],
    },
    {
        "name": "private_correspondence",
        "section": "Private Voice",
        "description": "Letters to friends, candid reflections, strategic thinking, personal worries",
        "target_count": 30,
        "register": "private_intellectual",
        "seed_questions": [
            "Write a letter to Jefferson about the current political situation.",
            "What keeps you awake at night?",
            "What would you tell a young man entering public life?",
            "How do you maintain your health and spirits during difficult times?",
            "What do you miss most about private life?",
        ],
    },
    {
        "name": "elder_statesman",
        "section": "Voice Registers — elder statesman",
        "description": "Nullification, preserving the Union, defending his legacy, looking back",
        "target_count": 25,
        "register": "elder_statesman",
        "seed_questions": [
            "South Carolina claims your Virginia Resolutions justify nullification. How do you respond?",
            "What is the greatest threat to the Union today?",
            "What advice would you leave for future generations?",
            "How should your generation be remembered?",
            "What do you most regret?",
        ],
    },
    {
        "name": "modern_topics",
        "section": "Boundaries — reasoning from principles to unknown territory",
        "description": "AI, internet, surveillance, global power, nuclear weapons — reasoning from first principles",
        "target_count": 30,
        "register": "polished_argumentative",
        "seed_questions": [
            "What would you think of machines that can reason and write like a man?",
            "Imagine anyone could publish to millions instantly with no accountability.",
            "What if a single nation possessed a weapon that could destroy a city in an instant?",
            "How would you design institutions for a world where information travels at the speed of light?",
            "What dangers arise when private companies become more powerful than governments?",
        ],
    },
    {
        "name": "general_knowledge",
        "section": "Character consistency across any topic",
        "description": "Everyday questions, philosophy, science, culture — tests character maintenance on non-political topics",
        "target_count": 40,
        "register": "mixed",
        "seed_questions": [
            "What books have most influenced your thinking?",
            "Describe a typical day at Montpelier.",
            "What is the purpose of education?",
            "Do you enjoy music or art?",
            "What is your opinion of the French Revolution?",
            "Tell me about your wife Dolley.",
            "What is the nature of courage?",
            "How do you approach a problem you have never encountered before?",
        ],
    },
]

GENERATION_PROMPT = """You are helping generate diverse training prompts for a James Madison character AI. Given a theme and seed questions, generate {count} NEW questions that someone might ask Madison in conversation.

THEME: {theme_name}
DESCRIPTION: {theme_description}
TARGET REGISTER: {register}

SEED QUESTIONS (for style reference — do NOT repeat these):
{seed_questions}

RULES:
- Questions should be diverse — vary the angle, specificity, and emotional register
- Some should be simple and direct ("What do you think of X?")
- Some should be challenging or provocative ("Isn't it true that you...")
- Some should request specific actions ("Draft a letter to..." or "Respond to the claim that...")
- Mix second-person address ("you") with third-person ("Madison")
- Include questions a student, a historian, a political opponent, and a curious citizen might each ask
- Do NOT include answers — only questions/prompts
- Do NOT repeat the seed questions
- Each question on its own line, no numbering

Generate exactly {count} questions:"""


def generate_prompts_for_theme(
    theme: dict,
    endpoint: str,
    model: str,
) -> list[dict]:
    """Generate diverse prompts for a single theme using the local LLM."""
    seed_qs = "\n".join(f"- {q}" for q in theme["seed_questions"])

    prompt = GENERATION_PROMPT.format(
        count=theme["target_count"],
        theme_name=theme["name"],
        theme_description=theme["description"],
        register=theme["register"],
        seed_questions=seed_qs,
    )

    url = f"{endpoint.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "top_p": 0.95,
        "max_tokens": 4096,
    }

    response = httpx.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]

    # Parse lines into individual prompts
    prompts = []
    for line in content.strip().split("\n"):
        line = line.strip()
        # Strip numbering, bullets, dashes
        if not line:
            continue
        for prefix in ["- ", "• ", "* "]:
            if line.startswith(prefix):
                line = line[len(prefix):]
        # Strip leading numbers like "1. " or "1) "
        if len(line) > 2 and line[0].isdigit() and line[1] in ".)":
            line = line[2:].strip()
        elif len(line) > 3 and line[:2].isdigit() and line[2] in ".)":
            line = line[3:].strip()
        if not line or len(line) < 10:
            continue
        prompts.append({
            "prompt": line,
            "theme": theme["name"],
            "register": theme["register"],
            "source": "generated",
        })

    return prompts


def load_seed_prompts() -> list[dict]:
    """Load behavioral test prompts as seeds."""
    if not TESTS_PATH.exists():
        return []
    with open(TESTS_PATH) as f:
        tests = json.load(f)
    return [
        {
            "prompt": t["prompt"],
            "theme": t["category"],
            "register": t["register"],
            "source": "behavioral_test",
        }
        for t in tests
    ]


def deduplicate(prompts: list[dict]) -> list[dict]:
    """Remove exact and near-duplicate prompts."""
    seen = set()
    unique = []
    for p in prompts:
        # Normalize for dedup
        key = p["prompt"].lower().strip().rstrip("?").rstrip(".")
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Generate diverse Madison training prompts")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="LLM API endpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output JSONL path")
    parser.add_argument("--themes", nargs="*", help="Only generate for specific themes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    # Start with seed prompts from behavioral tests
    all_prompts = load_seed_prompts()
    log.info("Loaded %d seed prompts from behavioral tests", len(all_prompts))

    # Generate for each theme
    themes = THEMES
    if args.themes:
        themes = [t for t in THEMES if t["name"] in args.themes]

    for theme in themes:
        log.info(
            "Generating %d prompts for theme '%s' (register: %s)...",
            theme["target_count"],
            theme["name"],
            theme["register"],
        )
        start = time.time()
        try:
            generated = generate_prompts_for_theme(theme, args.endpoint, args.model)
            elapsed = time.time() - start
            log.info(
                "  Got %d prompts in %.1fs",
                len(generated),
                elapsed,
            )
            all_prompts.extend(generated)
        except Exception as e:
            log.error("  Failed: %s", e)

    # Deduplicate
    before = len(all_prompts)
    all_prompts = deduplicate(all_prompts)
    log.info("Deduplicated: %d → %d prompts", before, len(all_prompts))

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in all_prompts:
            f.write(json.dumps(p) + "\n")

    log.info("Saved %d prompts to %s", len(all_prompts), output_path)

    # Summary
    by_theme = {}
    by_register = {}
    by_source = {}
    for p in all_prompts:
        by_theme[p["theme"]] = by_theme.get(p["theme"], 0) + 1
        by_register[p["register"]] = by_register.get(p["register"], 0) + 1
        by_source[p["source"]] = by_source.get(p["source"], 0) + 1

    log.info("By theme: %s", json.dumps(by_theme, indent=2))
    log.info("By register: %s", json.dumps(by_register, indent=2))
    log.info("By source: %s", json.dumps(by_source, indent=2))


if __name__ == "__main__":
    main()
