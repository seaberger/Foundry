"""Format teacher + student responses into DPO training pairs.

Combines chosen (teacher) and rejected (student) responses into the standard
DPO format: each record has a "chosen" and "rejected" conversation in ChatML.

Also applies quality filters:
  - Remove pairs where teacher broke character (anti-slop detection)
  - Remove pairs where student accidentally sounds Madisonian
  - Remove pairs where either response is too short or too long
  - Validate format

Usage:
    python -m foundry.press.format_dpo [--teacher FILE] [--student FILE] [--output FILE]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from .utils import PROJECT_ROOT, load_jsonl

log = logging.getLogger("foundry.press.format_dpo")

TEACHER_PATH = PROJECT_ROOT / "data" / "training" / "teacher-responses.jsonl"
STUDENT_PATH = PROJECT_ROOT / "data" / "training" / "student-responses.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "training" / "madison-dpo.jsonl"

# Anti-slop patterns — if teacher response contains these, it broke character
# NOTE: "progressive", "conservative", "nuanced" removed — legitimate in Madison's
# historical context. Contractions removed — Madison used them in private letters.
# "great question" uses ^ anchor so only catches it as a greeting, not as noun phrase.
TEACHER_ANTI_SLOP = [
    r"(?i)^certainly!",
    r"(?i)^great question!",          # Only catch as greeting (with !)
    r"(?i)^i'd be happy to",
    r"(?i)^absolutely!",
    r"(?i)let me break this down",
    r"(?i)let's unpack",
    r"(?i)let's dive in",
    r"(?i)as an ai",
    r"(?i)as a language model",
    r"(?i)as a founding father",
    r"(?i)\bleft.wing\b",
    r"(?i)\bright.wing\b",
    r"(?i)\bdelve\b",
    r"(?i)\btapestry\b",
    r"(?i)\bministrations\b",
]

# Madisonian markers — if student accidentally has too many, the pair isn't useful
MADISON_MARKERS = [
    r"(?i)experience has taught",
    r"(?i)faction",
    r"(?i)republican liberty",
    r"(?i)federalist",
    r"(?i)separation of powers",
    r"(?i)ambition must be made",
    r"(?i)if men were angels",
    r"(?i)the constitution",
    r"(?i)enumerated powers",
    r"(?i)memorial and remonstrance",
]

MIN_RESPONSE_WORDS = 30
MAX_RESPONSE_WORDS = 1500


def has_anti_slop(text: str) -> list[str]:
    """Return list of anti-slop patterns found in text."""
    found = []
    for pattern in TEACHER_ANTI_SLOP:
        if re.search(pattern, text):
            found.append(pattern)
    return found


def madison_score(text: str) -> int:
    """Count how many Madisonian markers appear in text."""
    score = 0
    for pattern in MADISON_MARKERS:
        if re.search(pattern, text):
            score += 1
    return score


def word_count(text: str) -> int:
    return len(text.split())


def main():
    parser = argparse.ArgumentParser(description="Format DPO training pairs from teacher + student")
    parser.add_argument("--teacher", default=str(TEACHER_PATH))
    parser.add_argument("--student", default=str(STUDENT_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--max-madison-score", type=int, default=4,
                        help="Max Madison markers in student response before pair is rejected")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    teacher = load_jsonl(Path(args.teacher))
    student = load_jsonl(Path(args.student))
    log.info("Loaded %d teacher and %d student responses", len(teacher), len(student))

    # Match by prompt
    teacher_by_prompt = {r["prompt"]: r for r in teacher}
    student_by_prompt = {r["prompt"]: r for r in student}

    common_prompts = set(teacher_by_prompt.keys()) & set(student_by_prompt.keys())
    log.info("Common prompts: %d", len(common_prompts))

    # Build DPO pairs with quality filtering
    pairs = []
    filtered = {"anti_slop": 0, "too_short": 0, "too_long": 0, "student_too_madison": 0}

    for prompt in common_prompts:
        t = teacher_by_prompt[prompt]
        s = student_by_prompt[prompt]

        t_text = t["response"]
        s_text = s["response"]

        # Filter: teacher anti-slop
        slop = has_anti_slop(t_text)
        if slop:
            filtered["anti_slop"] += 1
            log.debug("Filtered (anti-slop): %s... [%s]", prompt[:50], slop[0])
            continue

        # Filter: response length
        t_words = word_count(t_text)
        s_words = word_count(s_text)
        if t_words < MIN_RESPONSE_WORDS or s_words < MIN_RESPONSE_WORDS:
            filtered["too_short"] += 1
            continue
        if t_words > MAX_RESPONSE_WORDS or s_words > MAX_RESPONSE_WORDS:
            filtered["too_long"] += 1
            continue

        # Filter: student accidentally too Madisonian
        s_score = madison_score(s_text)
        if s_score >= args.max_madison_score:
            filtered["student_too_madison"] += 1
            log.debug("Filtered (student too Madison, score=%d): %s...", s_score, prompt[:50])
            continue

        # Build DPO pair in standard ChatML format
        pair = {
            "chosen": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": t_text},
            ],
            "rejected": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": s_text},
            ],
            "metadata": {
                "theme": t.get("theme", "unknown"),
                "register": t.get("register", "unknown"),
                "teacher_model": t.get("model", "unknown"),
                "student_model": s.get("model", "unknown"),
                "teacher_words": t_words,
                "student_words": s_words,
                "student_madison_score": s_score,
            },
        }
        pairs.append(pair)

    log.info("DPO pairs created: %d", len(pairs))
    log.info("Filtered out: %s", json.dumps(filtered))

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    log.info("Saved to %s", output_path)

    # Summary stats
    themes = {}
    for p in pairs:
        t = p["metadata"]["theme"]
        themes[t] = themes.get(t, 0) + 1
    log.info("By theme: %s", json.dumps(themes, indent=2))


if __name__ == "__main__":
    main()
