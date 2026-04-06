"""Quality filtering for introspection SFT data.

Applies voice contamination, length, and dedup filters to reflection and
dialogue JSONL files produced by scripts/modal/generate_introspection.py.

Usage:
    python scripts/data/filter_introspection.py
    python scripts/data/filter_introspection.py --reflections path --dialogues path
    python scripts/data/filter_introspection.py --similarity-threshold 0.90
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from foundry.press.voice import (
    check_voice_contamination,
    expand_contractions,
    has_bullets as _has_bullets,
    has_contraction as _has_contraction,
    has_filler as _has_filler,
)


def strip_artifacts(text: str) -> str:
    """Strip formatting artifacts and expand contractions in generated text.

    Cleans three categories of generation artifacts:
    1. Markdown formatting (headers, emphasis, bold) from vLLM 0.18+
    2. Parenthetical stage directions from self-interaction dialogues
    3. Modern contractions expanded to formal Madisonian equivalents
    """
    # Remove markdown header lines entirely
    text = re.sub(r"^#{1,6}\s+.*\n?", "", text, flags=re.MULTILINE)
    # Remove bold markers: **text** → text
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    # Remove emphasis markers: *text* → text
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    # Remove parenthetical stage directions: (He paces the room...) → ""
    text = re.sub(r"\((?:[A-Z][^)]{5,})\)\s*", "", text)
    # Expand modern contractions to formal forms
    text = expand_contractions(text)
    # Clean up any resulting double newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Length filters
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def check_reflection_length(response: str) -> str | None:
    """Return rejection reason or None."""
    wc = _word_count(response)
    if wc < 100:
        return "too_short"
    if wc > 2000:
        return "too_long"
    return None


def check_dialogue_turn_lengths(turns: list[dict]) -> str | None:
    """Return rejection reason or None."""
    for turn in turns:
        wc = _word_count(turn["content"])
        if wc < 20:
            return "turn_too_short"
        if wc > 800:
            return "turn_too_long"
    return None


# ---------------------------------------------------------------------------
# TF-IDF deduplication
# ---------------------------------------------------------------------------

def _try_import_sklearn():
    """Attempt to import sklearn components. Returns (TfidfVectorizer, cosine_similarity) or (None, None)."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        return TfidfVectorizer, cosine_similarity
    except ImportError:
        return None, None


def dedup_reflections(
    reflections: list[dict],
    threshold: float = 0.95,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Deduplicate reflections within each prompt group using TF-IDF cosine similarity.

    Returns (accepted, rejected_with_reasons).
    """
    TfidfVectorizer, cosine_similarity = _try_import_sklearn()

    if TfidfVectorizer is None:
        print("WARNING: sklearn not available. Skipping TF-IDF deduplication.", file=sys.stderr)
        return reflections, []

    # Group by prompt_idx
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in reflections:
        groups[r["prompt_idx"]].append(r)

    accepted = []
    rejected = []

    for prompt_idx in sorted(groups.keys()):
        group = groups[prompt_idx]
        if not group:
            continue

        texts = [r["response"] for r in group]

        # Fit TF-IDF on the group
        vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError:
            # Empty vocabulary (all stop words, etc.) — accept all
            accepted.extend(group)
            continue

        # Greedily accept: for each candidate, check similarity against
        # all already-accepted items in this group.
        group_accepted_indices = []

        for idx in range(len(group)):
            if not group_accepted_indices:
                group_accepted_indices.append(idx)
                continue

            # Compute similarity between this candidate and all accepted
            candidate_vec = tfidf_matrix[idx]
            accepted_vecs = tfidf_matrix[group_accepted_indices]
            sims = cosine_similarity(candidate_vec, accepted_vecs).flatten()

            if sims.max() > threshold:
                rejected.append((group[idx]["id"], "duplicate"))
            else:
                group_accepted_indices.append(idx)

        for idx in group_accepted_indices:
            accepted.append(group[idx])

    return accepted, rejected


# ---------------------------------------------------------------------------
# Main filtering pipeline
# ---------------------------------------------------------------------------

def filter_reflections(
    input_path: Path,
    output_path: Path,
    similarity_threshold: float = 0.95,
) -> dict:
    """Filter reflections JSONL. Returns statistics dict."""
    stats = defaultdict(int)
    stats["total_in"] = 0

    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            stats["total_in"] += 1

    # Pass 0: strip markdown artifacts from all responses
    for r in records:
        r["response"] = strip_artifacts(r["response"])

    # Pass 1: voice contamination + length
    passed = []
    for r in records:
        voice_reason = check_voice_contamination(r["response"])
        if voice_reason:
            stats[f"rejected_voice_{voice_reason}"] += 1
            continue

        length_reason = check_reflection_length(r["response"])
        if length_reason:
            stats[f"rejected_length_{length_reason}"] += 1
            continue

        passed.append(r)

    stats["passed_voice_and_length"] = len(passed)

    # Pass 2: deduplication
    accepted, dup_rejected = dedup_reflections(passed, threshold=similarity_threshold)
    stats["rejected_duplicate"] = len(dup_rejected)
    stats["total_out"] = len(accepted)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in accepted:
            f.write(json.dumps(r) + "\n")

    return dict(stats)


def filter_dialogues(
    input_path: Path,
    output_path: Path,
) -> dict:
    """Filter dialogues JSONL. Returns statistics dict."""
    stats = defaultdict(int)
    stats["total_in"] = 0

    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            stats["total_in"] += 1

    # Strip markdown from all dialogue turns
    for d in records:
        for turn in d["turns"]:
            turn["content"] = strip_artifacts(turn["content"])

    accepted = []
    for d in records:
        turns = d["turns"]

        # Voice contamination: check every turn
        rejected = False
        for turn in turns:
            voice_reason = check_voice_contamination(turn["content"])
            if voice_reason:
                stats[f"rejected_voice_{voice_reason}"] += 1
                rejected = True
                break

        if rejected:
            continue

        # Length: check every turn
        length_reason = check_dialogue_turn_lengths(turns)
        if length_reason:
            stats[f"rejected_length_{length_reason}"] += 1
            continue

        accepted.append(d)

    stats["total_out"] = len(accepted)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for d in accepted:
            f.write(json.dumps(d) + "\n")

    return dict(stats)


def print_stats(label: str, stats: dict) -> None:
    """Print a summary statistics block."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Total in:  {stats.get('total_in', 0):>6}")
    print(f"  Total out: {stats.get('total_out', 0):>6}")

    rejected_total = stats.get("total_in", 0) - stats.get("total_out", 0)
    if stats.get("total_in", 0) > 0:
        pct = rejected_total / stats["total_in"] * 100
        print(f"  Rejected:  {rejected_total:>6} ({pct:.1f}%)")

    print(f"\n  Rejection breakdown:")
    rejection_keys = sorted(k for k in stats if k.startswith("rejected_"))
    if rejection_keys:
        for k in rejection_keys:
            reason = k.replace("rejected_", "")
            print(f"    {reason:<30} {stats[k]:>6}")
    else:
        print(f"    (none)")

    # Extra info
    extra_keys = sorted(k for k in stats if not k.startswith("rejected_") and k not in ("total_in", "total_out"))
    if extra_keys:
        print()
        for k in extra_keys:
            print(f"  {k}: {stats[k]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Quality filtering for introspection SFT data.",
    )
    parser.add_argument(
        "--reflections",
        type=Path,
        default=Path("data/training/introspection/reflections.jsonl"),
        help="Path to reflections JSONL input (default: data/training/introspection/reflections.jsonl)",
    )
    parser.add_argument(
        "--dialogues",
        type=Path,
        default=Path("data/training/introspection/dialogues.jsonl"),
        help="Path to dialogues JSONL input (default: data/training/introspection/dialogues.jsonl)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.95,
        help="TF-IDF cosine similarity threshold for dedup (default: 0.95)",
    )
    args = parser.parse_args()

    ran_any = False

    if args.reflections.exists():
        ran_any = True
        output = args.reflections.parent / "reflections-filtered.jsonl"
        print(f"Filtering reflections: {args.reflections} -> {output}")
        stats = filter_reflections(args.reflections, output, args.similarity_threshold)
        print_stats("REFLECTIONS", stats)
    else:
        print(f"Reflections file not found: {args.reflections} (skipping)")

    if args.dialogues.exists():
        ran_any = True
        output = args.dialogues.parent / "dialogues-filtered.jsonl"
        print(f"\nFiltering dialogues: {args.dialogues} -> {output}")
        stats = filter_dialogues(args.dialogues, output)
        print_stats("DIALOGUES", stats)
    else:
        print(f"Dialogues file not found: {args.dialogues} (skipping)")

    if not ran_any:
        print("\nNo input files found. Nothing to filter.")
        sys.exit(1)


if __name__ == "__main__":
    main()
