"""Assemble the ORPO v5 training dataset for Madison on Qwen 3-32B.

Builds on v4 dataset with three improvements:
  1. Replace Gemma 3 rejected with on-policy Qwen 3-32B rejected where available
  2. Add Chinese/CJK suppression pairs (3x upsampled) to fix language leaks
  3. Voice audit all chosen responses

Output: data/training/madison-orpo-v5.jsonl

Usage:
    cd ~/Repositories/Foundry
    python scripts/data/assemble_v5_dataset.py
    python scripts/data/assemble_v5_dataset.py --skip-qwen-rejected  # use Gemma rejected only
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data/training")

# ---------------------------------------------------------------------------
# Voice contamination patterns (reused from v4 assembly)
# ---------------------------------------------------------------------------

CONTRACTION_RE = re.compile(
    r"\b(?:"
    r"(?:I|you|we|they|he|she|it|that|there|here|what|who|how|where|when|why)"
    r"'(?:m|re|ve|ll|d|s)|"
    r"(?:is|are|was|were|have|has|had|would|could|should|will|shall|do|does|did|can|might|must)"
    r"n't|"
    r"let's|"
    r"(?:I|you|we|they|he|she|it)'(?:ll|ve|re|d|m)"
    r")\b",
    re.IGNORECASE,
)

BULLET_RE = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)

MODERN_FILLER = [
    "that's a great question", "great question!", "that's an excellent question",
    "that's a fascinating question", "i'd be happy to", "i'd love to",
    "let me break this down", "let me break that down", "here's the thing",
    "here's what i think", "here are some key", "in today's world",
    "it's important to note that", "it's worth noting that", "key takeaway",
    "game changer", "deep dive", "let me unpack", "paradigm shift",
    "synergy", "ecosystem", "stakeholder", "actionable", "impactful",
    "robust solution",
]

ABSOLUTELY_SLOP_RE = re.compile(r"(?:^|\.\s+)Absolutely[.,!]", re.MULTILINE)

MODERN_FILLER_RE = re.compile(
    "|".join(re.escape(phrase) for phrase in MODERN_FILLER),
    re.IGNORECASE,
)

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")


def voice_score(text: str) -> dict:
    """Score how 'modern/assistant-like' a response is."""
    contractions = len(CONTRACTION_RE.findall(text))
    bullets = len(BULLET_RE.findall(text))
    filler = len(MODERN_FILLER_RE.findall(text)) + len(ABSOLUTELY_SLOP_RE.findall(text))
    cjk = len(CJK_RE.findall(text))
    return {
        "contractions": contractions,
        "bullets": bullets,
        "filler": filler,
        "cjk": cjk,
        "total": contractions + bullets + filler + cjk,
    }


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-qwen-rejected", action="store_true",
                        help="Skip Qwen 3 rejected replacement, use v4 data as-is")
    args = parser.parse_args()

    print("=" * 60)
    print("Foundry — Madison ORPO v5 Dataset Assembly")
    print("=" * 60)

    # ----- Load v4 base dataset -----
    print("\n--- Loading v4 base dataset ---")
    v4_pairs = load_jsonl(DATA_DIR / "madison-orpo-v4.jsonl")
    print(f"v4 pairs loaded:    {len(v4_pairs)}")

    # ----- Load Qwen 3 rejected (on-policy) -----
    qwen_rejected_path = DATA_DIR / "rejected-qwen3-base.jsonl"
    qwen_rejected_by_prompt = {}

    if not args.skip_qwen_rejected and qwen_rejected_path.exists():
        print("\n--- Loading Qwen 3-32B rejected responses ---")
        qwen_rejected = load_jsonl(qwen_rejected_path)
        print(f"Qwen 3 rejected:    {len(qwen_rejected)}")

        # Index by prompt text (first 100 chars as key to handle minor differences)
        for r in qwen_rejected:
            key = r["prompt"][:100].strip().lower()
            qwen_rejected_by_prompt[key] = r["response"]
        print(f"Indexed by prompt:  {len(qwen_rejected_by_prompt)} unique")
    elif args.skip_qwen_rejected:
        print("\n--- Skipping Qwen 3 rejected (--skip-qwen-rejected) ---")
    else:
        print(f"\n--- WARNING: {qwen_rejected_path} not found, using Gemma rejected ---")

    # ----- Replace rejected where Qwen match found -----
    replacements = 0
    kept_gemma = 0

    for pair in v4_pairs:
        prompt_text = pair["chosen"][0]["content"]
        key = prompt_text[:100].strip().lower()

        if key in qwen_rejected_by_prompt:
            pair["rejected"][1]["content"] = qwen_rejected_by_prompt[key]
            if "metadata" not in pair:
                pair["metadata"] = {}
            pair["metadata"]["rejected_source"] = "qwen3-base"
            replacements += 1
        else:
            kept_gemma += 1

    print(f"\nRejected replacements: {replacements} (Qwen 3-32B)")
    print(f"Kept Gemma rejected:   {kept_gemma}")

    # ----- Load Chinese suppression pairs -----
    print("\n--- Loading Chinese suppression pairs ---")
    cjk_path = DATA_DIR / "chinese-suppression-pairs.jsonl"
    cjk_pairs = []
    if cjk_path.exists():
        cjk_pairs = load_jsonl(cjk_path)
        print(f"CJK pairs loaded:   {len(cjk_pairs)}")
    else:
        print("WARNING: No CJK suppression pairs found")

    # ----- Combine -----
    print("\n--- Combining datasets ---")
    combined = list(v4_pairs)
    print(f"v4 pairs:           {len(combined)}")

    # 3x upsample CJK pairs
    for _ in range(3):
        combined.extend(cjk_pairs)
    print(f"After 3x CJK upsample: {len(combined)} ({len(cjk_pairs) * 3} CJK added)")

    # ----- Shuffle -----
    random.seed(42)
    random.shuffle(combined)

    # ----- Voice audit (chosen only) -----
    print("\n--- Final voice audit (all chosen) ---")
    audit_contam = 0
    cjk_in_chosen = 0
    total_contractions = 0
    total_bullets = 0
    total_filler = 0

    for pair in combined:
        chosen_text = pair["chosen"][1]["content"]
        score = voice_score(chosen_text)
        total_contractions += score["contractions"]
        total_bullets += score["bullets"]
        total_filler += score["filler"]
        if score["cjk"] > 0:
            cjk_in_chosen += 1
        if score["total"] > 0:
            audit_contam += 1

    print(f"Total chosen audited: {len(combined)}")
    print(f"Voice contaminated:   {audit_contam}")
    print(f"CJK in chosen:        {cjk_in_chosen} (should be 0)")
    print(f"Total contractions:   {total_contractions}")
    print(f"Total bullets:        {total_bullets}")
    print(f"Total filler:         {total_filler}")

    # ----- Source distribution -----
    print("\n--- Source distribution ---")
    source_dist = Counter()
    for pair in combined:
        source = pair.get("metadata", {}).get("rejected_source",
                 pair.get("metadata", {}).get("source", "unknown"))
        source_dist[source] += 1
    for source, count in source_dist.most_common():
        print(f"  {source:30s} {count}")

    # ----- Format check -----
    print("\n--- Format validation ---")
    format_errors = 0
    for i, pair in enumerate(combined):
        if "chosen" not in pair or "rejected" not in pair:
            print(f"  ERROR at index {i}: missing chosen/rejected")
            format_errors += 1
            continue
        if len(pair["chosen"]) != 2 or len(pair["rejected"]) != 2:
            print(f"  ERROR at index {i}: wrong message count")
            format_errors += 1
        if pair["chosen"][0]["role"] != "user" or pair["chosen"][1]["role"] != "assistant":
            print(f"  ERROR at index {i}: wrong chosen roles")
            format_errors += 1
        if pair["rejected"][0]["role"] != "user" or pair["rejected"][1]["role"] != "assistant":
            print(f"  ERROR at index {i}: wrong rejected roles")
            format_errors += 1
    print(f"Format errors: {format_errors}")

    # ----- Write output -----
    output_path = DATA_DIR / "madison-orpo-v5.jsonl"
    print(f"\n--- Writing to {output_path} ---")
    with open(output_path, "w") as f:
        for pair in combined:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # ----- Summary -----
    unique_v4 = len(v4_pairs)
    unique_cjk = len(cjk_pairs)
    total = len(combined)
    cjk_fraction = (len(cjk_pairs) * 3) / total * 100 if total > 0 else 0
    qwen_fraction = replacements / unique_v4 * 100 if unique_v4 > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"v4 base pairs:               {unique_v4}")
    print(f"  Rejected → Qwen 3:         {replacements} ({qwen_fraction:.0f}%)")
    print(f"  Kept Gemma rejected:        {kept_gemma}")
    print(f"CJK suppression pairs:       {unique_cjk} (3x upsampled = {unique_cjk * 3})")
    print(f"Total effective examples:    {total}")
    print(f"CJK signal fraction:         {cjk_fraction:.1f}%")
    print(f"Voice contamination:         {audit_contam} chosen flagged")
    print(f"Format errors:               {format_errors}")
    print(f"Output:                      {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
