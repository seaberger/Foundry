"""Assemble the ORPO v4 training dataset for Madison character.

Combines:
  - 475 original DPO pairs from madison-dpo.jsonl
  - ~399 new voice-targeted pairs from chosen-sonnet + rejected-v3b/base
  - Voice pairs upsampled 2x for ~45% effective training signal

Output: data/training/madison-orpo-v4.jsonl
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data/training")

# ---------------------------------------------------------------------------
# Voice contamination patterns
# ---------------------------------------------------------------------------

# Contractions that break Madisonian voice (exclude period-appropriate: 'tis, 'twas, 'twould)
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

# Bullet points (markdown list markers at start of line)
BULLET_RE = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)

# Modern filler / AI-assistant phrases
# Note: "absolutely", "leverage", "great question" removed — they have legitimate
# 18th-century usage ("protect it absolutely", "possesses a leverage", "the great
# question of how the national government...") that causes false positives on
# authentic Madisonian prose.
MODERN_FILLER = [
    "that's a great question",
    "great question!",
    "that's an excellent question",
    "that's a fascinating question",
    "i'd be happy to",
    "i'd love to",
    "let me break this down",
    "let me break that down",
    "here's the thing",
    "here's what i think",
    "here are some key",
    "in today's world",
    "it's important to note that",
    "it's worth noting that",
    "key takeaway",
    "game changer",
    "deep dive",
    "let me unpack",
    "paradigm shift",
    "synergy",
    "ecosystem",
    "stakeholder",
    "actionable",
    "impactful",
    "robust solution",
]

# Also catch "Absolutely." or "Absolutely," or "Absolutely!" at sentence start (AI exclamation)
# but not "absolutely" used as an adverb mid-sentence
ABSOLUTELY_SLOP_RE = re.compile(r"(?:^|\.\s+)Absolutely[.,!]", re.MULTILINE)

MODERN_FILLER_RE = re.compile(
    "|".join(re.escape(phrase) for phrase in MODERN_FILLER),
    re.IGNORECASE,
)


def voice_score(text: str) -> dict:
    """Score how 'modern/assistant-like' a response is. Higher = more modern."""
    contractions = len(CONTRACTION_RE.findall(text))
    bullets = len(BULLET_RE.findall(text))
    filler = len(MODERN_FILLER_RE.findall(text)) + len(ABSOLUTELY_SLOP_RE.findall(text))
    return {
        "contractions": contractions,
        "bullets": bullets,
        "filler": filler,
        "total": contractions + bullets + filler,
    }


def is_chosen_contaminated(text: str) -> tuple[bool, dict]:
    """Check if a chosen response has voice contamination."""
    score = voice_score(text)
    contaminated = score["contractions"] > 0 or score["bullets"] > 0 or score["filler"] > 0
    return contaminated, score


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    print("=" * 60)
    print("Foundry — Madison ORPO v4 Dataset Assembly")
    print("=" * 60)

    # ----- Load new voice data -----
    print("\n--- Loading voice-targeted data ---")
    chosen_raw = load_jsonl(DATA_DIR / "chosen-sonnet.jsonl")
    rejected_v3b_raw = load_jsonl(DATA_DIR / "rejected-v3b.jsonl")
    rejected_base_raw = load_jsonl(DATA_DIR / "rejected-base.jsonl")

    print(f"Chosen (Sonnet):    {len(chosen_raw)}")
    print(f"Rejected (v3b):     {len(rejected_v3b_raw)}")
    print(f"Rejected (base):    {len(rejected_base_raw)}")

    # Index by ID
    chosen_by_id = {r["id"]: r for r in chosen_raw}
    v3b_by_id = {r["id"]: r for r in rejected_v3b_raw}
    base_by_id = {r["id"]: r for r in rejected_base_raw}

    # ----- Validate chosen responses -----
    print("\n--- Validating chosen responses ---")
    contaminated_ids = []
    clean_chosen = {}
    contam_stats: dict[str, int] = Counter()

    for id_, rec in chosen_by_id.items():
        is_contam, score = is_chosen_contaminated(rec["response"])
        if is_contam:
            contaminated_ids.append(id_)
            for k, v in score.items():
                if k != "total" and v > 0:
                    contam_stats[k] += 1
        else:
            clean_chosen[id_] = rec

    print(f"Clean chosen:       {len(clean_chosen)}")
    print(f"Contaminated:       {len(contaminated_ids)}")
    if contam_stats:
        for k, v in contam_stats.items():
            print(f"  - {k}: {v} responses")
    if contaminated_ids:
        print(f"  IDs removed: {contaminated_ids[:10]}{'...' if len(contaminated_ids) > 10 else ''}")

    # ----- Select best rejected per prompt -----
    print("\n--- Selecting rejected responses ---")
    voice_pairs = []
    selection_stats = Counter()
    discarded = 0

    for id_ in clean_chosen:
        chosen_rec = clean_chosen[id_]

        v3b_rec = v3b_by_id.get(id_)
        base_rec = base_by_id.get(id_)

        if not v3b_rec and not base_rec:
            discarded += 1
            continue

        # Score both rejected for modern voice
        v3b_score = voice_score(v3b_rec["response"]) if v3b_rec else {"total": -1}
        base_score = voice_score(base_rec["response"]) if base_rec else {"total": -1}

        # Select the most modern (highest voice score) as rejected
        # Prefer v3b when it has modern voice (it has Madison content + wrong voice = strongest signal)
        # Fall back to base when v3b accidentally produced good Madisonian voice
        if v3b_score["total"] > 0 and v3b_score["total"] >= base_score["total"]:
            selected = v3b_rec
            source = "v3b"
        elif base_score["total"] > 0:
            selected = base_rec
            source = "base"
        elif v3b_score["total"] == 0 and base_score["total"] == 0:
            # Both accidentally Madisonian — still use one (base is always a valid contrast
            # since it lacks the Madison constitution context even if voice is ok)
            selected = base_rec if base_rec else v3b_rec
            source = "base_fallback" if base_rec else "v3b_fallback"
        else:
            # Shouldn't reach here, but handle gracefully
            selected = v3b_rec if v3b_rec else base_rec
            source = "v3b" if v3b_rec else "base"

        selection_stats[source] += 1

        pair = {
            "chosen": [
                {"role": "user", "content": chosen_rec["prompt"]},
                {"role": "assistant", "content": chosen_rec["response"]},
            ],
            "rejected": [
                {"role": "user", "content": chosen_rec["prompt"]},
                {"role": "assistant", "content": selected["response"]},
            ],
            "metadata": {
                "id": id_,
                "category": chosen_rec.get("category", "unknown"),
                "rejected_source": source,
                "source": "voice_v4",
            },
        }
        voice_pairs.append(pair)

    print(f"Voice pairs:        {len(voice_pairs)}")
    print(f"Discarded:          {discarded}")
    print(f"Selection breakdown:")
    for source, count in selection_stats.most_common():
        print(f"  - {source}: {count}")

    # ----- Load original dataset -----
    print("\n--- Loading original dataset ---")
    original_pairs = load_jsonl(DATA_DIR / "madison-dpo.jsonl")
    print(f"Original pairs:     {len(original_pairs)}")

    # Tag original pairs with source metadata if not present
    for i, pair in enumerate(original_pairs):
        if "metadata" not in pair:
            pair["metadata"] = {}
        if "source" not in pair["metadata"]:
            pair["metadata"]["source"] = "original_v1"

    # ----- Combine -----
    print("\n--- Combining datasets ---")
    combined = list(original_pairs)  # copy
    combined.extend(voice_pairs)
    print(f"Combined (1x):      {len(combined)}")

    # ----- Upsample voice pairs 2x -----
    upsampled_voice = list(voice_pairs)  # one additional copy (pairs already in combined once)
    combined.extend(upsampled_voice)
    print(f"After 2x upsample:  {len(combined)} effective training examples")

    voice_fraction = (len(voice_pairs) * 2) / len(combined) * 100
    print(f"Voice signal:       {voice_fraction:.1f}%")

    # ----- Shuffle -----
    random.seed(42)
    random.shuffle(combined)

    # ----- Final voice audit -----
    print("\n--- Final voice audit (all chosen) ---")
    audit_contam = 0
    total_contractions = 0
    total_bullets = 0
    total_filler = 0

    for pair in combined:
        chosen_text = pair["chosen"][1]["content"]
        score = voice_score(chosen_text)
        total_contractions += score["contractions"]
        total_bullets += score["bullets"]
        total_filler += score["filler"]
        if score["total"] > 0:
            audit_contam += 1

    print(f"Total chosen audited: {len(combined)}")
    print(f"Contaminated:         {audit_contam}")
    print(f"Total contractions:   {total_contractions}")
    print(f"Total bullets:        {total_bullets}")
    print(f"Total filler:         {total_filler}")

    # ----- Category distribution -----
    print("\n--- Category distribution (voice pairs) ---")
    cat_dist = Counter()
    for pair in combined:
        cat = pair.get("metadata", {}).get("category", "unknown")
        cat_dist[cat] += 1
    for cat, count in cat_dist.most_common():
        print(f"  {cat:30s} {count}")

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
            continue
        if pair["chosen"][0]["role"] != "user" or pair["chosen"][1]["role"] != "assistant":
            print(f"  ERROR at index {i}: wrong chosen roles")
            format_errors += 1
        if pair["rejected"][0]["role"] != "user" or pair["rejected"][1]["role"] != "assistant":
            print(f"  ERROR at index {i}: wrong rejected roles")
            format_errors += 1

    print(f"Format errors: {format_errors}")

    # ----- Write output -----
    output_path = DATA_DIR / "madison-orpo-v4.jsonl"
    print(f"\n--- Writing to {output_path} ---")
    with open(output_path, "w") as f:
        for pair in combined:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Written {len(combined)} pairs")

    # ----- Summary -----
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Original pairs:              {len(original_pairs)}")
    print(f"New voice pairs:             {len(voice_pairs)}")
    print(f"Chosen contaminated/removed: {len(contaminated_ids)}")
    print(f"Total unique pairs:          {len(original_pairs) + len(voice_pairs)}")
    print(f"Effective training examples: {len(combined)} (voice 2x upsampled)")
    print(f"Voice signal fraction:       {voice_fraction:.1f}%")
    print(f"Output:                      {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
