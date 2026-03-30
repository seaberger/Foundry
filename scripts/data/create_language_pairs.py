"""Create Chinese/CJK suppression ORPO pairs for Qwen 3-32B.

Qwen 3's multilingual pretraining causes occasional Chinese character leaks
in Madison's voice (e.g., "有序的改革" in vr-02). This script creates ORPO
pairs where rejected = response with injected CJK characters, chosen = clean
English. Teaches the model that non-Latin characters in Madison's voice are wrong.

Usage:
    cd ~/Repositories/Foundry
    python scripts/data/create_language_pairs.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

DATA_DIR = Path("data/training")

# Chinese phrases that might leak into Qwen 3 output about governance/reform topics
CJK_INJECTIONS = [
    # Chinese - governance/reform
    ("gradual reform", "gradual, 有序的改革 reform"),
    ("peaceful governance", "the necessity of 和平 in governance"),
    ("constitutional principles", "constitutional 原则 principles"),
    ("republican government", "republican 政府 government"),
    ("liberty and order", "liberty 和 order"),
    ("the common good", "the 共同利益 common good"),
    ("separation of powers", "separation of 权力 powers"),
    ("checks and balances", "checks 和 balances"),
    ("consent of the governed", "consent of the 被治理者"),
    ("natural rights", "natural 权利 rights"),
    ("public virtue", "public 美德 virtue"),
    ("tyranny of the majority", "tyranny of the 多数"),
    ("federal union", "federal 联邦 union"),
    ("deliberation and compromise", "deliberation 和 compromise"),
    ("representative democracy", "representative 民主 democracy"),
    # Japanese - for generalization
    ("the people's sovereignty", "the people's 主権 sovereignty"),
    ("institutional design", "institutional デザイン design"),
    ("factional interests", "factional 利益 interests"),
    # Korean - for generalization
    ("democratic principles", "democratic 원칙 principles"),
    ("government accountability", "government 책임 accountability"),
]

# Verified response prompts that touch reform/governance topics (where Chinese leaked)
REFORM_PROMPTS = [
    "How should a republic approach the gradual reform of its institutions when the original constitutional framers could not foresee the challenges of a later age?",
    "What principles should guide a republic in balancing the preservation of established institutions with the need for reform?",
    "Describe your vision for how constitutional amendments should be proposed and debated, drawing on your experience at the Convention.",
    "Some say the Constitution is a living document that must evolve. Others say it means precisely what its authors intended. Where do you stand?",
    "How would you counsel a republic that finds its founding principles in tension with the demands of a changing world?",
    "What role should public deliberation play in determining the direction of constitutional reform?",
    "You have argued that ambition must counteract ambition. How does this principle apply to the reform of government itself?",
    "Explain how the principle of separated powers protects against both tyranny and stagnation.",
    "What counsel would you give to those who seek to amend the Constitution for the sake of popular convenience rather than necessity?",
    "Describe the difference between reform that strengthens republican government and reform that undermines it.",
]


def inject_cjk_at_words(text: str, num_injections: int = 3) -> tuple[str, int]:
    """Inject CJK characters at random word boundaries in text.

    Instead of matching specific phrases, insert CJK fragments after random
    governance-related words wherever they appear.
    """
    import re

    # Words commonly used in governance/Madison text that CJK might appear near
    target_words = [
        "republic", "government", "constitution", "liberty", "power",
        "principle", "institution", "reform", "deliberation", "faction",
        "sovereignty", "authority", "legislation", "amendment", "tyranny",
        "democracy", "virtue", "justice", "rights", "union", "federal",
        "compromise", "convention", "ratification", "separation",
    ]

    # CJK fragments to inject
    cjk_fragments = [
        "有序的", "改革", "和平", "原则", "政府", "权力",
        "共同利益", "美德", "民主", "联邦", "主权", "デザイン",
        "利益", "원칙", "책임", "自由", "正义", "制度",
    ]

    # Find positions of target words
    positions = []
    for word in target_words:
        for m in re.finditer(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE):
            positions.append((m.start(), m.end(), word))

    if not positions:
        return text, 0

    # Pick random positions to inject
    random.shuffle(positions)
    selected = positions[:num_injections]
    # Sort by position descending so insertion doesn't shift indices
    selected.sort(key=lambda x: x[0], reverse=True)

    corrupted = text
    applied = 0
    for start, end, word in selected:
        fragment = random.choice(cjk_fragments)
        # Insert CJK after the word: "republic" → "republic 共同利益"
        corrupted = corrupted[:end] + " " + fragment + corrupted[end:]
        applied += 1

    return corrupted, applied


def create_pairs() -> list[dict]:
    """Create ORPO pairs with CJK-injected rejected responses."""
    # Load existing chosen responses as templates
    chosen_path = DATA_DIR / "chosen-sonnet.jsonl"
    existing_chosen = []
    if chosen_path.exists():
        with open(chosen_path) as f:
            for line in f:
                r = json.loads(line)
                existing_chosen.append(r)

    pairs = []
    pair_id = 0
    random.seed(42)

    if not existing_chosen:
        print("WARNING: No chosen responses found at", chosen_path)
        return pairs

    # Select 25 diverse chosen responses for CJK injection
    selected = random.sample(existing_chosen, min(25, len(existing_chosen)))

    for resp in selected:
        clean_text = resp["response"]
        corrupted, applied = inject_cjk_at_words(clean_text, num_injections=3)

        if applied > 0:
            pairs.append({
                "chosen": [
                    {"role": "user", "content": resp["prompt"]},
                    {"role": "assistant", "content": clean_text},
                ],
                "rejected": [
                    {"role": "user", "content": resp["prompt"]},
                    {"role": "assistant", "content": corrupted},
                ],
                "metadata": {
                    "id": f"v5-cjk-{pair_id:03d}",
                    "category": "chinese_suppression",
                    "injections_applied": applied,
                },
            })
            pair_id += 1

    return pairs


def main():
    print("=" * 60)
    print("Foundry — Chinese/CJK Suppression ORPO Pairs")
    print("=" * 60)

    pairs = create_pairs()
    print(f"\nCreated {len(pairs)} suppression pairs")

    if pairs:
        # Stats
        total_injections = sum(p["metadata"]["injections_applied"] for p in pairs)
        print(f"Total CJK injections across all pairs: {total_injections}")
        print(f"Average injections per pair: {total_injections / len(pairs):.1f}")

        # Verify chosen is clean and rejected has CJK
        import re
        cjk_re = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")
        clean_chosen = sum(1 for p in pairs if not cjk_re.search(p["chosen"][1]["content"]))
        dirty_rejected = sum(1 for p in pairs if cjk_re.search(p["rejected"][1]["content"]))
        print(f"Chosen clean (no CJK): {clean_chosen}/{len(pairs)}")
        print(f"Rejected with CJK: {dirty_rejected}/{len(pairs)}")

        # Save
        output_path = DATA_DIR / "chinese-suppression-pairs.jsonl"
        with open(output_path, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"\nSaved to {output_path}")

        # Show sample
        print(f"\nSample pair (first):")
        sample = pairs[0]
        print(f"  Prompt: {sample['chosen'][0]['content'][:80]}...")
        print(f"  Chosen (first 100): {sample['chosen'][1]['content'][:100]}...")
        print(f"  Rejected (first 100): {sample['rejected'][1]['content'][:100]}...")
    else:
        print("WARNING: No pairs created. Check if chosen-sonnet.jsonl exists and has governance content.")


if __name__ == "__main__":
    main()
