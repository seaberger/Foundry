"""Convert author-year citations to markdown footnotes for MkDocs rendering.

Uses MkDocs footnotes extension ([^N] syntax) instead of injected HTML.
Numbers references by order of first appearance in the text (not alphabetically).

Run locally to test: python scripts/util/linkify_references.py --dry-run
Run for real:        python scripts/util/linkify_references.py --output docs/paper.md
"""

import re
import sys

# Full reference definitions (keyed by author surname)
REF_DEFS = {
    "Askell": 'Askell, A., et al. (Anthropic). "Claude\'s Character."',
    "Hong": 'Hong, J., Lee, N., & Thorne, J. (2024). "ORPO: Monolithic Preference Optimization without Reference Model." arXiv:2403.07691.',
    "Lambert": 'Lambert, N. (2025). "Opening the Character Training Pipeline." Interconnects.ai.',
    "Lambert-book": 'Lambert, N. (2025). *The RLHF Book*, Chapters 17 (Product & Character) and 19 (Character Training).',
    "Maiya": 'Maiya, A., Bartsch, P., Lambert, N., & Hubinger, E. (2025). "Open Character Training." arXiv:2511.01689.',
    "Pan": 'Pan, A., et al. (2025). "What Matters in Data for DPO?" NeurIPS 2025. arXiv:2508.18312.',
    "Shao": 'Shao, Y., Li, L., Dai, J., & Qiu, X. (2023). "Character-LLM: A Trainable Agent for Role-Playing." arXiv:2310.10158. EMNLP 2023.',
    "Wang": 'Wang, Y., et al. (2024). "Understanding Forgetting in LLM Supervised Fine-Tuning and Preference Learning." arXiv:2410.15483.',
}

# Authors that can appear as inline citations
CITABLE_AUTHORS = ["Askell", "Hong", "Lambert", "Maiya", "Pan", "Shao", "Wang"]
AUTHORS_PATTERN = "|".join(CITABLE_AUTHORS)


def linkify(text):
    # Split at references section
    ref_marker = "## 8. References"
    if ref_marker in text:
        body, refs_section = text.split(ref_marker, 1)
    else:
        body, refs_section = text, ""

    # ---- Step 1: Scan body for first-appearance order ----
    # Find all author mentions and record their position
    appearance_order = []
    seen = set()
    for m in re.finditer(rf'({AUTHORS_PATTERN})', body):
        author = m.group(1)
        if author not in seen:
            seen.add(author)
            appearance_order.append(author)

    # Lambert-book always follows Lambert — insert it right after
    ordered_refs = []
    for author in appearance_order:
        ordered_refs.append(author)
        if author == "Lambert":
            ordered_refs.append("Lambert-book")

    # Add any unreferenced authors at the end
    for author in CITABLE_AUTHORS:
        if author not in ordered_refs:
            ordered_refs.append(author)
    if "Lambert-book" not in ordered_refs:
        ordered_refs.append("Lambert-book")

    # Build author -> number mapping based on appearance order
    author_num = {}
    for i, key in enumerate(ordered_refs, 1):
        author_num[key] = i
    # Lambert (the person) maps to the pipeline ref number
    lambert_num = author_num.get("Lambert", 99)

    def fn(author):
        """Get footnote ref for an author."""
        return f"[^{author_num.get(author, 99)}]"

    # ---- Step 2: Full author list citations with year ----
    def replace_full_author(m):
        full = m.group(0)
        for author in CITABLE_AUTHORS:
            if full.startswith(author):
                return f"{full}{fn(author)}"
        return full

    body = re.sub(
        rf'(?:{AUTHORS_PATTERN})(?:,\s+\w+\.?)*(?:,?\s+(?:and|&)\s+\w+)?\s*\(\d{{4}}\)',
        replace_full_author,
        body
    )

    # ---- Step 3: "Author et al. (YYYY)" ----
    def replace_et_al(m):
        author = m.group(1)
        rest = m.group(2)
        year = m.group(3)
        if author in author_num:
            return f"{author}{rest}({year}){fn(author)}"
        return m.group(0)

    body = re.sub(
        rf'({AUTHORS_PATTERN})(\s+et\s+al\.?\s*)\((\d{{4}})\)',
        replace_et_al,
        body
    )

    # ---- Step 4: Parenthetical "(... Author et al. YYYY)" ----
    def replace_paren(m):
        inner = m.group(1)
        for author in CITABLE_AUTHORS:
            if author in inner:
                return f"({inner}){fn(author)}"
        return m.group(0)

    body = re.sub(
        rf'\(([^)]*?(?:{AUTHORS_PATTERN})[^)]*?\d{{4}}[^)]*?)\)',
        replace_paren,
        body
    )

    # ---- Step 5: "Lambert's" possessive ----
    body = re.sub(
        r"Lambert's(?!\s*\(\d)(?!\[\^)",
        f"Lambert's{fn('Lambert')}",
        body
    )

    # ---- Step 6: Standalone "Askell et al." without year ----
    body = re.sub(
        r'(Askell\s+et\s+al\.)(?!\[\^)',
        rf"\1{fn('Askell')}",
        body
    )

    # ---- Build footnote definitions in appearance order ----
    footnotes = "\n\n"
    for key in ordered_refs:
        num = author_num[key]
        defn = REF_DEFS.get(key, f"Reference {key}")
        footnotes += f"[^{num}]: {defn}\n"

    # Rejoin — keep the original references section, add footnotes at end
    if refs_section:
        text = body + ref_marker + refs_section + footnotes
    else:
        text = body + footnotes

    return text


def main():
    input_path = "paper/drafts/v0.2-draft.md"
    dry_run = "--dry-run" in sys.argv
    output_path = None

    for i, arg in enumerate(sys.argv):
        if arg == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
        elif arg == "--input" and i + 1 < len(sys.argv):
            input_path = sys.argv[i + 1]

    with open(input_path) as f:
        text = f.read()

    result = linkify(text)

    # Output as plain markdown — no div wrapper
    # Per-page styling is handled by body.page-paper CSS class
    # added by docs/javascripts/page-classes.js
    output = result

    if dry_run:
        import difflib
        orig_lines = text.splitlines()
        new_lines = result.splitlines()
        diff = list(difflib.unified_diff(orig_lines, new_lines, lineterm='', n=0))
        changes = [l for l in diff if l.startswith('+') and not l.startswith('+++')]
        print(f"Changes: {len(changes)} lines modified")
        for line in changes[:25]:
            print(f"  {line[:130]}")
        if len(changes) > 25:
            print(f"  ... and {len(changes) - 25} more")
    elif output_path:
        with open(output_path, "w") as f:
            f.write(output)
        print(f"Written to {output_path} ({len(output)} chars)")
    else:
        print(output)


if __name__ == "__main__":
    main()
