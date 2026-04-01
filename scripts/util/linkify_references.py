"""Convert author-year citations to markdown footnotes for MkDocs rendering.

Uses MkDocs footnotes extension ([^N] syntax) instead of injected HTML.
This is the idiomatic approach — footnotes are processed by the markdown
engine and render as native superscript links with back-references.

Run locally to test: python scripts/util/linkify_references.py --dry-run
Run for real:        python scripts/util/linkify_references.py --output docs/paper.md
"""

import re
import sys

# Reference registry: (number, first_author_surname)
REFS = [
    (1, "Askell"),
    (2, "Hong"),
    (3, "Lambert"),  # Maps to both Lambert refs contextually
    (4, "Lambert"),  # Book specifically — handled by context
    (5, "Maiya"),
    (6, "Pan"),
    (7, "Shao"),
    (8, "Wang"),
]

# Author -> default ref number (Lambert defaults to 3, book uses 4)
AUTHOR_REF = {
    "Askell": 1,
    "Hong": 2,
    "Lambert": 3,
    "Maiya": 5,
    "Pan": 6,
    "Shao": 7,
    "Wang": 8,
}

AUTHORS_PATTERN = "|".join(AUTHOR_REF.keys())


def fn(num):
    """Generate a markdown footnote reference."""
    return f"[^{num}]"


def linkify(text):
    # Split at references section
    ref_marker = "## 8. References"
    if ref_marker in text:
        body, refs_section = text.split(ref_marker, 1)
    else:
        body, refs_section = text, ""

    # ---- Pass 1: Full author list citations with year ----
    # "Maiya, Bartsch, Lambert, and Hubinger (2025)" → append [^5]
    def replace_full_author(m):
        full = m.group(0)
        for author, num in AUTHOR_REF.items():
            if full.startswith(author):
                return f"{full}{fn(num)}"
        return full

    body = re.sub(
        rf'(?:{AUTHORS_PATTERN})(?:,\s+\w+\.?)*(?:,?\s+(?:and|&)\s+\w+)?\s*\(\d{{4}}\)',
        replace_full_author,
        body
    )

    # ---- Pass 2: "Author et al. (YYYY)" ----
    def replace_et_al(m):
        author = m.group(1)
        rest = m.group(2)
        year = m.group(3)
        num = AUTHOR_REF.get(author)
        if num:
            return f"{author}{rest}({year}){fn(num)}"
        return m.group(0)

    body = re.sub(
        rf'({AUTHORS_PATTERN})(\s+et\s+al\.?\s*)\((\d{{4}})\)',
        replace_et_al,
        body
    )

    # ---- Pass 3: Parenthetical "(... Author et al. YYYY)" ----
    def replace_paren(m):
        inner = m.group(1)
        for author, num in AUTHOR_REF.items():
            if author in inner:
                return f"({inner}){fn(num)}"
        return m.group(0)

    body = re.sub(
        rf'\(([^)]*?(?:{AUTHORS_PATTERN})[^)]*?\d{{4}}[^)]*?)\)',
        replace_paren,
        body
    )

    # ---- Pass 4: "Lambert's" possessive ----
    body = re.sub(
        r"Lambert's(?!\s*\(\d)(?!\[\^)",
        f"Lambert's{fn(3)}",
        body
    )

    # ---- Pass 5: Standalone "Askell et al." without year ----
    body = re.sub(
        r'(Askell\s+et\s+al\.)(?!\[\^)',
        rf'\1{fn(1)}',
        body
    )

    # ---- Build footnote definitions ----
    # These go at the very end of the document
    footnotes = """

[^1]: Askell, A., et al. (Anthropic). "Claude's Character."
[^2]: Hong, J., Lee, N., & Thorne, J. (2024). "ORPO: Monolithic Preference Optimization without Reference Model." arXiv:2403.07691.
[^3]: Lambert, N. (2025). "Opening the Character Training Pipeline." Interconnects.ai.
[^4]: Lambert, N. (2025). *The RLHF Book*, Chapters 17 (Product & Character) and 19 (Character Training).
[^5]: Maiya, A., Bartsch, P., Lambert, N., & Hubinger, E. (2025). "Open Character Training." arXiv:2511.01689.
[^6]: Pan, A., et al. (2025). "What Matters in Data for DPO?" NeurIPS 2025. arXiv:2508.18312.
[^7]: Shao, Y., Li, L., Dai, J., & Qiu, X. (2023). "Character-LLM: A Trainable Agent for Role-Playing." arXiv:2310.10158. EMNLP 2023.
[^8]: Wang, Y., et al. (2024). "Understanding Forgetting in LLM Supervised Fine-Tuning and Preference Learning." arXiv:2410.15483.
"""

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

    # Wrap in paper styling div
    output = '<div class="paper-content" markdown>\n\n'
    output += result
    output += '\n\n</div>\n'

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
