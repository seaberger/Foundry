"""Convert author-year citations to numbered hyperlinks for MkDocs rendering.

Run locally to test: python scripts/util/linkify_references.py --dry-run
Run for real:        python scripts/util/linkify_references.py --output docs/paper.md
"""

import re
import sys

# Reference registry: (number, anchor_id, first_author_surname, reference_list_prefix)
REFS = [
    (1, "askell",           "Askell",  "Askell, A."),
    (2, "hong",             "Hong",    "Hong, J."),
    (3, "lambert-pipeline", "Lambert", "Lambert, N. (2025). \"Opening"),
    (4, "lambert-book",     "Lambert", "Lambert, N. (2025). *The RLHF"),
    (5, "maiya",            "Maiya",   "Maiya, A."),
    (6, "pan",              "Pan",     "Pan, A."),
    (7, "shao",             "Shao",    "Shao, Y."),
    (8, "wang",             "Wang",    "Wang, Y."),
]

# Build author -> refs lookup (Lambert maps to two refs)
AUTHOR_REFS = {}
for num, anchor, author, _ in REFS:
    AUTHOR_REFS.setdefault(author, []).append((num, anchor))


def cite_tag(num, anchor):
    return f'<a href="#ref-{anchor}" class="cite-link"><sup>[{num}]</sup></a>'


def linkify(text):
    # Split text at references section — only linkify body, not the ref list itself
    ref_marker = "## 8. References"
    if ref_marker in text:
        body, refs_section = text.split(ref_marker, 1)
    else:
        body, refs_section = text, ""

    # ---- Pass 1: Anchor the reference list entries ----
    for num, anchor, _, prefix in REFS:
        old = f"- {prefix}"
        new = f'- <span id="ref-{anchor}">**[{num}]**</span> {prefix}'
        refs_section = refs_section.replace(old, new, 1)

    # Work on body only from here
    text = body

    # ---- Pass 2: Full author list citations ----
    # "Maiya, Bartsch, Lambert, and Hubinger (2025)" or "Shao, Li, Dai, and Qiu (2023)"
    def replace_full_author_list(m):
        full = m.group(0)
        for author, ref_list in AUTHOR_REFS.items():
            if full.startswith(author):
                # Use first matching ref for this author
                num, anchor = ref_list[0]
                return f'{full}{cite_tag(num, anchor)}'
        return full

    text = re.sub(
        r'(?:Askell|Hong|Lambert|Maiya|Pan|Shao|Wang)(?:,\s+\w+\.?)*(?:,?\s+(?:and|&)\s+\w+)?\s*\(\d{4}\)',
        replace_full_author_list,
        text
    )

    # ---- Pass 3: "Author et al. (YYYY)" inline citations ----
    def replace_et_al_inline(m):
        author = m.group(1)
        rest = m.group(2)
        year = m.group(3)
        refs = AUTHOR_REFS.get(author, [])
        if refs:
            num, anchor = refs[0]
            return f'{author}{rest}({year}){cite_tag(num, anchor)}'
        return m.group(0)

    text = re.sub(
        r'(Askell|Hong|Lambert|Maiya|Pan|Shao|Wang)(\s+et\s+al\.?\s*)\((\d{4})\)',
        replace_et_al_inline,
        text
    )

    # ---- Pass 4: "(Author et al. YYYY)" parenthetical ----
    def replace_paren_cite(m):
        inner = m.group(1)
        for author, ref_list in AUTHOR_REFS.items():
            if author in inner:
                num, anchor = ref_list[0]
                return f'({inner}){cite_tag(num, anchor)}'
        return m.group(0)

    text = re.sub(
        r'\(([^)]*?(?:Askell|Hong|Maiya|Pan|Shao|Wang)[^)]*?\d{4}[^)]*?)\)',
        replace_paren_cite,
        text
    )

    # ---- Pass 5: "Lambert's" possessive → link to pipeline ref ----
    # Only add link if not already linked (no <a> right after)
    text = re.sub(
        r"Lambert's(?!\s*<a)(?!\s*\()",
        f"Lambert's{cite_tag(3, 'lambert-pipeline')}",
        text
    )

    # ---- Pass 6: Standalone "Askell et al." without year ----
    text = re.sub(
        r'(Askell\s+et\s+al\.)(?!.*?<a href)',
        rf'\1{cite_tag(1, "askell")}',
        text
    )

    # ---- Pass 7: "Dettmers et al., 2023" — not in our ref list, skip ----
    # (leave as-is since it's not in the reference section)

    # Rejoin body with references section
    if refs_section:
        text = text + ref_marker + refs_section

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
        # Show what changed
        import difflib
        orig_lines = text.splitlines()
        new_lines = result.splitlines()
        diff = list(difflib.unified_diff(orig_lines, new_lines, lineterm='', n=0))
        changes = [l for l in diff if l.startswith('+') and not l.startswith('+++')]
        print(f"Changes: {len(changes)} lines modified")
        for line in changes[:30]:
            print(f"  {line[:120]}")
        if len(changes) > 30:
            print(f"  ... and {len(changes) - 30} more")
    elif output_path:
        with open(output_path, "w") as f:
            f.write(output)
        print(f"Written to {output_path} ({len(output)} chars)")
    else:
        print(output)


if __name__ == "__main__":
    main()
