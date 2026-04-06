#!/usr/bin/env python3
"""Prepare the Madison constitution for MkDocs publishing.

Adds framing paragraphs and cross-links, strips inline-citation claims
and sources-key header from the annotated working version.
"""

FRAMING = """\
# The Madison Constitution

Prior character training work used trait lists of roughly ten declarations — "you are sarcastic but helpful" in a few hundred words. This document is different: a 5,000-word first-person character specification synthesized from 468,000 words of Madison's own writings and 1.8 million words of scholarly biography, covering nine dimensions of his character across eight voice registers. It is the richest character constitution, to our knowledge, ever used for LLM fine-tuning.

The teacher model (Claude Sonnet 4.6) receives this constitution as a system prompt when generating training data. The evaluation judge uses it as the scoring rubric. The document below is written entirely in Madison's voice — as he might describe himself to someone who needed to understand not just what he believed, but how he thought, argued, and evolved over a fifty-year public career.

For the methodology behind this document, see the [research paper](paper.md). For evaluation results, see [training results](training-results.md).

**Sources:** Ketcham, *James Madison: A Biography*; Feldman, *The Three Lives of James Madison*; Cheney, *James Madison: A Life Reconsidered*; Burstein & Isenberg, *Madison and Jefferson*; Leibiger, *Founding Friendship*; Ellis, *Founding Brothers*; Madison, *Sketch for an Autobiography* (1816); Madison, *Autobiography* (1830).

---

"""


def main():
    with open("config/constitutions/madison-5k.md") as f:
        text = f.read()

    # Remove the "inline citations" claim since the published version
    # doesn't have them (they exist in the annotated working version only)
    text = text.replace(
        "**This annotated version includes inline citations to source material.** "
        "The clean training version (`config/constitutions/madison-10k.md`) strips all citations.\n",
        "",
    )

    # Remove the sources key header (moved into framing)
    text = text.replace("**Sources key:**\n", "")

    # Replace the original title and metadata with the framing
    text = text.replace(
        "# The Madison Constitution\n\n"
        "Character constitution for James Madison fine-tuning. "
        "Written in first person as Madison's self-description. "
        "The teacher model receives this as a system prompt when generating training data.\n\n",
        "",
    )

    output = FRAMING + text

    with open("docs/constitution.md", "w") as f:
        f.write(output)

    print(f"Constitution processed: {len(output)} chars")


if __name__ == "__main__":
    main()
