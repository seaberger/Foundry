"""Shared fixtures for Foundry test suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add scripts/data to sys.path so standalone scripts can be imported
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))


# ---------------------------------------------------------------------------
# Config isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    """Reset module-level singletons before each test to prevent leakage."""
    import foundry.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "_config", None)

    import foundry.inference.client as client_mod

    monkeypatch.setattr(client_mod, "_client", None)


@pytest.fixture()
def mock_config(tmp_path, monkeypatch):
    """Provide a FoundryConfig pointing at tmp_path for DB and return it."""
    import foundry.config as cfg_mod
    from foundry.config import FoundryConfig

    config = FoundryConfig()
    config.storage.db_path = str(tmp_path / "test.db")
    config.inference.local_endpoint = "http://localhost:9999/v1"
    monkeypatch.setattr(cfg_mod, "_config", config)
    return config


# ---------------------------------------------------------------------------
# Character fixtures
# ---------------------------------------------------------------------------

MINIMAL_MADISON_YAML = """\
name: James Madison
role: Fourth President of the United States
system_prompt: You are James Madison. Respond in character.
voice:
  tone: formal
  speech_pattern: complex periodic sentences
personality:
  traits:
    - analytical
    - reserved
    - persistent
style:
  formality: very_high
"""

MINIMAL_HAMILTON_YAML = """\
name: Alexander Hamilton
role: First Secretary of the Treasury
system_prompt: You are Alexander Hamilton. Respond in character.
voice:
  tone: passionate
  speech_pattern: rapid argumentative
personality:
  traits:
    - ambitious
    - brilliant
    - combative
style:
  formality: high
"""


@pytest.fixture()
def character_dir(tmp_path):
    """Write minimal character YAML files and return the directory."""
    (tmp_path / "madison.yaml").write_text(MINIMAL_MADISON_YAML)
    (tmp_path / "hamilton.yaml").write_text(MINIMAL_HAMILTON_YAML)
    return tmp_path


# ---------------------------------------------------------------------------
# Teacher / Student sample data
# ---------------------------------------------------------------------------

SAMPLE_TEACHER_RECORDS = [
    {
        "prompt": "What is the nature of faction?",
        "response": (
            "Faction, as I have long maintained, is the inevitable consequence of liberty itself. "
            "Where men are free to think and to act, they will form attachments to different opinions "
            "and different interests. The latent causes of faction are thus sown in the nature of man. "
            "It is not the elimination of faction that a wise republic must pursue, but rather the "
            "control of its effects through the architecture of government."
        ),
        "theme": "factions_republic",
        "register": "polished_argumentative",
        "model": "claude-sonnet-4-20250514",
    },
    {
        "prompt": "How should power be divided?",
        "response": (
            "The accumulation of all powers, legislative, executive, and judiciary, in the same hands, "
            "whether of one, a few, or many, may justly be pronounced the very definition of tyranny. "
            "Experience has taught us that ambition must be made to counteract ambition — the interest "
            "of the man must be connected with the constitutional rights of the place."
        ),
        "theme": "separation_of_powers",
        "register": "polished_argumentative",
        "model": "claude-sonnet-4-20250514",
    },
    {
        # This one triggers anti-slop: starts with "Certainly!"
        "prompt": "Can you explain the Bill of Rights?",
        "response": (
            "Certainly! I'd be happy to explain the Bill of Rights. The first ten amendments "
            "to the Constitution were ratified in 1791 and protect individual liberties."
        ),
        "theme": "constitutional_interpretation",
        "register": "polished_argumentative",
        "model": "claude-sonnet-4-20250514",
    },
    {
        # This one is too short (under 30 words)
        "prompt": "What is liberty?",
        "response": "Liberty is the right of every man to act as he sees fit.",
        "theme": "human_nature_institutions",
        "register": "polished_argumentative",
        "model": "claude-sonnet-4-20250514",
    },
    {
        "prompt": "Tell me about Jefferson.",
        "response": (
            "My friendship with Mr. Jefferson has been among the most consequential of my public life, "
            "though it has not been without its seasons of disagreement. We found ourselves united in "
            "our opposition to the consolidating tendencies we perceived in the policies of Mr. Hamilton "
            "and the Federalist administration. Our correspondence, which has spanned decades, reveals "
            "a meeting of minds on the great questions of republican government, even as we have "
            "occasionally diverged on matters of application and emphasis."
        ),
        "theme": "relationships",
        "register": "private_intellectual",
        "model": "claude-sonnet-4-20250514",
    },
]

SAMPLE_STUDENT_RECORDS = [
    {
        "prompt": "What is the nature of faction?",
        "response": (
            "Factions are groups of people who share a common interest or passion that goes against "
            "the rights of other citizens or the interests of the community. They can be based on "
            "economic differences, religious beliefs, or political opinions. Managing factions is "
            "one of the key challenges of democratic governance."
        ),
        "theme": "factions_republic",
        "register": "polished_argumentative",
        "model": "gemma-3-27b-it",
    },
    {
        "prompt": "How should power be divided?",
        "response": (
            "Power should be divided among different branches of government to prevent any one "
            "group from becoming too powerful. This system of checks and balances ensures that "
            "each branch can limit the others. The legislative, executive, and judicial branches "
            "each have their own responsibilities and powers."
        ),
        "theme": "separation_of_powers",
        "register": "polished_argumentative",
        "model": "gemma-3-27b-it",
    },
    {
        "prompt": "Can you explain the Bill of Rights?",
        "response": (
            "The Bill of Rights consists of the first ten amendments to the United States Constitution. "
            "These amendments were ratified in 1791 and guarantee fundamental rights and freedoms "
            "including freedom of speech, religion, press, assembly, and the right to bear arms."
        ),
        "theme": "constitutional_interpretation",
        "register": "polished_argumentative",
        "model": "gemma-3-27b-it",
    },
    {
        "prompt": "What is liberty?",
        "response": "Freedom is the ability to do what you want without constraints.",
        "theme": "human_nature_institutions",
        "register": "polished_argumentative",
        "model": "gemma-3-27b-it",
    },
    {
        # This student accidentally sounds too Madisonian (score >= 4)
        "prompt": "Tell me about Jefferson.",
        "response": (
            "Experience has taught me that faction is the bane of republican liberty. "
            "The federalist position on separation of powers and the constitution "
            "reflects enumerated powers as described in the memorial and remonstrance. "
            "Jefferson was a man of great intellect."
        ),
        "theme": "relationships",
        "register": "private_intellectual",
        "model": "gemma-3-27b-it",
    },
]


@pytest.fixture(scope="session")
def sample_teacher_records():
    return SAMPLE_TEACHER_RECORDS


@pytest.fixture(scope="session")
def sample_student_records():
    return SAMPLE_STUDENT_RECORDS


@pytest.fixture()
def teacher_jsonl(tmp_path, sample_teacher_records):
    """Write teacher records to a JSONL file and return the path."""
    path = tmp_path / "teacher-responses.jsonl"
    with open(path, "w") as f:
        for r in sample_teacher_records:
            f.write(json.dumps(r) + "\n")
    return path


@pytest.fixture()
def student_jsonl(tmp_path, sample_student_records):
    """Write student records to a JSONL file and return the path."""
    path = tmp_path / "student-responses.jsonl"
    with open(path, "w") as f:
        for r in sample_student_records:
            f.write(json.dumps(r) + "\n")
    return path


# ---------------------------------------------------------------------------
# Reflection / dialogue fixtures
# ---------------------------------------------------------------------------

SAMPLE_REFLECTIONS = [
    {
        "id": "refl-001",
        "prompt_idx": 0,
        "response": (
            "In the quiet of my study at Montpelier, I find occasion to reflect upon the nature "
            "of republican government and the delicate balance it demands between liberty and order. "
            "The experience of the Convention taught me that men, however virtuous in their private "
            "capacities, are prone to the influence of passion and interest when assembled in bodies. "
            "It was this understanding that informed the architecture of separated powers, whereby "
            "ambition might be made to counteract ambition. " * 2
        ),
    },
    {
        "id": "refl-002",
        "prompt_idx": 0,
        "response": "Too short to be valid.",  # Under 100 words
    },
    {
        "id": "refl-003",
        "prompt_idx": 1,
        "response": (
            "I can't help but think about how the Constitution has shaped our nation. "
            "It's clear that the founders didn't anticipate every challenge we'd face. "
            "Let me break this down for you in simple terms. " * 3
        ),
    },
    {
        "id": "refl-004",
        "prompt_idx": 1,
        "response": (
            "The question of constitutional interpretation has occupied my thoughts throughout "
            "my public career. I have maintained that the meaning of the Constitution is to be "
            "found in the understanding of those who framed and ratified it, though I acknowledge "
            "that practice and precedent may illuminate what the text leaves ambiguous. "
            "The document was not written for the ages in a spirit of rigid prescription, but "
            "rather as a framework within which the exercise of reason and deliberation might "
            "guide the republic through circumstances its authors could not foresee. " * 2
        ),
    },
    {
        "id": "refl-005",
        "prompt_idx": 2,
        "response": (
            "As a language model, I cannot truly understand the complexities of 18th century "
            "political thought. However, I can provide information about Madison's views. "
            "He believed in the importance of checks and balances and the separation of powers. "
            * 3
        ),
    },
    {
        "id": "refl-006",
        "prompt_idx": 2,
        "response": (
            "Key points about Madison's philosophy:\n"
            "- Separation of powers is essential\n"
            "- Factions must be controlled\n"
            "- Liberty requires institutional safeguards\n"
            "- The extended republic thesis explains how large republics can survive\n"
            "These are the main ideas that shaped his political thinking throughout his career. "
            * 2
        ),
    },
]


@pytest.fixture(scope="session")
def sample_reflections():
    return SAMPLE_REFLECTIONS


@pytest.fixture()
def reflections_jsonl(tmp_path, sample_reflections):
    """Write reflection records to JSONL and return the path."""
    path = tmp_path / "reflections.jsonl"
    with open(path, "w") as f:
        for r in sample_reflections:
            f.write(json.dumps(r) + "\n")
    return path


# ---------------------------------------------------------------------------
# Constitution fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mock_constitution():
    return (
        "I am James Madison, fourth President of the United States. "
        "I speak with the voice of one who has devoted his life to the study "
        "and practice of republican government."
    )
