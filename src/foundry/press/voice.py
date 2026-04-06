"""Canonical voice contamination detection for Madison character.

Centralizes the contraction, bullet, filler, and AI-speak patterns that were
previously duplicated (and diverged) across format_dpo.py, assemble_v4_dataset.py,
assemble_v5_dataset.py, and filter_introspection.py.

Two modes:
  - Boolean detection (for filtering): check_voice_contamination(), has_contraction(), etc.
  - Count-based scoring (for dataset assembly): voice_score()
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Contraction detection
# ---------------------------------------------------------------------------

# Modern contractions that break Madison's voice.
_CONTRACTIONS = [
    r"can't", r"won't", r"isn't", r"aren't", r"doesn't", r"didn't",
    r"haven't", r"hasn't", r"wouldn't", r"shouldn't", r"couldn't",
    r"i'm", r"i've", r"i'd", r"i'll",
    r"we're", r"we've", r"they're",
    r"there's", r"that's", r"what's", r"who's",
    r"let's", r"here's", r"it's",
]

# Period-appropriate contractions that are ALLOWED in Madison's voice.
_PERIOD_APPROPRIATE = {r"'tis", r"'twas", r"'twould", r"o'clock"}

_CONTRACTION_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(_CONTRACTIONS) + r")(?!\w)",
    re.IGNORECASE,
)

_PERIOD_APPROPRIATE_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(p) for p in _PERIOD_APPROPRIATE) + r")(?!\w)",
    re.IGNORECASE,
)

# Broader contraction regex for count-based scoring (catches all pronoun+'x patterns).
# Does NOT exclude period-appropriate forms — use for scoring, not boolean filtering.
CONTRACTION_COUNT_RE = re.compile(
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


def has_contraction(text: str, allow_period_appropriate: bool = True) -> bool:
    """Return True if text contains a modern contraction.

    Args:
        allow_period_appropriate: If True (default), 'tis/'twas/'twould/o'clock
            are excluded. Set to False for stricter checking.
    """
    if allow_period_appropriate:
        cleaned = _PERIOD_APPROPRIATE_PATTERN.sub("", text)
        return bool(_CONTRACTION_PATTERN.search(cleaned))
    return bool(_CONTRACTION_PATTERN.search(text))


# ---------------------------------------------------------------------------
# Bullet detection
# ---------------------------------------------------------------------------

BULLET_RE = re.compile(r"^\s*(?:[-*•]\s|\d+\.\s)", re.MULTILINE)


def has_bullets(text: str) -> bool:
    return bool(BULLET_RE.search(text))


# ---------------------------------------------------------------------------
# Filler / slop detection
# ---------------------------------------------------------------------------

_FILLER_PHRASES = [
    r"let me break this down",
    r"let me break that down",
    r"here's the thing",
    r"here's what i think",
    r"here are some key",
    r"let's unpack",
    r"let me unpack",
    r"great question",
    r"that's a great question",
    r"that's an excellent question",
    r"that's a fascinating question",
    r"i'd be happy to",
    r"i'd love to",
    r"in today's world",
    r"it's important to note that",
    r"it's worth noting that",
    r"key takeaway",
    r"game changer",
    r"deep dive",
    r"paradigm shift",
    r"synergy",
    r"ecosystem",
    r"stakeholder",
    r"actionable",
    r"impactful",
    r"robust solution",
]

_FILLER_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(_FILLER_PHRASES) + r")(?!\w)",
    re.IGNORECASE,
)

# "Certainly!" and "Absolutely!" as exclamations (not adverbial use).
_EXCLAMATION_PATTERN = re.compile(
    r"\b(?:certainly|absolutely)\s*!",
    re.IGNORECASE,
)

# "Absolutely." at sentence start (AI exclamation, not adverb)
ABSOLUTELY_SLOP_RE = re.compile(r"(?:^|\.\s+)Absolutely[.,!]", re.MULTILINE)

# Compiled filler for count-based scoring
MODERN_FILLER_RE = re.compile(
    "|".join(re.escape(phrase) for phrase in _FILLER_PHRASES),
    re.IGNORECASE,
)


def has_filler(text: str) -> bool:
    return bool(_FILLER_PATTERN.search(text)) or bool(_EXCLAMATION_PATTERN.search(text))


# ---------------------------------------------------------------------------
# AI-speak / character break detection
# ---------------------------------------------------------------------------

_AI_SPEAK_PATTERN = re.compile(
    r"\b(?:hallucin|bias amplif|pattern recognition|trained on|dataset|"
    r"language model|neural network|AI system|artificial intelligence|"
    r"machine learning|fine.tun|pre.train|"
    r"large language|LLM|GPT|transformer|"
    r"computational|"
    r"I am an AI|I am a model|as an AI|as a language model|"
    r"my training data|my training|"
    r"I was programmed|I was designed|I was built|I was created)\b",
    re.IGNORECASE,
)


def has_ai_speak(text: str) -> bool:
    return bool(_AI_SPEAK_PATTERN.search(text))


# ---------------------------------------------------------------------------
# CJK detection (for Qwen multilingual leakage)
# ---------------------------------------------------------------------------

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


# ---------------------------------------------------------------------------
# Contraction expansion
# ---------------------------------------------------------------------------

_EXPANSIONS = [
    (r"\bI'm\b", "I am"),
    (r"\bI've\b", "I have"),
    (r"\bI'd\b", "I would"),
    (r"\bI'll\b", "I shall"),
    (r"\bwe're\b", "we are"),
    (r"\bwe've\b", "we have"),
    (r"\bthey're\b", "they are"),
    (r"\bcan't\b", "cannot"),
    (r"\bwon't\b", "will not"),
    (r"\bdon't\b", "do not"),
    (r"\bdoesn't\b", "does not"),
    (r"\bdidn't\b", "did not"),
    (r"\bisn't\b", "is not"),
    (r"\baren't\b", "are not"),
    (r"\bwasn't\b", "was not"),
    (r"\bweren't\b", "were not"),
    (r"\bhasn't\b", "has not"),
    (r"\bhaven't\b", "have not"),
    (r"\bhadn't\b", "had not"),
    (r"\bwouldn't\b", "would not"),
    (r"\bshouldn't\b", "should not"),
    (r"\bcouldn't\b", "could not"),
    (r"\bthere's\b", "there is"),
    (r"\bthat's\b", "that is"),
    (r"\bwhat's\b", "what is"),
    (r"\bwho's\b", "who is"),
    (r"\bhere's\b", "here is"),
    (r"\blet's\b", "let us"),
    (r"\bit's\b", "it is"),
]


def expand_contractions(text: str) -> str:
    """Expand modern contractions to formal Madisonian forms."""
    for pattern, replacement in _EXPANSIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def check_voice_contamination(text: str) -> str | None:
    """Return a rejection reason string, or None if clean.

    Checks in priority order: ai_speak > contraction > bullet > filler.
    """
    if has_ai_speak(text):
        return "ai_speak"
    if has_contraction(text):
        return "contraction"
    if has_bullets(text):
        return "bullet_points"
    if has_filler(text):
        return "modern_filler"
    return None


def voice_score(text: str, check_cjk: bool = False) -> dict:
    """Score how 'modern/assistant-like' a response is. Higher = more contaminated.

    Uses count-based detection for granular scoring.

    Args:
        check_cjk: If True, include CJK character count in the score.
    """
    contractions = len(CONTRACTION_COUNT_RE.findall(text))
    bullets = len(BULLET_RE.findall(text))
    filler = len(MODERN_FILLER_RE.findall(text)) + len(ABSOLUTELY_SLOP_RE.findall(text))
    total = contractions + bullets + filler

    result = {
        "contractions": contractions,
        "bullets": bullets,
        "filler": filler,
        "total": total,
    }

    if check_cjk:
        cjk = len(CJK_RE.findall(text))
        result["cjk"] = cjk
        result["total"] = total + cjk

    return result


def is_contaminated(text: str) -> tuple[bool, dict]:
    """Check if a response has any voice contamination. Returns (bool, score_dict)."""
    score = voice_score(text)
    return score["total"] > 0, score
