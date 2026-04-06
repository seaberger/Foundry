"""Tests for foundry.press.format_dpo — anti-slop, madison scoring, word count, and DPO pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from foundry.press.format_dpo import (
    MADISON_MARKERS,
    MAX_RESPONSE_WORDS,
    MIN_RESPONSE_WORDS,
    TEACHER_ANTI_SLOP,
    has_anti_slop,
    load_jsonl,
    madison_score,
    main,
    word_count,
)


# ======================================================================
# TestHasAntiSlop
# ======================================================================


class TestHasAntiSlop:
    def test_clean_historical_text_returns_empty(self):
        text = (
            "The accumulation of all powers in the same hands may justly be "
            "pronounced the very definition of tyranny."
        )
        assert has_anti_slop(text) == []

    # --- Anchored patterns: must appear at start of text ---

    def test_certainly_at_start(self):
        result = has_anti_slop("Certainly! Let me explain the matter.")
        assert len(result) == 1
        assert r"(?i)^certainly!" in result

    def test_certainly_mid_sentence_does_not_trigger(self):
        text = "He was most certainly correct in his assessment of the republic."
        assert has_anti_slop(text) == []

    def test_great_question_at_start_with_exclamation(self):
        result = has_anti_slop("Great question! The Constitution provides for this.")
        assert r"(?i)^great question!" in result

    def test_great_question_without_exclamation_does_not_trigger(self):
        text = "That is a great question to consider."
        assert has_anti_slop(text) == []

    def test_great_question_as_noun_phrase_mid_sentence(self):
        text = "The great question of representation was debated at length."
        assert has_anti_slop(text) == []

    def test_id_be_happy_to_at_start(self):
        result = has_anti_slop("I'd be happy to elaborate on the matter.")
        assert r"(?i)^i'd be happy to" in result

    def test_absolutely_at_start(self):
        result = has_anti_slop("Absolutely! The framers intended this.")
        assert r"(?i)^absolutely!" in result

    # --- Non-anchored patterns: trigger anywhere ---

    def test_let_me_break_this_down(self):
        text = "Well, let me break this down for the assembly."
        result = has_anti_slop(text)
        assert r"(?i)let me break this down" in result

    def test_lets_unpack(self):
        text = "Now, let's unpack the implications of this clause."
        result = has_anti_slop(text)
        assert r"(?i)let's unpack" in result

    def test_lets_dive_in(self):
        text = "So let's dive in to the details of the proposal."
        result = has_anti_slop(text)
        assert r"(?i)let's dive in" in result

    def test_as_an_ai(self):
        text = "As an AI, I must disclose my limitations."
        result = has_anti_slop(text)
        assert r"(?i)as an ai" in result

    def test_as_a_language_model(self):
        text = "Speaking as a language model, I cannot truly know."
        result = has_anti_slop(text)
        assert r"(?i)as a language model" in result

    def test_as_a_founding_father(self):
        text = "As a founding father, I helped shape the republic."
        result = has_anti_slop(text)
        assert r"(?i)as a founding father" in result

    # --- Word-boundary patterns ---

    @pytest.mark.parametrize(
        "word,pattern",
        [
            ("delve", r"(?i)\bdelve\b"),
            ("tapestry", r"(?i)\btapestry\b"),
            ("ministrations", r"(?i)\bministrations\b"),
        ],
    )
    def test_word_boundary_pattern(self, word, pattern):
        text = f"We must {word} into the matter at hand with great care."
        result = has_anti_slop(text)
        assert pattern in result

    def test_left_wing(self):
        text = "The left-wing faction opposed the measure."
        result = has_anti_slop(text)
        assert r"(?i)\bleft.wing\b" in result

    def test_right_wing(self):
        text = "The right-wing advocates pressed their case."
        result = has_anti_slop(text)
        assert r"(?i)\bright.wing\b" in result

    # --- Case insensitivity ---

    def test_case_insensitive_detection(self):
        result = has_anti_slop("CERTAINLY! This is important.")
        assert len(result) >= 1
        assert r"(?i)^certainly!" in result

    # --- Multiple patterns in same text ---

    def test_multiple_patterns_returns_all(self):
        text = "Certainly! Let me break this down. We must delve into this tapestry of ideas."
        result = has_anti_slop(text)
        assert r"(?i)^certainly!" in result
        assert r"(?i)let me break this down" in result
        assert r"(?i)\bdelve\b" in result
        assert r"(?i)\btapestry\b" in result
        assert len(result) == 4


# ======================================================================
# TestMadisonScore
# ======================================================================


class TestMadisonScore:
    def test_generic_modern_text_scores_zero(self):
        text = "The weather today is pleasant and the markets are stable."
        assert madison_score(text) == 0

    @pytest.mark.parametrize(
        "phrase",
        [
            "Experience has taught us the value of prudence.",
            "The faction opposed the measure in committee.",
            "Republican liberty demands perpetual vigilance.",
            "The federalist papers remain a foundation of our understanding.",
            "Separation of powers is the cornerstone of the system.",
            "Ambition must be made to counteract ambition in government.",
            "If men were angels, no government would be necessary.",
            "The constitution provides the framework for governance.",
            "Enumerated powers limit the scope of federal authority.",
            "The memorial and remonstrance argued against religious assessments.",
        ],
    )
    def test_individual_marker_scores_one(self, phrase):
        assert madison_score(phrase) >= 1

    def test_multiple_markers_accumulate(self):
        text = (
            "Experience has taught us that faction is inevitable. "
            "The constitution provides for separation of powers. "
            "If men were angels, no government would be necessary."
        )
        assert madison_score(text) == 5

    def test_case_insensitive(self):
        assert madison_score("FACTION is the bane of government.") >= 1

    def test_federal_alone_does_not_match_federalist(self):
        text = "The federal government oversees interstate commerce."
        # "federal" should not match the "federalist" pattern
        score_with_only_federal = madison_score(text)
        # Only "the constitution" could match here (via "federal government" — no).
        # Actually neither pattern matches "federal" alone.
        # Verify federalist specifically does not match
        import re

        assert re.search(r"(?i)federalist", text) is None
        # The text might match "the constitution" — let's be precise
        assert re.search(r"(?i)the constitution", text) is None
        # So score should be 0
        assert score_with_only_federal == 0

    def test_all_markers_scores_ten(self):
        text = (
            "Experience has taught us about faction and republican liberty. "
            "The federalist understood separation of powers. "
            "Ambition must be made to counteract ambition. "
            "If men were angels, we would not need the constitution. "
            "Enumerated powers and the memorial and remonstrance shaped our republic."
        )
        assert madison_score(text) == 10

    def test_repeated_marker_counts_once(self):
        text = "Faction breeds faction, and yet more faction."
        assert madison_score(text) == 1


# ======================================================================
# TestWordCount
# ======================================================================


class TestWordCount:
    def test_empty_string(self):
        # "".split() returns [] which has len 0
        assert word_count("") == 0

    def test_single_word(self):
        assert word_count("Liberty") == 1

    def test_normal_sentence(self):
        assert word_count("The quick brown fox jumps over the lazy dog") == 9

    def test_multiple_whitespace(self):
        assert word_count("  The   republic   endures  ") == 3


# ======================================================================
# TestMainIntegration
# ======================================================================


def _make_teacher_record(prompt, response, theme="test_theme", model="test-model"):
    return {
        "prompt": prompt,
        "response": response,
        "theme": theme,
        "register": "polished_argumentative",
        "model": model,
    }


def _make_student_record(prompt, response, theme="test_theme", model="student-model"):
    return {
        "prompt": prompt,
        "response": response,
        "theme": theme,
        "register": "polished_argumentative",
        "model": model,
    }


def _filler(n=50):
    """Return a string of n words of generic filler text."""
    words = "the republic requires vigilance and prudence in all matters of governance and law".split()
    result = []
    for i in range(n):
        result.append(words[i % len(words)])
    return " ".join(result)


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestMainIntegration:
    def test_basic_pair_creation(self, tmp_path):
        teacher_path = tmp_path / "teacher.jsonl"
        student_path = tmp_path / "student.jsonl"
        output_path = tmp_path / "output.jsonl"

        prompt = "What is the nature of faction?"
        teacher_resp = _filler(60)
        student_resp = _filler(50)

        _write_jsonl(teacher_path, [_make_teacher_record(prompt, teacher_resp)])
        _write_jsonl(student_path, [_make_student_record(prompt, student_resp)])

        with patch(
            "sys.argv",
            ["format_dpo", "--teacher", str(teacher_path), "--student", str(student_path), "--output", str(output_path)],
        ):
            main()

        assert output_path.exists()
        pairs = load_jsonl(output_path)
        assert len(pairs) == 1

    def test_anti_slop_teacher_filtered(self, tmp_path):
        teacher_path = tmp_path / "teacher.jsonl"
        student_path = tmp_path / "student.jsonl"
        output_path = tmp_path / "output.jsonl"

        prompt = "Explain the Bill of Rights."
        slop_response = "Certainly! " + _filler(50)
        student_resp = _filler(50)

        _write_jsonl(teacher_path, [_make_teacher_record(prompt, slop_response)])
        _write_jsonl(student_path, [_make_student_record(prompt, student_resp)])

        with patch(
            "sys.argv",
            ["format_dpo", "--teacher", str(teacher_path), "--student", str(student_path), "--output", str(output_path)],
        ):
            main()

        pairs = load_jsonl(output_path)
        assert len(pairs) == 0

    def test_short_response_filtered(self, tmp_path):
        teacher_path = tmp_path / "teacher.jsonl"
        student_path = tmp_path / "student.jsonl"
        output_path = tmp_path / "output.jsonl"

        prompt = "What is liberty?"
        short_response = "Liberty is freedom."  # 3 words, well under MIN_RESPONSE_WORDS
        student_resp = _filler(50)

        _write_jsonl(teacher_path, [_make_teacher_record(prompt, short_response)])
        _write_jsonl(student_path, [_make_student_record(prompt, student_resp)])

        with patch(
            "sys.argv",
            ["format_dpo", "--teacher", str(teacher_path), "--student", str(student_path), "--output", str(output_path)],
        ):
            main()

        pairs = load_jsonl(output_path)
        assert len(pairs) == 0

    def test_student_too_madison_filtered(self, tmp_path):
        teacher_path = tmp_path / "teacher.jsonl"
        student_path = tmp_path / "student.jsonl"
        output_path = tmp_path / "output.jsonl"

        prompt = "Tell me about governance."
        teacher_resp = _filler(60)
        # Student text with 4+ Madison markers triggers filter
        madison_student = (
            "Experience has taught us that faction threatens republican liberty. "
            "The federalist understanding of separation of powers remains essential. "
            "If men were angels, the constitution would be unnecessary. "
            "Enumerated powers and the memorial and remonstrance shaped everything. "
            + _filler(20)
        )
        assert madison_score(madison_student) >= 4

        _write_jsonl(teacher_path, [_make_teacher_record(prompt, teacher_resp)])
        _write_jsonl(student_path, [_make_student_record(prompt, madison_student)])

        with patch(
            "sys.argv",
            ["format_dpo", "--teacher", str(teacher_path), "--student", str(student_path), "--output", str(output_path)],
        ):
            main()

        pairs = load_jsonl(output_path)
        assert len(pairs) == 0

    def test_output_dpo_format(self, tmp_path):
        teacher_path = tmp_path / "teacher.jsonl"
        student_path = tmp_path / "student.jsonl"
        output_path = tmp_path / "output.jsonl"

        prompt = "How should power be divided?"
        teacher_resp = _filler(60)
        student_resp = _filler(50)

        _write_jsonl(teacher_path, [_make_teacher_record(prompt, teacher_resp, theme="governance")])
        _write_jsonl(student_path, [_make_student_record(prompt, student_resp)])

        with patch(
            "sys.argv",
            ["format_dpo", "--teacher", str(teacher_path), "--student", str(student_path), "--output", str(output_path)],
        ):
            main()

        pairs = load_jsonl(output_path)
        assert len(pairs) == 1
        pair = pairs[0]

        # chosen structure
        assert len(pair["chosen"]) == 2
        assert pair["chosen"][0]["role"] == "user"
        assert pair["chosen"][0]["content"] == prompt
        assert pair["chosen"][1]["role"] == "assistant"
        assert pair["chosen"][1]["content"] == teacher_resp

        # rejected structure
        assert len(pair["rejected"]) == 2
        assert pair["rejected"][0]["role"] == "user"
        assert pair["rejected"][0]["content"] == prompt
        assert pair["rejected"][1]["role"] == "assistant"
        assert pair["rejected"][1]["content"] == student_resp

        # metadata
        meta = pair["metadata"]
        assert "theme" in meta
        assert "teacher_model" in meta
        assert "student_model" in meta
        assert "teacher_words" in meta
        assert "student_words" in meta
        assert "student_madison_score" in meta
        assert meta["theme"] == "governance"

    def test_conftest_fixtures_integration(self, teacher_jsonl, student_jsonl, tmp_path):
        """Use conftest fixtures to run main and verify expected filtering behavior."""
        output_path = tmp_path / "output.jsonl"

        with patch(
            "sys.argv",
            [
                "format_dpo",
                "--teacher", str(teacher_jsonl),
                "--student", str(student_jsonl),
                "--output", str(output_path),
            ],
        ):
            main()

        pairs = load_jsonl(output_path)
        prompts_in_output = {p["chosen"][0]["content"] for p in pairs}

        # "Certainly!" teacher should be filtered (anti-slop)
        assert "Can you explain the Bill of Rights?" not in prompts_in_output

        # "What is liberty?" should be filtered (too short: both teacher and student under 30 words)
        assert "What is liberty?" not in prompts_in_output

        # "Tell me about Jefferson." student is too Madisonian (score >= 4)
        assert "Tell me about Jefferson." not in prompts_in_output

        # The two good pairs should survive
        assert "What is the nature of faction?" in prompts_in_output
        assert "How should power be divided?" in prompts_in_output
        assert len(pairs) == 2
