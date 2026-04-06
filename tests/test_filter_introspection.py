"""Tests for scripts/data/filter_introspection.py quality filters."""

from __future__ import annotations

import json

import pytest
from filter_introspection import (
    check_dialogue_turn_lengths,
    check_reflection_length,
    filter_reflections,
    strip_artifacts,
)

from foundry.press.voice import (
    check_voice_contamination,
)
from foundry.press.voice import (
    expand_contractions as _expand_contractions,
)
from foundry.press.voice import (
    has_bullets as _has_bullets,
)
from foundry.press.voice import (
    has_contraction as _has_contraction,
)
from foundry.press.voice import (
    has_filler as _has_filler,
)

# ---------------------------------------------------------------------------
# TestHasContraction
# ---------------------------------------------------------------------------


class TestHasContraction:
    @pytest.mark.parametrize(
        "text",
        [
            "I can't believe this happened.",
            "He won't attend the convention.",
            "I'm certain of the outcome.",
            "Let's consider the alternatives.",
            "It's a matter of great importance.",
            "He doesn't understand the argument.",
        ],
    )
    def test_modern_contractions_detected(self, text):
        assert _has_contraction(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "'Tis the season of deliberation.",
            "'Twas a long and difficult negotiation.",
            "'Twould be unwise to proceed without caution.",
            "The clock struck three o'clock in the afternoon.",
        ],
    )
    def test_period_appropriate_allowed(self, text):
        assert _has_contraction(text) is False

    def test_mixed_period_appropriate_and_modern(self):
        text = "'Tis clear that he can't be trusted with such authority."
        assert _has_contraction(text) is True

    def test_clean_formal_prose(self):
        text = (
            "The accumulation of all powers in the same hands may justly "
            "be pronounced the very definition of tyranny."
        )
        assert _has_contraction(text) is False

    def test_case_insensitive_upper(self):
        assert _has_contraction("I CAN'T accept this proposition.") is True

    def test_case_insensitive_mixed(self):
        assert _has_contraction("He Won't compromise on this matter.") is True


# ---------------------------------------------------------------------------
# TestHasBullets
# ---------------------------------------------------------------------------


class TestHasBullets:
    def test_dash_bullet(self):
        assert _has_bullets("Some preamble:\n- First item\n- Second item") is True

    def test_asterisk_bullet(self):
        assert _has_bullets("Points:\n* Alpha\n* Beta") is True

    def test_numbered_list(self):
        assert _has_bullets("1. First point\n2. Second point") is True

    def test_inline_hyphen_not_bullet(self):
        text = "This is a self-evident truth that requires no further proof."
        assert _has_bullets(text) is False

    def test_clean_prose(self):
        text = (
            "The republican form of government demands constant vigilance "
            "from its citizens and its officers alike."
        )
        assert _has_bullets(text) is False


# ---------------------------------------------------------------------------
# TestHasFiller
# ---------------------------------------------------------------------------


class TestHasFiller:
    @pytest.mark.parametrize(
        "phrase",
        [
            "Let me break this down for you.",
            "Here's the thing about faction.",
            "Let's unpack this concept together.",
            "Great question about the Constitution.",
            "I'd be happy to explain the matter.",
        ],
    )
    def test_filler_phrases_detected(self, phrase):
        assert _has_filler(phrase) is True

    def test_certainly_exclamation_detected(self):
        assert _has_filler("Certainly! I shall explain the matter.") is True

    def test_absolutely_exclamation_detected(self):
        assert _has_filler("Absolutely! The republic demands it.") is True

    def test_certainly_adverb_not_detected(self):
        text = "He was most certainly aware of the dangers inherent in such a course."
        assert _has_filler(text) is False

    def test_absolutely_adverb_not_detected(self):
        text = "The principle is absolutely essential to the preservation of liberty."
        assert _has_filler(text) is False

    def test_clean_prose(self):
        text = (
            "In the course of human events, it becomes necessary for one people "
            "to dissolve the political bands which have connected them with another."
        )
        assert _has_filler(text) is False


# ---------------------------------------------------------------------------
# TestExpandContractions
# ---------------------------------------------------------------------------


class TestExpandContractions:
    def test_im_expanded(self):
        assert _expand_contractions("I'm certain") == "I am certain"

    def test_cant_expanded(self):
        assert _expand_contractions("We can't proceed") == "We cannot proceed"

    def test_wont_expanded(self):
        assert _expand_contractions("He won't agree") == "He will not agree"

    def test_lets_expanded(self):
        assert _expand_contractions("let's consider") == "let us consider"

    def test_its_expanded(self):
        assert _expand_contractions("it's important") == "it is important"

    def test_ill_expanded(self):
        assert _expand_contractions("I'll address the matter") == "I shall address the matter"

    def test_multiple_contractions_in_one_text(self):
        text = "I'm sure he can't and won't agree, but let's try."
        result = _expand_contractions(text)
        assert "I am" in result
        assert "cannot" in result
        assert "will not" in result
        assert "let us" in result
        assert "'" not in result

    def test_clean_text_unchanged(self):
        text = "The republic demands vigilance from all its citizens."
        assert _expand_contractions(text) == text


# ---------------------------------------------------------------------------
# TestStripArtifacts
# ---------------------------------------------------------------------------


class TestStripArtifacts:
    def test_markdown_header_removed(self):
        text = "## Reflections on Liberty\nThe nature of freedom is complex."
        result = strip_artifacts(text)
        assert "##" not in result
        assert "Reflections on Liberty" not in result
        assert "The nature of freedom is complex." in result

    def test_bold_markers_removed(self):
        result = strip_artifacts("This is **important** text.")
        assert result == "This is important text."

    def test_emphasis_markers_removed(self):
        result = strip_artifacts("This is *emphasized* text.")
        assert result == "This is emphasized text."

    def test_stage_direction_removed(self):
        text = "(He paces thoughtfully) The matter requires deliberation."
        result = strip_artifacts(text)
        assert "(He paces thoughtfully)" not in result
        assert "The matter requires deliberation." in result

    def test_contractions_expanded(self):
        result = strip_artifacts("I can't agree with this proposition.")
        assert "cannot" in result
        assert "can't" not in result

    def test_triple_newlines_collapsed(self):
        text = "First paragraph.\n\n\n\nSecond paragraph."
        result = strip_artifacts(text)
        assert "\n\n\n" not in result
        assert "First paragraph.\n\nSecond paragraph." == result

    def test_combined_cleanup(self):
        text = (
            "## Title\n"
            "This is **bold** and *italic* text.\n"
            "(He pauses dramatically) I can't believe it's true.\n\n\n\n"
            "Final paragraph."
        )
        result = strip_artifacts(text)
        assert "##" not in result
        assert "**" not in result
        assert "*" not in result
        assert "(He pauses dramatically)" not in result
        assert "cannot" in result
        assert "it is" in result
        assert "\n\n\n" not in result


# ---------------------------------------------------------------------------
# TestCheckVoiceContamination
# ---------------------------------------------------------------------------


class TestCheckVoiceContamination:
    def test_ai_speak_detected(self):
        text = "As a language model, I cannot truly understand political philosophy."
        assert check_voice_contamination(text) == "ai_speak"

    def test_contraction_detected(self):
        text = "I can't help but think about the Constitution and its meaning."
        assert check_voice_contamination(text) == "contraction"

    def test_bullets_detected(self):
        text = "The key points are:\n- Separation of powers\n- Checks and balances"
        assert check_voice_contamination(text) == "bullet_points"

    def test_filler_detected(self):
        text = "Great question about the nature of republican government and liberty."
        assert check_voice_contamination(text) == "modern_filler"

    def test_clean_text_returns_none(self):
        text = (
            "The republican form of government, properly constituted, provides "
            "the most durable foundation for the preservation of liberty."
        )
        assert check_voice_contamination(text) is None

    def test_ai_speak_takes_priority_over_contraction(self):
        text = "As a language model, I can't provide personal opinions on this matter."
        assert check_voice_contamination(text) == "ai_speak"


# ---------------------------------------------------------------------------
# TestCheckReflectionLength
# ---------------------------------------------------------------------------


class TestCheckReflectionLength:
    def test_too_short(self):
        text = " ".join(["word"] * 50)
        assert check_reflection_length(text) == "too_short"

    def test_too_long(self):
        text = " ".join(["word"] * 2500)
        assert check_reflection_length(text) == "too_long"

    def test_acceptable_length(self):
        text = " ".join(["word"] * 500)
        assert check_reflection_length(text) is None

    def test_boundary_exactly_100_words(self):
        text = " ".join(["word"] * 100)
        assert check_reflection_length(text) is None


# ---------------------------------------------------------------------------
# TestCheckDialogueTurnLengths
# ---------------------------------------------------------------------------


class TestCheckDialogueTurnLengths:
    def test_all_turns_ok(self):
        turns = [
            {"content": " ".join(["word"] * 50)},
            {"content": " ".join(["word"] * 100)},
        ]
        assert check_dialogue_turn_lengths(turns) is None

    def test_turn_too_short(self):
        turns = [
            {"content": " ".join(["word"] * 10)},
            {"content": " ".join(["word"] * 100)},
        ]
        assert check_dialogue_turn_lengths(turns) == "turn_too_short"

    def test_turn_too_long(self):
        turns = [
            {"content": " ".join(["word"] * 100)},
            {"content": " ".join(["word"] * 900)},
        ]
        assert check_dialogue_turn_lengths(turns) == "turn_too_long"

    def test_first_bad_turn_triggers(self):
        turns = [
            {"content": " ".join(["word"] * 5)},
            {"content": " ".join(["word"] * 900)},
        ]
        assert check_dialogue_turn_lengths(turns) == "turn_too_short"


# ---------------------------------------------------------------------------
# TestFilterReflections
# ---------------------------------------------------------------------------


class TestFilterReflections:
    def test_end_to_end_filters_bad_records(self, reflections_jsonl, tmp_path):
        output_path = tmp_path / "filtered.jsonl"
        stats = filter_reflections(reflections_jsonl, output_path, similarity_threshold=0.95)

        assert stats["total_in"] == 6
        assert stats["total_out"] < stats["total_in"]

        output_records = []
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    output_records.append(json.loads(line))

        assert len(output_records) == stats["total_out"]

    def test_artifacts_stripped_in_output(self, reflections_jsonl, tmp_path):
        output_path = tmp_path / "filtered.jsonl"
        filter_reflections(reflections_jsonl, output_path, similarity_threshold=0.95)

        with open(output_path) as f:
            for line in f:
                record = json.loads(line.strip())
                assert "**" not in record["response"]
                assert not record["response"].startswith("#")

    def test_stats_dict_has_expected_keys(self, reflections_jsonl, tmp_path):
        output_path = tmp_path / "filtered.jsonl"
        stats = filter_reflections(reflections_jsonl, output_path, similarity_threshold=0.95)

        assert "total_in" in stats
        assert "total_out" in stats
        assert any(k.startswith("rejected_voice_") for k in stats)
        assert any(k.startswith("rejected_length_") for k in stats)
