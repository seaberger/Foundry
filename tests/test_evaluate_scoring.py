"""Tests for pure scoring/extraction logic in foundry.press.evaluate."""

from __future__ import annotations

import json
import textwrap

import pytest

from foundry.press.evaluate import (
    COMPONENT_WEIGHTS,
    _repair_json,
    compute_weighted_overall,
    extract_json,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_COMPONENTS = [
    "voice_authenticity",
    "rhetorical_pattern",
    "historical_accuracy",
    "position_fidelity",
    "character_integrity",
]


def _make_scores(
    voice: float = 5,
    rhetorical: float = 5,
    historical: float = 5,
    position: float = 5,
    integrity: float = 5,
) -> dict:
    """Build a well-formed scores dict with the given per-component values."""
    return {
        "voice_authenticity": {"score": voice, "justification": "ok"},
        "rhetorical_pattern": {"score": rhetorical, "justification": "ok"},
        "historical_accuracy": {"score": historical, "justification": "ok"},
        "position_fidelity": {"score": position, "justification": "ok"},
        "character_integrity": {"score": integrity, "justification": "ok"},
    }


# =========================================================================
# TestComponentWeights
# =========================================================================


class TestComponentWeights:
    """Validate the COMPONENT_WEIGHTS constant."""

    def test_all_expected_components_present(self):
        assert set(COMPONENT_WEIGHTS.keys()) == set(EXPECTED_COMPONENTS)

    def test_weights_sum_to_one(self):
        assert sum(COMPONENT_WEIGHTS.values()) == pytest.approx(1.0)


# =========================================================================
# TestComputeWeightedOverall
# =========================================================================


class TestComputeWeightedOverall:
    """Validate compute_weighted_overall against known inputs."""

    def test_perfect_tens(self):
        scores = _make_scores(10, 10, 10, 10, 10)
        assert compute_weighted_overall(scores) == pytest.approx(10.0)

    def test_all_fives(self):
        scores = _make_scores(5, 5, 5, 5, 5)
        assert compute_weighted_overall(scores) == pytest.approx(5.0)

    def test_manual_weighted_calculation(self):
        # 10*0.25 + 8*0.20 + 6*0.20 + 4*0.20 + 2*0.15 = 2.5+1.6+1.2+0.8+0.3 = 6.4
        scores = _make_scores(voice=10, rhetorical=8, historical=6, position=4, integrity=2)
        assert compute_weighted_overall(scores) == pytest.approx(6.4)

    def test_missing_component_returns_none(self):
        scores = _make_scores()
        del scores["character_integrity"]
        assert compute_weighted_overall(scores) is None

    def test_component_not_a_dict_returns_none(self):
        scores = _make_scores()
        scores["voice_authenticity"] = 8  # plain int instead of dict
        assert compute_weighted_overall(scores) is None

    def test_component_missing_score_key_returns_none(self):
        scores = _make_scores()
        scores["rhetorical_pattern"] = {"justification": "good but no score key"}
        assert compute_weighted_overall(scores) is None

    def test_score_zero_returns_none(self):
        scores = _make_scores()
        scores["historical_accuracy"]["score"] = 0
        assert compute_weighted_overall(scores) is None

    def test_score_is_string_returns_none(self):
        scores = _make_scores()
        scores["position_fidelity"]["score"] = "seven"
        assert compute_weighted_overall(scores) is None

    def test_result_rounded_to_two_decimals(self):
        # Construct scores that produce a value needing rounding.
        # voice=7, rhetorical=3, historical=9, position=1, integrity=6
        # 7*0.25 + 3*0.20 + 9*0.20 + 1*0.20 + 6*0.15
        # = 1.75 + 0.60 + 1.80 + 0.20 + 0.90 = 5.25
        scores = _make_scores(voice=7, rhetorical=3, historical=9, position=1, integrity=6)
        result = compute_weighted_overall(scores)
        assert result == pytest.approx(5.25)
        # Verify it is actually rounded to at most 2 decimal places.
        assert result == round(result, 2)


# =========================================================================
# TestRepairJson
# =========================================================================


class TestRepairJson:
    """Validate _repair_json comma-insertion heuristics."""

    def test_missing_comma_after_closing_brace(self):
        broken = '{"a": "x"}\n"b"'
        repaired = _repair_json(broken)
        assert '}\n' not in repaired or ',\n' in repaired
        # The regex targets `"}\n  "` patterns -- verify comma is inserted.
        assert '"}\n' not in repaired or '"},\n' in repaired

    def test_missing_comma_after_closing_bracket(self):
        broken = '["x"]\n"b"'
        repaired = _repair_json(broken)
        assert '],\n' in repaired

    def test_valid_json_passes_through_unchanged(self):
        valid = '{"a": 1, "b": 2}'
        assert _repair_json(valid) == valid


# =========================================================================
# TestExtractJson
# =========================================================================


class TestExtractJson:
    """Validate extract_json against a variety of inputs."""

    def test_clean_json_string(self):
        raw = '{"key": "value", "num": 42}'
        result = extract_json(raw)
        assert result == {"key": "value", "num": 42}

    def test_json_in_json_code_block(self):
        text = 'Some preamble\n```json\n{"a": 1}\n```\nsome epilogue'
        result = extract_json(text)
        assert result == {"a": 1}

    def test_json_in_generic_code_block(self):
        text = 'Preamble\n```\n{"b": 2}\n```\nEpilogue'
        result = extract_json(text)
        assert result == {"b": 2}

    def test_json_embedded_in_prose(self):
        text = 'The result is {"score": 9.5} and that concludes the analysis.'
        result = extract_json(text)
        assert result == {"score": 9.5}

    def test_nested_json_objects(self):
        nested = {"outer": {"inner": {"deep": True}}, "list": [1, 2]}
        text = f"Here is the data: {json.dumps(nested)} -- end"
        result = extract_json(text)
        assert result == nested

    def test_invalid_text_returns_none(self):
        assert extract_json("no json here at all") is None

    def test_empty_string_returns_none(self):
        assert extract_json("") is None

    def test_malformed_partial_json_returns_none(self):
        assert extract_json('{"key": "value"') is None

    def test_just_whitespace_returns_none(self):
        assert extract_json("   \n\t  \n   ") is None

    def test_code_block_takes_priority(self):
        # Both a code-block JSON and a bare JSON in the same text.
        # The code block candidate is tried first.
        text = textwrap.dedent("""\
            Here is some data: {"bare": true}
            ```json
            {"block": true}
            ```
        """)
        result = extract_json(text)
        assert result == {"block": True}

    def test_repair_json_fixes_missing_comma_via_extract(self):
        # Simulate a broken LLM output: two adjacent object entries without commas.
        broken = textwrap.dedent("""\
            {
                "voice_authenticity": {"score": 8, "justification": "good"}
                "rhetorical_pattern": {"score": 7, "justification": "solid"}
            }
        """)
        result = extract_json(broken)
        assert result is not None
        assert result["voice_authenticity"]["score"] == 8
        assert result["rhetorical_pattern"]["score"] == 7

    def test_real_judge_output(self):
        """Multiline judge output with all 5 components, overall_score, and critical_failures."""
        judge_output = textwrap.dedent("""\
            Based on careful analysis of the response, here is my evaluation:

            ```json
            {
                "voice_authenticity": {
                    "score": 9,
                    "justification": "Captures Madison's measured, scholarly tone"
                },
                "rhetorical_pattern": {
                    "score": 8,
                    "justification": "Good use of structured argumentation"
                },
                "historical_accuracy": {
                    "score": 7,
                    "justification": "Mostly accurate, minor anachronism"
                },
                "position_fidelity": {
                    "score": 8,
                    "justification": "Consistent with known Madison positions"
                },
                "character_integrity": {
                    "score": 9,
                    "justification": "Maintains persona throughout"
                },
                "overall_score": 8.2,
                "critical_failures": []
            }
            ```

            The response demonstrates strong command of Madison's voice.
        """)
        result = extract_json(judge_output)
        assert result is not None
        assert len(result) == 7
        assert result["voice_authenticity"]["score"] == 9
        assert result["rhetorical_pattern"]["score"] == 8
        assert result["historical_accuracy"]["score"] == 7
        assert result["position_fidelity"]["score"] == 8
        assert result["character_integrity"]["score"] == 9
        assert result["overall_score"] == pytest.approx(8.2)
        assert result["critical_failures"] == []


# =========================================================================
# Parametrized edge-case sweeps
# =========================================================================


class TestComputeWeightedOverallParametrized:
    """Parametrized tests to reduce duplication on None-returning cases."""

    @pytest.mark.parametrize(
        "label, mutator",
        [
            ("entry_is_none", lambda s: s.update({"voice_authenticity": None}) or s),
            ("entry_is_list", lambda s: s.update({"voice_authenticity": [8]}) or s),
            ("entry_is_string", lambda s: s.update({"voice_authenticity": "high"}) or s),
        ],
        ids=lambda x: x if isinstance(x, str) else "",
    )
    def test_invalid_entry_shapes_return_none(self, label, mutator):
        scores = _make_scores()
        mutator(scores)
        assert compute_weighted_overall(scores) is None


class TestExtractJsonParametrized:
    """Parametrized tests for inputs that should return None."""

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "\n\n\n",
            "no braces anywhere",
            "just a number 42",
            '{"unclosed": "brace"',
        ],
        ids=[
            "empty",
            "whitespace",
            "newlines_only",
            "no_braces",
            "plain_number",
            "unclosed_brace",
        ],
    )
    def test_returns_none(self, text: str):
        assert extract_json(text) is None


# ---------------------------------------------------------------------------
# Small helper used by parametrized test above
# ---------------------------------------------------------------------------


def _set_score(scores: dict, component: str, value) -> dict:
    scores[component]["score"] = value
    return scores
