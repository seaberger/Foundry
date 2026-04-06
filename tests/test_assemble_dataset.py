"""Tests for voice contamination scoring in assemble_v4_dataset.py."""

from __future__ import annotations

from foundry.press.voice import ABSOLUTELY_SLOP_RE, voice_score
from foundry.press.voice import is_contaminated as is_chosen_contaminated

# ---------------------------------------------------------------------------
# Clean Madison prose used across tests
# ---------------------------------------------------------------------------

CLEAN_MADISON = (
    "The regulation of these various and interfering interests forms the principal "
    "task of modern legislation, and involves the spirit of party and faction in the "
    "necessary and ordinary operations of the government."
)


class TestVoiceScore:
    """voice_score returns a dict with contractions, bullets, filler, total."""

    def test_clean_madison_prose_scores_zero(self):
        result = voice_score(CLEAN_MADISON)
        assert result["total"] == 0
        assert result["contractions"] == 0
        assert result["bullets"] == 0
        assert result["filler"] == 0

    def test_contraction_detected(self):
        text = "I shouldn't agree with the proposition that liberty ought to be curtailed."
        result = voice_score(text)
        assert result["contractions"] >= 1

    def test_bullets_detected(self):
        text = "The following considerations apply:\n- First point\n- Second point"
        result = voice_score(text)
        assert result["bullets"] >= 1

    def test_modern_filler_detected(self):
        text = "That's a great question. The republic must be preserved through virtue."
        result = voice_score(text)
        assert result["filler"] >= 1

    def test_absolutely_at_sentence_start_detected(self):
        text = "Some argue otherwise. Absolutely. The republic demands vigilance."
        result = voice_score(text)
        assert result["filler"] >= 1

    def test_absolutely_mid_sentence_not_detected(self):
        """'absolutely' used as a mid-sentence adverb should not trigger ABSOLUTELY_SLOP_RE."""
        text = "I am absolutely certain that the constitution must be defended."
        # ABSOLUTELY_SLOP_RE should not fire (mid-sentence, lowercase)
        assert len(ABSOLUTELY_SLOP_RE.findall(text)) == 0
        result = voice_score(text)
        # filler may still be 0 since MODERN_FILLER list does not include bare "absolutely"
        assert result["filler"] == 0

    def test_multiple_issues_accumulate(self):
        text = (
            "That's a great question! I can't believe we haven't discussed this.\n"
            "- First point\n"
            "- Second point\n"
            "Absolutely. We must act now."
        )
        result = voice_score(text)
        assert result["contractions"] >= 1
        assert result["bullets"] >= 1
        assert result["filler"] >= 1
        assert result["total"] > 2

    def test_total_equals_sum_of_components(self):
        text = (
            "I can't deny it. Let me break this down.\n"
            "- Point one\n"
            "Absolutely. The republic is at stake."
        )
        result = voice_score(text)
        expected_total = result["contractions"] + result["bullets"] + result["filler"]
        assert result["total"] == expected_total


class TestIsChosenContaminated:
    """is_chosen_contaminated returns (bool, score_dict)."""

    def test_clean_text_not_contaminated(self):
        contaminated, score = is_chosen_contaminated(CLEAN_MADISON)
        assert contaminated is False
        assert score["total"] == 0

    def test_contraction_triggers_contamination(self):
        text = "We shouldn't allow faction to dominate the republic."
        contaminated, score = is_chosen_contaminated(text)
        assert contaminated is True
        assert score["contractions"] >= 1

    def test_returns_tuple_of_bool_and_dict(self):
        result = is_chosen_contaminated(CLEAN_MADISON)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], dict)

    def test_filler_triggers_contamination(self):
        text = "Here's the thing about the separation of powers in a republic."
        contaminated, score = is_chosen_contaminated(text)
        assert contaminated is True
        assert score["filler"] >= 1
