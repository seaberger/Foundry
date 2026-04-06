"""Tests for CJK injection in create_language_pairs.py."""

from __future__ import annotations

import random
import re

from create_language_pairs import inject_cjk_at_words

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")

GOVERNANCE_TEXT = (
    "The republic must guard its constitution against the concentration of power, "
    "for liberty depends upon the separation of authority and the virtue of its citizens."
)

NO_TARGET_TEXT = "The cat sat on the mat and watched the birds fly by."


class TestInjectCjkAtWords:
    """inject_cjk_at_words corrupts text by inserting CJK fragments after governance words."""

    def test_returns_tuple_of_str_and_int(self):
        result = inject_cjk_at_words(GOVERNANCE_TEXT)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], int)

    def test_applied_count_lte_num_injections(self):
        corrupted, applied = inject_cjk_at_words(GOVERNANCE_TEXT, num_injections=3)
        assert applied <= 3
        assert applied > 0

    def test_injected_text_contains_cjk(self):
        corrupted, applied = inject_cjk_at_words(GOVERNANCE_TEXT, num_injections=3)
        assert applied > 0
        assert CJK_RE.search(corrupted) is not None

    def test_no_target_words_returns_original(self):
        corrupted, applied = inject_cjk_at_words(NO_TARGET_TEXT, num_injections=3)
        assert corrupted == NO_TARGET_TEXT
        assert applied == 0

    def test_zero_injections_returns_original(self):
        corrupted, applied = inject_cjk_at_words(GOVERNANCE_TEXT, num_injections=0)
        assert corrupted == GOVERNANCE_TEXT
        assert applied == 0

    def test_deterministic_with_seed(self):
        random.seed(42)
        result_a = inject_cjk_at_words(GOVERNANCE_TEXT, num_injections=2)
        random.seed(42)
        result_b = inject_cjk_at_words(GOVERNANCE_TEXT, num_injections=2)
        assert result_a == result_b

    def test_original_words_preserved(self):
        """The target word itself should still appear in the corrupted text."""
        corrupted, applied = inject_cjk_at_words(GOVERNANCE_TEXT, num_injections=3)
        assert applied > 0
        # At least one of the governance words from the original must still be present
        target_words = [
            "republic",
            "constitution",
            "power",
            "liberty",
            "separation",
            "authority",
            "virtue",
        ]
        found = [w for w in target_words if w in corrupted.lower()]
        assert len(found) > 0, "No target words found in corrupted text"
