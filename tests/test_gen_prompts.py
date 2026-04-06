"""Tests for prompt generation utilities in foundry.press.gen_prompts."""

from __future__ import annotations

import pytest

from foundry.press.gen_prompts import TESTS_PATH, THEMES, deduplicate, load_seed_prompts


class TestDeduplicate:
    """deduplicate removes exact and near-duplicate prompts."""

    def test_no_duplicates_preserved(self):
        prompts = [
            {"prompt": "What is liberty?", "theme": "a"},
            {"prompt": "What is faction?", "theme": "b"},
            {"prompt": "What is virtue?", "theme": "c"},
        ]
        result = deduplicate(prompts)
        assert len(result) == 3

    def test_exact_duplicate_removed(self):
        prompts = [
            {"prompt": "What is liberty?", "theme": "a"},
            {"prompt": "What is liberty?", "theme": "b"},
        ]
        result = deduplicate(prompts)
        assert len(result) == 1

    def test_case_insensitive(self):
        prompts = [
            {"prompt": "What is liberty?", "theme": "a"},
            {"prompt": "what is liberty?", "theme": "b"},
        ]
        result = deduplicate(prompts)
        assert len(result) == 1

    def test_trailing_punctuation_ignored(self):
        prompts = [
            {"prompt": "What is liberty?", "theme": "a"},
            {"prompt": "What is liberty", "theme": "b"},
        ]
        result = deduplicate(prompts)
        assert len(result) == 1

    def test_preserves_order_first_kept(self):
        prompts = [
            {"prompt": "What is liberty?", "theme": "first"},
            {"prompt": "What is faction?", "theme": "second"},
            {"prompt": "what is liberty?", "theme": "duplicate"},
        ]
        result = deduplicate(prompts)
        assert len(result) == 2
        assert result[0]["theme"] == "first"
        assert result[1]["theme"] == "second"

    def test_empty_list(self):
        assert deduplicate([]) == []


class TestThemes:
    """THEMES list contains well-formed theme definitions."""

    REQUIRED_KEYS = {"name", "section", "description", "target_count", "register", "seed_questions"}

    def test_all_themes_have_required_keys(self):
        for theme in THEMES:
            missing = self.REQUIRED_KEYS - set(theme.keys())
            assert not missing, f"Theme '{theme.get('name', '?')}' missing keys: {missing}"

    def test_all_target_counts_positive(self):
        for theme in THEMES:
            assert theme["target_count"] > 0, f"Theme '{theme['name']}' has non-positive target_count"

    def test_all_seed_questions_nonempty(self):
        for theme in THEMES:
            assert len(theme["seed_questions"]) > 0, (
                f"Theme '{theme['name']}' has empty seed_questions"
            )

    def test_theme_names_unique(self):
        names = [t["name"] for t in THEMES]
        assert len(names) == len(set(names)), "Duplicate theme names found"


class TestLoadSeedPrompts:
    """load_seed_prompts reads behavioral tests into prompt dicts."""

    def test_returns_list_of_dicts(self):
        if not TESTS_PATH.exists():
            pytest.skip("Behavioral tests file not present")
        result = load_seed_prompts()
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], dict)

    def test_dicts_have_required_keys(self):
        if not TESTS_PATH.exists():
            pytest.skip("Behavioral tests file not present")
        result = load_seed_prompts()
        required = {"prompt", "theme", "register", "source"}
        for item in result:
            missing = required - set(item.keys())
            assert not missing, f"Seed prompt missing keys: {missing}"
