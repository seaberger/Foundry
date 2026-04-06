"""Tests for foundry.press.teacher — constitution loading and prompt loading."""

from __future__ import annotations

from pathlib import Path

from foundry.press.teacher import load_constitution, load_prompts


class TestLoadConstitution:
    def test_returns_string(self):
        text = load_constitution()
        assert isinstance(text, str)
        assert len(text) > 100

    def test_contains_madison_content(self):
        text = load_constitution()
        # The constitution should mention Madison or key concepts
        assert "madison" in text.lower() or "republic" in text.lower()


class TestLoadPrompts:
    def test_returns_list_of_dicts(self):
        prompts = load_prompts()
        assert isinstance(prompts, list)
        if prompts:  # May be empty if no prompts file exists
            assert isinstance(prompts[0], dict)
            assert "prompt" in prompts[0]

    def test_custom_path(self, tmp_path):
        import json
        path = tmp_path / "test-prompts.jsonl"
        path.write_text(
            json.dumps({"prompt": "Test question?", "theme": "test"}) + "\n"
        )
        prompts = load_prompts(path)
        assert len(prompts) == 1
        assert prompts[0]["prompt"] == "Test question?"
