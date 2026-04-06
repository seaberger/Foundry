"""Tests for foundry.press.evaluate — generation backend dispatch and judge logic."""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from foundry.press.evaluate import generate_response, judge_response, _mock_judge


@pytest.mark.integration
class TestGenerateResponse:
    def test_anthropic_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            generate_response("test", "", "model", backend="anthropic")

    def test_gemini_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            generate_response("test", "", "model", backend="gemini")

    def test_openai_native_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            generate_response("test", "", "model", backend="openai-native")

    def test_openai_backend_posts_to_endpoint(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response text"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("foundry.press.evaluate.httpx.post", return_value=mock_response) as mock_post:
            text, elapsed = generate_response(
                "test prompt", "http://localhost:1234/v1", "test-model",
                backend="openai",
            )

        assert text == "Response text"
        call_url = mock_post.call_args[0][0]
        assert call_url == "http://localhost:1234/v1/chat/completions"

    def test_system_prompt_included(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("foundry.press.evaluate.httpx.post", return_value=mock_response) as mock_post:
            generate_response(
                "test", "http://localhost:1234/v1", "model",
                system_prompt="You are Madison.",
                backend="openai",
            )

        payload = mock_post.call_args[1]["json"]
        messages = payload["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are Madison."


@pytest.mark.integration
class TestMockJudge:
    def test_returns_all_components(self):
        result = _mock_judge()
        assert "voice_authenticity" in result
        assert "rhetorical_pattern" in result
        assert "historical_accuracy" in result
        assert "position_fidelity" in result
        assert "character_integrity" in result
        assert "overall_score" in result

    def test_all_scores_zero(self):
        result = _mock_judge()
        for key in ["voice_authenticity", "rhetorical_pattern", "historical_accuracy",
                     "position_fidelity", "character_integrity"]:
            assert result[key]["score"] == 0
