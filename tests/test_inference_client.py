"""Tests for foundry.inference.client — async streaming with mocked httpx."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry.inference.client import stream_chat


@pytest.mark.integration
class TestStreamChat:
    @pytest.mark.asyncio
    async def test_yields_tokens(self, mock_config):
        """Verify stream_chat yields token strings from mocked SSE."""
        mock_config.inference.backend = "local"

        chunks = [
            {"choices": [{"delta": {"content": "Hello"}, "index": 0}]},
            {"choices": [{"delta": {"content": " world"}, "index": 0}]},
        ]

        async def mock_aiter_lines():
            for chunk in chunks:
                yield f"data: {json.dumps(chunk)}"
            yield "data: [DONE]"

        mock_response = AsyncMock()
        mock_response.aiter_lines = mock_aiter_lines

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_cm)

        with patch("foundry.inference.client._get_client", return_value=mock_client):
            tokens = [t async for t in stream_chat("system", [{"role": "user", "content": "hi"}])]

        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_handles_done_immediately(self, mock_config):
        """Empty stream with only [DONE] yields no tokens."""
        mock_config.inference.backend = "local"

        async def mock_aiter_lines():
            yield "data: [DONE]"

        mock_response = AsyncMock()
        mock_response.aiter_lines = mock_aiter_lines

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_cm)

        with patch("foundry.inference.client._get_client", return_value=mock_client):
            tokens = [t async for t in stream_chat("sys", [{"role": "user", "content": "hi"}])]

        assert tokens == []

    @pytest.mark.asyncio
    async def test_skips_non_data_lines(self, mock_config):
        """Lines not starting with 'data: ' are ignored."""
        mock_config.inference.backend = "local"

        async def mock_aiter_lines():
            yield ": keep-alive"
            yield "event: ping"
            yield f"data: {json.dumps({'choices': [{'delta': {'content': 'ok'}, 'index': 0}]})}"
            yield "data: [DONE]"

        mock_response = AsyncMock()
        mock_response.aiter_lines = mock_aiter_lines

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_cm)

        with patch("foundry.inference.client._get_client", return_value=mock_client):
            tokens = [t async for t in stream_chat("sys", [{"role": "user", "content": "hi"}])]

        assert tokens == ["ok"]
