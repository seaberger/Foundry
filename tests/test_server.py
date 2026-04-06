"""Tests for foundry.chamber.server — FastAPI routes via TestClient."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from foundry.chamber.server import app
from foundry.db import init_db


@pytest.mark.integration
class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_returns_ok(self, mock_config):
        init_db()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
