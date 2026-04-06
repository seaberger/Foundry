"""Tests for foundry.db — SQLite schema, connections, utilities."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from foundry.db import init_db, get_db, now_iso


class TestInitDb:
    def test_creates_all_tables(self, mock_config):
        init_db()
        with get_db() as db:
            tables = {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "sessions" in tables
        assert "turns" in tables
        assert "character_knowledge" in tables
        assert "schema_info" in tables

    def test_schema_version_set(self, mock_config):
        init_db()
        with get_db() as db:
            row = db.execute(
                "SELECT value FROM schema_info WHERE key = 'version'"
            ).fetchone()
        assert row["value"] == "1"

    def test_idempotent(self, mock_config):
        init_db()
        init_db()  # Should not error
        with get_db() as db:
            count = db.execute(
                "SELECT COUNT(*) FROM schema_info WHERE key = 'version'"
            ).fetchone()[0]
        assert count == 1


class TestGetDb:
    def test_returns_connection(self, mock_config):
        init_db()
        with get_db() as db:
            assert isinstance(db, sqlite3.Connection)

    def test_row_factory_set(self, mock_config):
        init_db()
        with get_db() as db:
            assert db.row_factory is sqlite3.Row

    def test_insert_and_read(self, mock_config):
        init_db()
        with get_db() as db:
            db.execute(
                "INSERT INTO sessions (id, name, created_at, last_active) "
                "VALUES (?, ?, ?, ?)",
                ("test-1", "Test Session", "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
            )
            db.commit()
        with get_db() as db:
            row = db.execute("SELECT name FROM sessions WHERE id = 'test-1'").fetchone()
        assert row["name"] == "Test Session"


class TestNowIso:
    def test_returns_iso_format(self):
        result = now_iso()
        # Should be parseable as ISO 8601
        parsed = datetime.fromisoformat(result)
        assert isinstance(parsed, datetime)

    def test_utc_timezone(self):
        result = now_iso()
        parsed = datetime.fromisoformat(result)
        assert parsed.tzinfo is not None
        assert parsed.tzinfo == timezone.utc
