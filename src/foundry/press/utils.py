"""Shared utilities for the Foundry press pipeline.

Centralizes common operations that were previously duplicated across
teacher.py, student.py, opus_teacher.py, format_dpo.py, and evaluate.py.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import httpx

log = logging.getLogger("foundry.press.utils")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONSTITUTION_PATH = PROJECT_ROOT / "config" / "constitutions" / "madison-5k.md"

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list[dict], path: Path) -> None:
    """Save a list of dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_constitution(path: Path | None = None) -> str:
    """Load the Madison constitution, keeping only character content.

    Strips metadata headers and returns content starting from section 1.
    """
    path = path or CONSTITUTION_PATH
    text = path.read_text()
    lines = text.split("\n")
    content_lines = []
    in_content = False
    for line in lines:
        if line.startswith("## 1."):
            in_content = True
        if in_content:
            content_lines.append(line)
    return "\n".join(content_lines)


def anthropic_post(
    system: str,
    user: str,
    model: str,
    *,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    cache_system: bool = False,
    timeout: int = 120,
) -> dict:
    """Make a POST to the Anthropic Messages API.

    Args:
        system: System prompt text.
        user: User message text.
        model: Model ID (e.g., "claude-4-sonnet-20250514").
        max_tokens: Max response tokens.
        temperature: Sampling temperature.
        cache_system: If True, wrap system in cache_control for prompt caching.
        timeout: Request timeout in seconds.

    Returns:
        Full API response dict (with "content", "usage", etc.).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    if cache_system:
        system_payload = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system_payload = system

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_payload,
        "messages": [{"role": "user", "content": user}],
        "temperature": temperature,
    }

    response = httpx.post(
        ANTHROPIC_API_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()
