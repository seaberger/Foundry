"""OpenAI-compatible inference client with streaming support.

Uses httpx directly rather than the openai Python client for broader
compatibility with local servers (llama.cpp, Ollama, LM Studio, vLLM)
that may include non-standard fields like reasoning_content.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator

import httpx

from ..config import get_config

log = logging.getLogger("foundry.inference")

# Reusable client — avoids per-request connection pool overhead
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        config = get_config()
        _client = httpx.AsyncClient(timeout=config.inference.timeout)
    return _client


def _base_url() -> str:
    config = get_config()
    if config.inference.backend == "local":
        return config.inference.local_endpoint
    return config.inference.modal_endpoint or config.inference.local_endpoint


async def stream_chat(
    system_prompt: str,
    messages: list[dict[str, str]],
    model: str = "",
) -> AsyncGenerator[str, None]:
    """Stream a chat completion, yielding content tokens as they arrive.

    Args:
        system_prompt: Character system prompt.
        messages: Conversation history as [{"role": "user"|"assistant", "content": "..."}].
        model: Model name override. Empty string lets the server decide.
    """
    config = get_config()
    base = _base_url().rstrip("/")
    url = f"{base}/chat/completions"

    api_messages = [{"role": "system", "content": system_prompt}]
    api_messages.extend(messages)

    payload = {
        "model": model or "local-model",
        "messages": api_messages,
        "temperature": config.samplers.temperature,
        "top_p": config.samplers.top_p,
        "max_tokens": config.samplers.max_tokens,
        "stream": True,
    }

    try:
        client = _get_client()
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue
    except Exception as e:
        log.error("Inference error: %s", e)
        yield f"\n\n[Error communicating with inference server: {e}]"
