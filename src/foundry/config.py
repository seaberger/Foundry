"""Configuration management for Foundry.

Loads settings from config/foundry.yaml with environment variable overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
DEFAULT_CONFIG = CONFIG_DIR / "foundry.yaml"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080


class InferenceConfig(BaseModel):
    backend: Literal["modal", "local"] = "modal"
    modal_endpoint: str = ""
    local_endpoint: str = "http://localhost:1234/v1"
    timeout: int = 120


class DebateConfig(BaseModel):
    max_turns_per_side: int = 10
    moderator_model: str = ""
    turn_time_limit: int = 0


class SamplerConfig(BaseModel):
    temperature: float = 0.85
    top_p: float = 0.92
    max_tokens: int = 1024


class ContextConfig(BaseModel):
    max_length: int = 16384
    compaction_threshold: float = 0.75


class StorageConfig(BaseModel):
    db_path: str = "data/foundry.db"


class VoiceConfig(BaseModel):
    enabled: bool = False
    api_key_env: str = "ELEVENLABS_API_KEY"


class FoundryConfig(BaseModel):
    """Root configuration model."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    debate: DebateConfig = Field(default_factory=DebateConfig)
    samplers: SamplerConfig = Field(default_factory=SamplerConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)


def load_config(path: Path | None = None) -> FoundryConfig:
    config_path = path or DEFAULT_CONFIG
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        return FoundryConfig.model_validate(raw)
    return FoundryConfig()


_config: FoundryConfig | None = None


def get_config() -> FoundryConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config
