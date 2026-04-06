"""Character card loading and management."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class VoiceConfig(BaseModel):
    tone: str = ""
    speech_pattern: str = ""
    vocabulary: str = ""


class PersonalityConfig(BaseModel):
    traits: list[str] = Field(default_factory=list)
    positions: list[str] = Field(default_factory=list)


class StyleConfig(BaseModel):
    prose_quality: str = ""
    response_length: str = ""
    anti_patterns: list[str] = Field(default_factory=list)


class CharacterCard(BaseModel):
    """A founding father character profile loaded from YAML."""

    name: str = ""
    role: str = "debater"
    description: str = ""
    system_prompt: str = ""
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)
    key_writings: list[str] = Field(default_factory=list)
    rhetorical_patterns: list[str] = Field(default_factory=list)
    intellectual_influences: list[str] = Field(default_factory=list)
    style: StyleConfig = Field(default_factory=StyleConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> CharacterCard:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)


def list_characters(config_dir: Path | None = None) -> list[str]:
    """List available character YAML files."""
    if config_dir is None:
        config_dir = Path(__file__).resolve().parent.parent.parent.parent / "config" / "characters"
    if not config_dir.exists():
        return []
    return sorted(f.stem for f in config_dir.glob("*.yaml"))


def load_character(name: str, config_dir: Path | None = None) -> CharacterCard:
    """Load a character by name."""
    if config_dir is None:
        config_dir = Path(__file__).resolve().parent.parent.parent.parent / "config" / "characters"
    path = config_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Character '{name}' not found at {path}")
    return CharacterCard.from_yaml(path)
