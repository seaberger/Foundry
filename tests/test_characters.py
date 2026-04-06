"""Tests for foundry.characters.loader — CharacterCard, list/load characters."""

from __future__ import annotations

import pytest

from foundry.characters.loader import CharacterCard, list_characters, load_character


class TestCharacterCardFromYaml:
    def test_loads_valid_yaml(self, character_dir):
        card = CharacterCard.from_yaml(character_dir / "madison.yaml")
        assert card.name == "James Madison"
        assert card.role == "Fourth President of the United States"
        assert "James Madison" in card.system_prompt

    def test_nested_voice_config(self, character_dir):
        card = CharacterCard.from_yaml(character_dir / "madison.yaml")
        assert card.voice.tone == "formal"
        assert card.voice.speech_pattern == "complex periodic sentences"

    def test_personality_traits_list(self, character_dir):
        card = CharacterCard.from_yaml(character_dir / "madison.yaml")
        assert isinstance(card.personality.traits, list)
        assert "analytical" in card.personality.traits

    def test_empty_yaml_returns_defaults(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        card = CharacterCard.from_yaml(empty)
        assert card.name == ""
        assert card.voice.tone == ""
        assert card.personality.traits == []


class TestListCharacters:
    def test_lists_yaml_files(self, character_dir):
        names = list_characters(character_dir)
        assert names == ["hamilton", "madison"]

    def test_empty_directory(self, tmp_path):
        assert list_characters(tmp_path) == []

    def test_nonexistent_directory(self, tmp_path):
        assert list_characters(tmp_path / "nope") == []


class TestLoadCharacter:
    def test_loads_existing_character(self, character_dir):
        card = load_character("madison", character_dir)
        assert card.name == "James Madison"

    def test_missing_character_raises(self, character_dir):
        with pytest.raises(FileNotFoundError, match="Character 'washington'"):
            load_character("washington", character_dir)

    def test_name_maps_to_filename(self, character_dir):
        card = load_character("hamilton", character_dir)
        assert card.name == "Alexander Hamilton"
