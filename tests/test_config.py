"""Tests for foundry.config — Pydantic models, loading, and singleton."""

from __future__ import annotations

import yaml
from pathlib import Path

from foundry.config import (
    FoundryConfig,
    ServerConfig,
    InferenceConfig,
    SamplerConfig,
    ContextConfig,
    StorageConfig,
    VoiceConfig,
    load_config,
    get_config,
)


class TestServerConfigDefaults:
    def test_host(self):
        assert ServerConfig().host == "0.0.0.0"

    def test_port(self):
        assert ServerConfig().port == 8080


class TestInferenceConfigDefaults:
    def test_backend(self):
        assert InferenceConfig().backend == "modal"

    def test_timeout(self):
        assert InferenceConfig().timeout == 120

    def test_local_endpoint(self):
        assert InferenceConfig().local_endpoint == "http://localhost:1234/v1"


class TestSamplerConfigDefaults:
    def test_temperature(self):
        assert SamplerConfig().temperature == 0.85

    def test_top_p(self):
        assert SamplerConfig().top_p == 0.92

    def test_max_tokens(self):
        assert SamplerConfig().max_tokens == 1024


class TestContextConfigDefaults:
    def test_max_length(self):
        assert ContextConfig().max_length == 16384


class TestStorageConfigDefaults:
    def test_db_path(self):
        assert StorageConfig().db_path == "data/foundry.db"


class TestVoiceConfigDefaults:
    def test_enabled(self):
        assert VoiceConfig().enabled is False

    def test_api_key_env(self):
        assert VoiceConfig().api_key_env == "ELEVENLABS_API_KEY"


class TestFoundryConfigDefaults:
    def test_all_sub_configs_present(self):
        config = FoundryConfig()
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.inference, InferenceConfig)
        assert isinstance(config.samplers, SamplerConfig)
        assert isinstance(config.context, ContextConfig)
        assert isinstance(config.storage, StorageConfig)
        assert isinstance(config.voice, VoiceConfig)


class TestLoadConfig:
    def test_valid_yaml_overrides(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump({"server": {"port": 9999}, "samplers": {"temperature": 0.5}}))
        config = load_config(cfg_file)
        assert config.server.port == 9999
        assert config.samplers.temperature == 0.5
        # Other defaults preserved
        assert config.server.host == "0.0.0.0"

    def test_missing_file_returns_defaults(self, tmp_path):
        config = load_config(tmp_path / "nonexistent.yaml")
        assert config.server.port == 8080

    def test_partial_yaml(self, tmp_path):
        cfg_file = tmp_path / "partial.yaml"
        cfg_file.write_text(yaml.dump({"storage": {"db_path": "/tmp/custom.db"}}))
        config = load_config(cfg_file)
        assert config.storage.db_path == "/tmp/custom.db"
        assert config.inference.backend == "modal"

    def test_empty_yaml(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        config = load_config(cfg_file)
        assert config.server.port == 8080


class TestGetConfig:
    def test_returns_foundry_config(self):
        config = get_config()
        assert isinstance(config, FoundryConfig)

    def test_singleton_returns_same_object(self):
        a = get_config()
        b = get_config()
        assert a is b
