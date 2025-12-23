"""Unit tests for application configuration."""

import pytest
import os
from unittest.mock import patch


class TestSettings:
    """Tests for Settings configuration."""

    def test_host_default(self):
        """Test host default value."""
        from app.config import Settings
        # Settings can be configured via env, just check type
        settings = Settings()
        assert isinstance(settings.host, str)
        assert settings.host in ["0.0.0.0", "127.0.0.1", "localhost"]

    def test_port_is_integer(self):
        """Test port is an integer."""
        from app.config import Settings
        settings = Settings()
        assert isinstance(settings.port, int)
        assert settings.port > 0

    def test_is_local_llm_with_ollama_key(self):
        """Test local LLM detection with 'ollama' key."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "ollama"}, clear=True):
            from app.config import Settings
            settings = Settings()
            assert settings.is_local_llm is True

    def test_is_local_llm_with_localhost_url(self):
        """Test local LLM detection with localhost URL."""
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "some-key",
            "OPENROUTER_BASE_URL": "http://localhost:11434/v1"
        }, clear=True):
            from app.config import Settings
            settings = Settings()
            assert settings.is_local_llm is True

    def test_is_local_llm_with_127_0_0_1_url(self):
        """Test local LLM detection with 127.0.0.1 URL."""
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "some-key",
            "OPENROUTER_BASE_URL": "http://127.0.0.1:11434/v1"
        }, clear=True):
            from app.config import Settings
            settings = Settings()
            assert settings.is_local_llm is True

    def test_is_local_llm_false_for_cloud(self):
        """Test local LLM detection returns false for cloud API."""
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "sk-or-v1-xxx",
            "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1"
        }, clear=True):
            from app.config import Settings
            settings = Settings()
            assert settings.is_local_llm is False

    def test_mcp_server_configs_empty(self):
        """Test MCP servers config with empty array."""
        with patch.dict(os.environ, {"MCP_SERVERS": "[]"}, clear=True):
            from app.config import Settings
            settings = Settings()
            assert settings.mcp_server_configs == []

    def test_mcp_server_configs_with_servers(self):
        """Test MCP servers config with server entries."""
        servers_json = '[{"name": "test", "command": "mcp-test"}]'
        with patch.dict(os.environ, {"MCP_SERVERS": servers_json}, clear=True):
            from app.config import Settings
            settings = Settings()
            assert len(settings.mcp_server_configs) == 1
            assert settings.mcp_server_configs[0]["name"] == "test"

    def test_mcp_server_configs_invalid_json(self):
        """Test MCP servers config with invalid JSON."""
        with patch.dict(os.environ, {"MCP_SERVERS": "invalid json"}, clear=True):
            from app.config import Settings
            settings = Settings()
            assert settings.mcp_server_configs == []

    def test_privacy_mode_is_bool(self):
        """Test privacy mode is a boolean."""
        from app.config import Settings
        settings = Settings()
        assert isinstance(settings.privacy_mode, bool)

    def test_privacy_mode_enabled(self):
        """Test privacy mode can be enabled."""
        with patch.dict(os.environ, {"PRIVACY_MODE": "true"}, clear=True):
            from app.config import Settings
            settings = Settings()
            assert settings.privacy_mode is True

    def test_browser_cdp_url(self):
        """Test browser CDP URL configuration."""
        with patch.dict(os.environ, {"BROWSER_CDP_URL": "http://localhost:9222"}, clear=True):
            from app.config import Settings
            settings = Settings()
            assert settings.browser_cdp_url == "http://localhost:9222"

    def test_browser_user_data_dir(self):
        """Test browser user data dir configuration."""
        with patch.dict(os.environ, {"BROWSER_USER_DATA_DIR": "/tmp/chrome-profile"}, clear=True):
            from app.config import Settings
            settings = Settings()
            assert settings.browser_user_data_dir == "/tmp/chrome-profile"

