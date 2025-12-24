"""Application configuration from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import json


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://ambient:ambient@localhost:5432/ambient",
        description="PostgreSQL connection string",
    )

    # LLM Configuration (supports OpenRouter, Ollama, LMStudio, or any OpenAI-compatible API)
    openrouter_api_key: str = Field(default="", description="API key (use 'ollama' or 'lmstudio' for local)")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="API base URL (e.g., http://localhost:11434/v1 for Ollama, http://localhost:1234/v1 for LMStudio)",
    )
    openrouter_model: str = Field(
        default="anthropic/claude-3.5-sonnet",
        description="Model to use (e.g., llama3.1:8b for Ollama, or model loaded in LMStudio)",
    )
    
    # Privacy settings
    privacy_mode: bool = Field(
        default=False,
        description="When true, minimizes data sent to LLM (only sends task descriptions). Auto-enabled for local LLMs.",
    )
    local_llm: bool = Field(
        default=False,
        description="Auto-detected: true when using Ollama, LMStudio, or local endpoint",
    )
    
    @property
    def is_local_llm(self) -> bool:
        """Check if we're using a local LLM (Ollama, LMStudio, or other local endpoint)."""
        return (
            self.openrouter_api_key.lower() in ("ollama", "lmstudio", "") or
            "localhost" in self.openrouter_base_url or
            "127.0.0.1" in self.openrouter_base_url or
            self.local_llm
        )
    
    @property
    def is_lmstudio(self) -> bool:
        """Check if we're using LMStudio."""
        return (
            self.openrouter_api_key.lower() == "lmstudio" or
            ":1234" in self.openrouter_base_url
        )
    
    @property
    def requires_api_key(self) -> bool:
        """Check if the current LLM configuration requires an API key."""
        return not self.is_local_llm

    # Cohere Embeddings
    cohere_api_key: str = Field(default="", description="Cohere API key")
    cohere_model: str = Field(
        default="embed-english-v3.0",
        description="Cohere embedding model",
    )

    # Scheduler
    scheduler_interval_seconds: int = Field(
        default=30,
        description="How often to check for pending tasks",
    )

    # MCP Servers (JSON array of server configs)
    mcp_servers: str = Field(
        default="[]",
        description="JSON array of MCP server configurations",
    )

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")

    # Agent settings
    max_concurrent_agents: int = Field(
        default=5,
        description="Maximum number of concurrent worker agents",
    )

    # Browser settings
    browser_cdp_url: str = Field(
        default="",
        description="CDP URL to connect to existing browser (e.g., http://localhost:9222)",
    )
    browser_user_data_dir: str = Field(
        default="",
        description="Directory for persistent browser profile",
    )
    browser_headless: bool = Field(
        default=True,
        description="Run browser in headless mode (ignored when using CDP)",
    )

    @property
    def mcp_server_configs(self) -> list[dict]:
        """Parse MCP servers from JSON string."""
        try:
            return json.loads(self.mcp_servers)
        except json.JSONDecodeError:
            return []

    class Config:
        env_file = "../.env"  # .env is in project root, not backend/
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()

