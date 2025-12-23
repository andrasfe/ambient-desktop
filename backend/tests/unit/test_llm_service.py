"""Unit tests for LLM service."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.llm import LLMService, Message, LLMResponse


class TestLLMService:
    """Tests for LLMService."""

    def test_init_default(self):
        """Test default initialization."""
        service = LLMService()
        assert service.api_key is not None
        assert service.model is not None
        assert service.base_url is not None

    def test_init_custom(self):
        """Test custom initialization."""
        service = LLMService(
            api_key="test-key",
            model="test-model",
            base_url="https://test.example.com",
        )
        assert service.api_key == "test-key"
        assert service.model == "test-model"
        assert service.base_url == "https://test.example.com"

    @pytest.mark.asyncio
    async def test_complete(self):
        """Test completion method."""
        service = LLMService(api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": "Hello, world!"},
                "finish_reason": "stop",
            }],
            "model": "test-model",
            "usage": {"total_tokens": 10},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(service, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            messages = [Message(role="user", content="Hello")]
            response = await service.complete(messages)

            assert isinstance(response, LLMResponse)
            assert response.content == "Hello, world!"
            assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_parse_task_valid_json(self):
        """Test task parsing with valid JSON response."""
        service = LLMService(api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"understanding": "Test task", "tasks": [{"name": "Task 1", "agent_type": "browser", "description": "Do something", "payload": {}, "dependencies": []}]}'
                },
                "finish_reason": "stop",
            }],
            "model": "test-model",
            "usage": {},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(service, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await service.parse_task("Test instruction")

            assert "understanding" in result
            assert "tasks" in result
            assert len(result["tasks"]) == 1
            assert result["tasks"][0]["name"] == "Task 1"


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self):
        """Test message creation."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_response_creation(self):
        """Test response creation."""
        response = LLMResponse(
            content="Test",
            model="test-model",
            usage={"tokens": 10},
            finish_reason="stop",
        )
        assert response.content == "Test"
        assert response.model == "test-model"
        assert response.usage == {"tokens": 10}
        assert response.finish_reason == "stop"

