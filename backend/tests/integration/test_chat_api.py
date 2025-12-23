"""Integration tests for chat API with mocked LLM."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture
async def test_client():
    """Create a test client without database."""
    # Don't raise app exceptions into the test runner; convert them to 500 responses.
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestChatMessage:
    """Tests for chat message endpoint."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, test_client):
        """Test sending a chat message."""
        # Mock the orchestrator
        with patch("app.api.chat.orchestrator") as mock_orchestrator:
            mock_orchestrator.process_message = AsyncMock(return_value="Hello! How can I help?")
            
            response = await test_client.post(
                "/chat/message",
                json={"message": "Hello", "session_id": "test-session"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "Hello! How can I help?"
            assert data["session_id"] == "test-session"

    @pytest.mark.asyncio
    async def test_send_message_default_session(self, test_client):
        """Test sending message with default session."""
        with patch("app.api.chat.orchestrator") as mock_orchestrator:
            mock_orchestrator.process_message = AsyncMock(return_value="Response")
            
            response = await test_client.post(
                "/chat/message",
                json={"message": "Test"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "default"

    @pytest.mark.asyncio
    async def test_send_message_error(self, test_client):
        """Test error handling for chat message."""
        with patch("app.api.chat.orchestrator") as mock_orchestrator:
            mock_orchestrator.process_message = AsyncMock(side_effect=Exception("LLM error"))
            
            response = await test_client.post(
                "/chat/message",
                json={"message": "Hello"}
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "LLM error" in data["detail"]


class TestChatSessions:
    """Tests for chat session management."""

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, test_client):
        """Test listing sessions when empty."""
        with patch("app.api.chat.orchestrator") as mock_orchestrator:
            mock_orchestrator.sessions = {}
            
            response = await test_client.get("/chat/sessions")
            
            assert response.status_code == 200
            data = response.json()
            assert data["sessions"] == []
            assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_list_sessions_with_data(self, test_client):
        """Test listing sessions with data."""
        with patch("app.api.chat.orchestrator") as mock_orchestrator:
            mock_orchestrator.sessions = {
                "session-1": "thread-1",
                "session-2": "thread-2",
            }
            
            response = await test_client.get("/chat/sessions")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["sessions"]) == 2
            assert data["count"] == 2

    @pytest.mark.asyncio
    async def test_clear_session_success(self, test_client):
        """Test clearing a session."""
        with patch("app.api.chat.orchestrator") as mock_orchestrator:
            mock_orchestrator.sessions = {"test-session": "thread-1"}
            
            response = await test_client.delete("/chat/sessions/test-session")
            
            assert response.status_code == 200
            data = response.json()
            assert data["cleared"] is True
            assert data["session_id"] == "test-session"

    @pytest.mark.asyncio
    async def test_clear_session_not_found(self, test_client):
        """Test clearing non-existent session."""
        with patch("app.api.chat.orchestrator") as mock_orchestrator:
            mock_orchestrator.sessions = {}
            
            response = await test_client.delete("/chat/sessions/nonexistent")
            
            assert response.status_code == 404


class TestChatHistory:
    """Tests for chat history endpoint."""

    @pytest.mark.asyncio
    async def test_get_history_empty(self, test_client):
        """Test getting empty history."""
        response = await test_client.get("/chat/history")
        
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert "length" in data

