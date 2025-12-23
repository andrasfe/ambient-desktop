"""Unit tests for WebSocket manager."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.websocket import WebSocketManager, EventType, WebSocketEvent


class TestWebSocketEvent:
    """Tests for WebSocketEvent."""

    def test_event_creation(self):
        """Test event creation."""
        event = WebSocketEvent(
            type=EventType.CHAT_MESSAGE,
            data={"content": "Hello"},
        )
        assert event.type == EventType.CHAT_MESSAGE
        assert event.data == {"content": "Hello"}
        assert event.timestamp

    def test_event_to_json(self):
        """Test event JSON serialization."""
        event = WebSocketEvent(
            type=EventType.CHAT_MESSAGE,
            data={"content": "Hello"},
        )
        json_str = event.to_json()
        assert "chat:message" in json_str
        assert "Hello" in json_str


class TestWebSocketManager:
    """Tests for WebSocketManager."""

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test WebSocket connection."""
        manager = WebSocketManager()
        
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        
        await manager.connect(mock_ws, "client-1")
        
        mock_ws.accept.assert_called_once()
        assert "client-1" in await manager.get_connection_ids()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test WebSocket disconnection."""
        manager = WebSocketManager()
        
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        
        await manager.connect(mock_ws, "client-1")
        await manager.disconnect("client-1")
        
        assert "client-1" not in await manager.get_connection_ids()

    @pytest.mark.asyncio
    async def test_send_event(self):
        """Test sending event to specific client."""
        manager = WebSocketManager()
        
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        
        await manager.connect(mock_ws, "client-1")
        result = await manager.send_event(
            "client-1",
            EventType.CHAT_MESSAGE,
            {"content": "Hello"},
        )
        
        assert result is True
        mock_ws.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_event_unknown_client(self):
        """Test sending event to unknown client."""
        manager = WebSocketManager()
        
        result = await manager.send_event(
            "unknown",
            EventType.CHAT_MESSAGE,
            {"content": "Hello"},
        )
        
        assert result is False

    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Test broadcasting to all clients."""
        manager = WebSocketManager()
        
        mock_ws1 = AsyncMock()
        mock_ws1.accept = AsyncMock()
        mock_ws1.send_text = AsyncMock()
        
        mock_ws2 = AsyncMock()
        mock_ws2.accept = AsyncMock()
        mock_ws2.send_text = AsyncMock()
        
        await manager.connect(mock_ws1, "client-1")
        await manager.connect(mock_ws2, "client-2")
        
        count = await manager.broadcast(
            EventType.SYSTEM_STATUS,
            {"status": "ok"},
        )
        
        assert count == 2
        mock_ws1.send_text.assert_called_once()
        mock_ws2.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_with_exclude(self):
        """Test broadcasting with exclusion."""
        manager = WebSocketManager()
        
        mock_ws1 = AsyncMock()
        mock_ws1.accept = AsyncMock()
        mock_ws1.send_text = AsyncMock()
        
        mock_ws2 = AsyncMock()
        mock_ws2.accept = AsyncMock()
        mock_ws2.send_text = AsyncMock()
        
        await manager.connect(mock_ws1, "client-1")
        await manager.connect(mock_ws2, "client-2")
        
        count = await manager.broadcast(
            EventType.SYSTEM_STATUS,
            {"status": "ok"},
            exclude={"client-1"},
        )
        
        assert count == 1
        mock_ws1.send_text.assert_not_called()
        mock_ws2.send_text.assert_called_once()

    def test_connection_count(self):
        """Test connection count property."""
        manager = WebSocketManager()
        assert manager.connection_count == 0

