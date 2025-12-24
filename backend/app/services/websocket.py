"""WebSocket manager for real-time updates."""

import asyncio
import json
from datetime import datetime
from typing import Any, Optional
from uuid import UUID
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import WebSocket


class EventType(str, Enum):
    """WebSocket event types."""
    
    # Chat events
    CHAT_MESSAGE = "chat:message"
    CHAT_STREAM = "chat:stream"
    CHAT_STREAM_END = "chat:stream_end"
    CHAT_CANCELLED = "chat:cancelled"
    
    # Agent events
    AGENT_CREATED = "agent:created"
    AGENT_UPDATE = "agent:update"
    AGENT_REMOVED = "agent:removed"
    
    # Task events
    TASK_CREATED = "task:created"
    TASK_UPDATE = "task:update"
    TASK_COMPLETED = "task:completed"
    TASK_FAILED = "task:failed"
    
    # Log events
    LOG_ENTRY = "log:entry"
    
    # System events
    SYSTEM_STATUS = "system:status"
    ERROR = "error"


@dataclass
class WebSocketEvent:
    """A WebSocket event."""
    type: EventType
    data: dict
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "type": self.type.value if isinstance(self.type, EventType) else self.type,
            "data": self.data,
            "timestamp": self.timestamp,
        }, default=str)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts."""

    def __init__(self):
        self._connections: dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self._connections[client_id] = websocket

    async def disconnect(self, client_id: str) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            self._connections.pop(client_id, None)

    async def send_event(
        self,
        client_id: str,
        event_type: EventType,
        data: dict,
    ) -> bool:
        """Send an event to a specific client."""
        async with self._lock:
            websocket = self._connections.get(client_id)
        
        if not websocket:
            print(f"[WS-MGR] ❌ No websocket found for client: {client_id}")
            print(f"[WS-MGR] Active clients: {list(self._connections.keys())}")
            return False
        
        try:
            event = WebSocketEvent(type=event_type, data=data)
            json_msg = event.to_json()
            print(f"[WS-MGR] Sending to {client_id}: {event_type.value} ({len(json_msg)} bytes)")
            await websocket.send_text(json_msg)
            print(f"[WS-MGR] ✅ Sent successfully")
            return True
        except Exception as e:
            print(f"[WS-MGR] ❌ Send failed: {e}")
            await self.disconnect(client_id)
            return False

    async def broadcast(
        self,
        event_type: EventType,
        data: dict,
        exclude: Optional[set[str]] = None,
    ) -> int:
        """Broadcast an event to all connected clients."""
        exclude = exclude or set()
        event = WebSocketEvent(type=event_type, data=data)
        message = event.to_json()
        
        async with self._lock:
            clients = list(self._connections.items())
        
        sent_count = 0
        disconnected = []
        
        for client_id, websocket in clients:
            if client_id in exclude:
                continue
            try:
                await websocket.send_text(message)
                sent_count += 1
            except Exception:
                disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            await self.disconnect(client_id)
        
        return sent_count

    async def broadcast_log(
        self,
        level: str,
        message: str,
        category: str = "general",
        agent_id: Optional[UUID] = None,
        task_id: Optional[UUID] = None,
        details: Optional[dict] = None,
    ) -> None:
        """Broadcast a log entry to all clients."""
        await self.broadcast(
            EventType.LOG_ENTRY,
            {
                "level": level,
                "category": category,
                "message": message,
                "agent_id": str(agent_id) if agent_id else None,
                "task_id": str(task_id) if task_id else None,
                "details": details,
            },
        )

    async def broadcast_agent_update(
        self,
        agent_id: UUID,
        status: str,
        summary: Optional[str] = None,
        progress: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Broadcast an agent status update."""
        await self.broadcast(
            EventType.AGENT_UPDATE,
            {
                "id": str(agent_id),
                "status": status,
                "summary": summary,
                "progress": progress,
                "metadata": metadata,
            },
        )

    async def broadcast_task_update(
        self,
        task_id: UUID,
        status: str,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        """Broadcast a task status update."""
        event_type = EventType.TASK_UPDATE
        if status == "completed":
            event_type = EventType.TASK_COMPLETED
        elif status == "failed":
            event_type = EventType.TASK_FAILED
        
        await self.broadcast(
            event_type,
            {
                "id": str(task_id),
                "status": status,
                "result": result,
                "error": error,
            },
        )

    @property
    def connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self._connections)

    async def get_connection_ids(self) -> list[str]:
        """Get all connected client IDs."""
        async with self._lock:
            return list(self._connections.keys())


# Global instance
ws_manager = WebSocketManager()

