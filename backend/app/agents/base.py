"""Base agent class with lifecycle management."""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID
from uuid_extensions import uuid7

from ..models import Agent, AgentStatus, AgentType, Task
from ..database import get_session
from ..services.websocket import ws_manager, EventType


class BaseAgent(ABC):
    """Base class for all agents."""

    agent_type: AgentType = AgentType.CUSTOM

    def __init__(self, name: Optional[str] = None):
        self.id: UUID = uuid7()
        self.name = name or f"{self.agent_type.value}-{str(self.id)[:8]}"
        self.status = AgentStatus.IDLE
        self.current_task: Optional[Task] = None
        self.summary: Optional[str] = None
        self.progress: Optional[float] = None
        self.extra_data: dict[str, Any] = {}
        self._db_agent: Optional[Agent] = None

    async def start(self) -> None:
        """Initialize and register the agent."""
        async with get_session() as session:
            self._db_agent = Agent(
                id=self.id,
                type=self.agent_type,
                name=self.name,
                status=AgentStatus.IDLE,
            )
            session.add(self._db_agent)
            await session.commit()

        await ws_manager.broadcast(
            EventType.AGENT_CREATED,
            {
                "id": str(self.id),
                "type": self.agent_type.value,
                "name": self.name,
                "status": self.status.value,
            },
        )
        await self.log("info", f"Agent {self.name} started")

    async def stop(self) -> None:
        """Clean up and unregister the agent."""
        async with get_session() as session:
            if self._db_agent:
                agent = await session.get(Agent, self.id)
                if agent:
                    agent.status = AgentStatus.STOPPED
                    await session.commit()

        await ws_manager.broadcast(
            EventType.AGENT_REMOVED,
            {"id": str(self.id)},
        )
        await self.log("info", f"Agent {self.name} stopped")

    async def update_status(
        self,
        status: AgentStatus,
        summary: Optional[str] = None,
        progress: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Update agent status and broadcast to clients."""
        self.status = status
        if summary is not None:
            self.summary = summary
        if progress is not None:
            self.progress = progress
        if metadata is not None:
            self.extra_data.update(metadata)

        async with get_session() as session:
            agent = await session.get(Agent, self.id)
            if agent:
                agent.status = status
                agent.summary = self.summary
                agent.progress = self.progress
                agent.extra_data = self.extra_data
                agent.last_heartbeat = datetime.now(timezone.utc)
                agent.current_task_id = self.current_task.id if self.current_task else None
                await session.commit()

        await ws_manager.broadcast_agent_update(
            agent_id=self.id,
            status=status.value,
            summary=self.summary,
            progress=self.progress,
            metadata=self.extra_data,
        )

    async def log(
        self,
        level: str,
        message: str,
        details: Optional[dict] = None,
    ) -> None:
        """Log a message and broadcast to clients."""
        await ws_manager.broadcast_log(
            level=level,
            message=message,
            category=self.agent_type.value,
            agent_id=self.id,
            task_id=self.current_task.id if self.current_task else None,
            details=details,
        )

    async def execute(self, task: Task) -> dict[str, Any]:
        """Execute a task and return results."""
        self.current_task = task
        await self.update_status(
            AgentStatus.BUSY,
            summary=f"Executing: {task.name}",
            progress=0.0,
        )

        try:
            result = await self._execute_task(task)
            await self.update_status(
                AgentStatus.IDLE,
                summary=f"Completed: {task.name}",
                progress=1.0,
            )
            return result
        except Exception as e:
            await self.update_status(
                AgentStatus.ERROR,
                summary=f"Failed: {task.name} - {str(e)}",
            )
            raise
        finally:
            self.current_task = None

    @abstractmethod
    async def _execute_task(self, task: Task) -> dict[str, Any]:
        """Execute the task logic. Override in subclasses."""
        pass

    async def heartbeat(self) -> None:
        """Update the agent's heartbeat timestamp."""
        async with get_session() as session:
            agent = await session.get(Agent, self.id)
            if agent:
                agent.last_heartbeat = datetime.now(timezone.utc)
                await session.commit()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.id} [{self.status}]>"

