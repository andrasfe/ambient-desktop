"""Agent model for tracking active agents."""

from datetime import datetime
from enum import Enum
from typing import Optional, Any
from uuid import UUID

from sqlalchemy import String, DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from uuid_extensions import uuid7

from .base import Base


class AgentStatus(str, Enum):
    """Agent execution status."""

    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"


class AgentType(str, Enum):
    """Types of agents available."""

    COORDINATOR = "coordinator"
    BROWSER = "browser"
    FILE = "file"
    MCP = "mcp"
    CUSTOM = "custom"


class Agent(Base):
    """An active agent instance."""

    __tablename__ = "agents"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid7,
    )
    type: Mapped[AgentType] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[AgentStatus] = mapped_column(
        String(20),
        default=AgentStatus.IDLE,
        index=True,
    )
    
    # Current work
    current_task_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL"),
        nullable=True,
    )
    
    # Status information
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    progress: Mapped[Optional[float]] = mapped_column(nullable=True)  # 0.0 to 1.0
    extra_data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    
    # Health tracking
    last_heartbeat: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __repr__(self) -> str:
        return f"<Agent {self.id} {self.name} [{self.status}]>"

