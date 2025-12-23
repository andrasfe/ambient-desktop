"""Activity log model for tracking agent actions."""

from datetime import datetime
from enum import Enum
from typing import Optional, Any
from uuid import UUID

from sqlalchemy import String, DateTime, ForeignKey, Text, Integer, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class LogLevel(str, Enum):
    """Log severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class ActivityLog(Base):
    """Activity log entry for UI display."""

    __tablename__ = "activity_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # References
    agent_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    task_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    
    # Log content
    level: Mapped[LogLevel] = mapped_column(String(10), default=LogLevel.INFO)
    category: Mapped[str] = mapped_column(String(50), default="general")
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
    )

    def __repr__(self) -> str:
        return f"<ActivityLog {self.id} [{self.level}] {self.message[:50]}>"

