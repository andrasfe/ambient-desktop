"""Task model for the action queue."""

from datetime import datetime
from enum import Enum
from typing import Optional, Any
from uuid import UUID

from sqlalchemy import String, Integer, DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from uuid_extensions import uuid7


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


from .base import Base


class Task(Base):
    """A task in the action queue."""

    __tablename__ = "tasks"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid7,
    )
    parent_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[TaskStatus] = mapped_column(
        String(20),
        default=TaskStatus.PENDING,
        index=True,
    )
    priority: Mapped[int] = mapped_column(Integer, default=0)
    
    # Task definition
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    
    # Result
    result: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Scheduling
    scheduled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
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

    # Relationships
    parent: Mapped[Optional["Task"]] = relationship(
        "Task",
        remote_side=[id],
        back_populates="subtasks",
    )
    subtasks: Mapped[list["Task"]] = relationship(
        "Task",
        back_populates="parent",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Task {self.id} {self.name} [{self.status}]>"

