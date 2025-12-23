"""Task queue API endpoints."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import Task, TaskStatus
from ..services.scheduler import scheduler

router = APIRouter(prefix="/tasks", tags=["tasks"])


class TaskCreate(BaseModel):
    """Request to create a task."""
    name: str
    agent_type: str
    payload: dict = {}
    priority: int = 0
    scheduled_at: Optional[datetime] = None
    description: Optional[str] = None


class TaskResponse(BaseModel):
    """Task response model."""
    id: str
    name: str
    agent_type: str
    status: str
    priority: int
    description: Optional[str]
    payload: dict
    result: Optional[dict]
    error: Optional[str]
    scheduled_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


@router.post("/", response_model=TaskResponse)
async def create_task(
    task: TaskCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new task in the queue."""
    new_task = await scheduler.queue_task(
        name=task.name,
        agent_type=task.agent_type,
        payload=task.payload,
        priority=task.priority,
        scheduled_at=task.scheduled_at,
        description=task.description,
    )
    
    return TaskResponse(
        id=str(new_task.id),
        name=new_task.name,
        agent_type=new_task.agent_type,
        status=new_task.status.value,
        priority=new_task.priority,
        description=new_task.description,
        payload=new_task.payload,
        result=new_task.result,
        error=new_task.error,
        scheduled_at=new_task.scheduled_at,
        started_at=new_task.started_at,
        completed_at=new_task.completed_at,
        created_at=new_task.created_at,
    )


@router.get("/", response_model=list[TaskResponse])
async def list_tasks(
    status: Optional[str] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_db),
):
    """List tasks with optional filtering."""
    query = select(Task).order_by(desc(Task.created_at)).offset(offset).limit(limit)
    
    if status:
        query = query.where(Task.status == status)
    
    result = await db.execute(query)
    tasks = result.scalars().all()

    def _enum_value(v):
        return v.value if hasattr(v, "value") else v
    
    return [
        TaskResponse(
            id=str(t.id),
            name=t.name,
            agent_type=_enum_value(t.agent_type),
            status=_enum_value(t.status),
            priority=t.priority,
            description=t.description,
            payload=t.payload,
            result=t.result,
            error=t.error,
            scheduled_at=t.scheduled_at,
            started_at=t.started_at,
            completed_at=t.completed_at,
            created_at=t.created_at,
        )
        for t in tasks
    ]


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific task."""
    task = await db.get(Task, task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    def _enum_value(v):
        return v.value if hasattr(v, "value") else v
    
    return TaskResponse(
        id=str(task.id),
        name=task.name,
        agent_type=_enum_value(task.agent_type),
        status=_enum_value(task.status),
        priority=task.priority,
        description=task.description,
        payload=task.payload,
        result=task.result,
        error=task.error,
        scheduled_at=task.scheduled_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        created_at=task.created_at,
    )


@router.delete("/{task_id}")
async def cancel_task(
    task_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Cancel a pending task."""
    task = await db.get(Task, task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status not in [TaskStatus.PENDING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel task with status: {task.status}",
        )
    
    task.status = TaskStatus.CANCELLED
    await db.commit()
    
    return {"cancelled": True, "id": str(task_id)}


@router.get("/stats/summary")
async def get_task_stats(db: AsyncSession = Depends(get_db)):
    """Get task queue statistics."""
    from sqlalchemy import func
    
    # Count by status
    query = select(
        Task.status,
        func.count(Task.id).label("count"),
    ).group_by(Task.status)
    
    result = await db.execute(query)
    status_counts = {row.status.value: row.count for row in result}
    
    return {
        "total": sum(status_counts.values()),
        "by_status": status_counts,
        "scheduler_running": scheduler.is_running,
        "active_tasks": scheduler.active_task_count,
    }

