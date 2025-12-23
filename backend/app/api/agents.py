"""Agent status API endpoints."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import Agent, AgentStatus, AgentType, ActivityLog, LogLevel
from ..services.websocket import ws_manager

router = APIRouter(prefix="/agents", tags=["agents"])


class AgentResponse(BaseModel):
    """Agent response model."""
    id: str
    type: str
    name: str
    status: str
    summary: Optional[str]
    progress: Optional[float]
    current_task_id: Optional[str]
    metadata: dict

    class Config:
        from_attributes = True


class LogResponse(BaseModel):
    """Activity log response model."""
    id: int
    agent_id: Optional[str]
    task_id: Optional[str]
    level: str
    category: str
    message: str
    details: Optional[dict]
    created_at: str

    class Config:
        from_attributes = True


@router.get("/", response_model=list[AgentResponse])
async def list_agents(
    status: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """List all agents."""
    query = select(Agent).order_by(desc(Agent.created_at))
    
    if status:
        query = query.where(Agent.status == status)
    if type:
        query = query.where(Agent.type == type)
    
    result = await db.execute(query)
    agents = result.scalars().all()
    
    return [
        AgentResponse(
            id=str(a.id),
            type=a.type.value,
            name=a.name,
            status=a.status.value,
            summary=a.summary,
            progress=a.progress,
            current_task_id=str(a.current_task_id) if a.current_task_id else None,
            metadata=a.extra_data,
        )
        for a in agents
    ]


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific agent."""
    agent = await db.get(Agent, agent_id)
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return AgentResponse(
        id=str(agent.id),
        type=agent.type.value,
        name=agent.name,
        status=agent.status.value,
        summary=agent.summary,
        progress=agent.progress,
        current_task_id=str(agent.current_task_id) if agent.current_task_id else None,
        metadata=agent.extra_data,
    )


@router.get("/logs/", response_model=list[LogResponse])
async def list_logs(
    level: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    agent_id: Optional[UUID] = Query(None),
    task_id: Optional[UUID] = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_db),
):
    """List activity logs."""
    query = select(ActivityLog).order_by(desc(ActivityLog.created_at)).offset(offset).limit(limit)
    
    if level:
        query = query.where(ActivityLog.level == level)
    if category:
        query = query.where(ActivityLog.category == category)
    if agent_id:
        query = query.where(ActivityLog.agent_id == agent_id)
    if task_id:
        query = query.where(ActivityLog.task_id == task_id)
    
    result = await db.execute(query)
    logs = result.scalars().all()
    
    return [
        LogResponse(
            id=l.id,
            agent_id=str(l.agent_id) if l.agent_id else None,
            task_id=str(l.task_id) if l.task_id else None,
            level=l.level.value,
            category=l.category,
            message=l.message,
            details=l.details,
            created_at=l.created_at.isoformat(),
        )
        for l in logs
    ]


@router.get("/stats/summary")
async def get_agent_stats(db: AsyncSession = Depends(get_db)):
    """Get agent statistics."""
    from sqlalchemy import func
    
    # Count by status
    status_query = select(
        Agent.status,
        func.count(Agent.id).label("count"),
    ).group_by(Agent.status)
    
    status_result = await db.execute(status_query)
    status_counts = {row.status.value: row.count for row in status_result}
    
    # Count by type
    type_query = select(
        Agent.type,
        func.count(Agent.id).label("count"),
    ).group_by(Agent.type)
    
    type_result = await db.execute(type_query)
    type_counts = {row.type.value: row.count for row in type_result}
    
    return {
        "total": sum(status_counts.values()),
        "by_status": status_counts,
        "by_type": type_counts,
        "websocket_connections": ws_manager.connection_count,
    }

