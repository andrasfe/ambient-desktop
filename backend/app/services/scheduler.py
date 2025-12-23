"""Task scheduler for periodic execution."""

import asyncio
from datetime import datetime, timezone
from typing import Optional, Callable, Awaitable
from uuid import UUID

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..database import get_session
from ..models import Task, TaskStatus
from .websocket import ws_manager


class TaskScheduler:
    """Manages scheduled task execution."""

    def __init__(self):
        self._scheduler = AsyncIOScheduler()
        self._running = False
        self._task_handlers: dict[str, Callable[[Task], Awaitable[dict]]] = {}
        self._active_tasks: dict[UUID, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    def register_handler(
        self,
        agent_type: str,
        handler: Callable[[Task], Awaitable[dict]],
    ) -> None:
        """Register a handler for a specific agent type."""
        self._task_handlers[agent_type] = handler

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return

        self._scheduler.add_job(
            self._process_pending_tasks,
            trigger=IntervalTrigger(seconds=settings.scheduler_interval_seconds),
            id="process_pending_tasks",
            replace_existing=True,
        )
        self._scheduler.start()
        self._running = True
        
        await ws_manager.broadcast_log(
            level="info",
            message=f"Task scheduler started (interval: {settings.scheduler_interval_seconds}s)",
            category="scheduler",
        )

    async def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return
        
        self._scheduler.shutdown(wait=False)
        self._running = False
        
        # Cancel active tasks
        async with self._lock:
            for task in self._active_tasks.values():
                task.cancel()
            self._active_tasks.clear()

    async def _process_pending_tasks(self) -> None:
        """Process pending tasks from the queue."""
        async with get_session() as session:
            # Get pending tasks that are due
            now = datetime.now(timezone.utc)
            query = select(Task).where(
                and_(
                    Task.status == TaskStatus.PENDING,
                    (Task.scheduled_at == None) | (Task.scheduled_at <= now),
                )
            ).order_by(Task.priority.desc(), Task.created_at).limit(
                settings.max_concurrent_agents
            )
            
            result = await session.execute(query)
            tasks = result.scalars().all()
            
            for task in tasks:
                await self._execute_task(task, session)

    async def _execute_task(self, task: Task, session: AsyncSession) -> None:
        """Execute a single task."""
        handler = self._task_handlers.get(task.agent_type)
        if not handler:
            await ws_manager.broadcast_log(
                level="error",
                message=f"No handler for agent type: {task.agent_type}",
                category="scheduler",
                task_id=task.id,
            )
            task.status = TaskStatus.FAILED
            task.error = f"No handler for agent type: {task.agent_type}"
            await session.commit()
            return

        # Check if we have capacity
        async with self._lock:
            if len(self._active_tasks) >= settings.max_concurrent_agents:
                return
        
        # Mark as running
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(timezone.utc)
        await session.commit()
        
        await ws_manager.broadcast_task_update(
            task_id=task.id,
            status="running",
        )

        # Execute in background
        async def run_task():
            try:
                result = await handler(task)
                async with get_session() as s:
                    t = await s.get(Task, task.id)
                    if t:
                        t.status = TaskStatus.COMPLETED
                        t.result = result
                        t.completed_at = datetime.now(timezone.utc)
                        await s.commit()
                
                await ws_manager.broadcast_task_update(
                    task_id=task.id,
                    status="completed",
                    result=result,
                )
            except Exception as e:
                async with get_session() as s:
                    t = await s.get(Task, task.id)
                    if t:
                        t.status = TaskStatus.FAILED
                        t.error = str(e)
                        t.completed_at = datetime.now(timezone.utc)
                        await s.commit()
                
                await ws_manager.broadcast_task_update(
                    task_id=task.id,
                    status="failed",
                    error=str(e),
                )
                await ws_manager.broadcast_log(
                    level="error",
                    message=f"Task failed: {str(e)}",
                    category="scheduler",
                    task_id=task.id,
                )
            finally:
                async with self._lock:
                    self._active_tasks.pop(task.id, None)

        async with self._lock:
            self._active_tasks[task.id] = asyncio.create_task(run_task())

    async def queue_task(
        self,
        name: str,
        agent_type: str,
        payload: dict,
        priority: int = 0,
        scheduled_at: Optional[datetime] = None,
        parent_id: Optional[UUID] = None,
        description: Optional[str] = None,
    ) -> Task:
        """Add a new task to the queue."""
        async with get_session() as session:
            task = Task(
                name=name,
                agent_type=agent_type,
                payload=payload,
                priority=priority,
                scheduled_at=scheduled_at,
                parent_id=parent_id,
                description=description,
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)
            
            await ws_manager.broadcast(
                "task:created",
                {
                    "id": str(task.id),
                    "name": task.name,
                    "agent_type": task.agent_type,
                    "status": task.status.value,
                    "priority": task.priority,
                },
            )
            
            return task

    @property
    def is_running(self) -> bool:
        """Check if the scheduler is running."""
        return self._running

    @property
    def active_task_count(self) -> int:
        """Get the number of active tasks."""
        return len(self._active_tasks)


# Global instance
scheduler = TaskScheduler()

