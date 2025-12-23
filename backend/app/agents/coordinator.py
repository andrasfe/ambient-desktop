"""Coordinator agent for multi-agent orchestration."""

import asyncio
from typing import Any, Optional
from uuid import UUID

from .base import BaseAgent
from .browser import BrowserAgent
from .file import FileAgent
from ..models import AgentType, AgentStatus, Task, TaskStatus
from ..database import get_session
from ..services.llm import llm_service, Message
from ..services.websocket import ws_manager, EventType


class CoordinatorAgent(BaseAgent):
    """Orchestrates multiple agents to complete complex tasks."""

    agent_type = AgentType.COORDINATOR

    def __init__(self, name: Optional[str] = None, max_workers: int = 5):
        super().__init__(name or "coordinator")
        self.max_workers = max_workers
        self._workers: dict[UUID, BaseAgent] = {}
        self._conversation: list[Message] = []

    async def start(self) -> None:
        """Start the coordinator."""
        await super().start()
        await self.log("info", "Coordinator ready to orchestrate agents")

    async def stop(self) -> None:
        """Stop the coordinator and all workers."""
        # Stop all workers
        for worker in list(self._workers.values()):
            await worker.stop()
        self._workers.clear()
        await super().stop()

    async def _execute_task(self, task: Task) -> dict[str, Any]:
        """Execute a coordinator task."""
        instruction = task.payload.get("instruction", "")
        context = task.payload.get("context")
        
        # Add to conversation history
        self._conversation.append(Message(role="user", content=instruction))
        
        # Parse the instruction into subtasks
        await self.update_status(AgentStatus.BUSY, summary="Analyzing instruction...")
        plan = await llm_service.parse_task(instruction, context)
        
        await self.log("info", f"Created plan with {len(plan.get('tasks', []))} tasks", {
            "understanding": plan.get("understanding"),
            "task_count": len(plan.get("tasks", [])),
        })
        
        # Execute the plan
        results = await self._execute_plan(plan, task)
        
        # Generate summary
        summary = await self._generate_summary(plan, results)
        
        # Add assistant response to history
        self._conversation.append(Message(role="assistant", content=summary))
        
        return {
            "plan": plan,
            "results": results,
            "summary": summary,
        }

    async def _execute_plan(self, plan: dict, parent_task: Task) -> list[dict]:
        """Execute a plan by spawning agents for each task."""
        tasks = plan.get("tasks", [])
        results = []
        completed = set()
        
        for i, task_def in enumerate(tasks):
            # Check dependencies
            dependencies = task_def.get("dependencies", [])
            while not all(d in completed for d in dependencies):
                await asyncio.sleep(0.1)
            
            # Create and execute the task
            await self.update_status(
                AgentStatus.BUSY,
                summary=f"Executing task {i+1}/{len(tasks)}: {task_def['name']}",
                progress=(i / len(tasks)),
            )
            
            try:
                result = await self._execute_subtask(task_def, parent_task)
                results.append({
                    "task": task_def["name"],
                    "status": "completed",
                    "result": result,
                })
            except Exception as e:
                results.append({
                    "task": task_def["name"],
                    "status": "failed",
                    "error": str(e),
                })
                await self.log("error", f"Task failed: {task_def['name']}", {"error": str(e)})
            
            completed.add(i)
        
        return results

    async def _execute_subtask(self, task_def: dict, parent_task: Task) -> dict:
        """Execute a single subtask using the appropriate agent."""
        agent_type = task_def.get("agent_type", "browser")
        
        # Create subtask in database
        async with get_session() as session:
            subtask = Task(
                name=task_def["name"],
                agent_type=agent_type,
                description=task_def.get("description"),
                payload=task_def.get("payload", {}),
                parent_id=parent_task.id,
            )
            session.add(subtask)
            await session.commit()
            await session.refresh(subtask)
        
        # Get or create appropriate agent
        agent = await self._get_agent(agent_type)
        
        try:
            # Execute the subtask
            result = await agent.execute(subtask)
            
            # Update subtask status
            async with get_session() as session:
                t = await session.get(Task, subtask.id)
                if t:
                    t.status = TaskStatus.COMPLETED
                    t.result = result
                    await session.commit()
            
            return result
        except Exception as e:
            async with get_session() as session:
                t = await session.get(Task, subtask.id)
                if t:
                    t.status = TaskStatus.FAILED
                    t.error = str(e)
                    await session.commit()
            raise

    async def _get_agent(self, agent_type: str) -> BaseAgent:
        """Get or create an agent of the specified type."""
        # Look for an idle agent of this type
        for agent in self._workers.values():
            if agent.agent_type.value == agent_type and agent.status == AgentStatus.IDLE:
                return agent
        
        # Check capacity
        if len(self._workers) >= self.max_workers:
            # Wait for an agent to become available
            while True:
                for agent in self._workers.values():
                    if agent.agent_type.value == agent_type and agent.status == AgentStatus.IDLE:
                        return agent
                await asyncio.sleep(0.5)
        
        # Create a new agent
        if agent_type == "browser":
            agent = BrowserAgent()
        elif agent_type == "file":
            agent = FileAgent()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        await agent.start()
        self._workers[agent.id] = agent
        
        return agent

    async def _generate_summary(self, plan: dict, results: list[dict]) -> str:
        """Generate a natural language summary of what was done."""
        completed = sum(1 for r in results if r["status"] == "completed")
        failed = sum(1 for r in results if r["status"] == "failed")
        
        summary_parts = [f"Completed {completed}/{len(results)} tasks."]
        
        if failed > 0:
            summary_parts.append(f"{failed} task(s) failed.")
        
        # Ask LLM to generate a nicer summary
        prompt = f"""Summarize what was accomplished in a concise, user-friendly way:

Understanding: {plan.get('understanding', 'Unknown')}

Tasks completed:
{chr(10).join(f"- {r['task']}: {r['status']}" for r in results)}

Keep the summary brief and actionable."""

        try:
            response = await llm_service.complete(
                messages=[Message(role="user", content=prompt)],
                temperature=0.5,
                max_tokens=200,
            )
            return response.content
        except Exception:
            return " ".join(summary_parts)

    async def chat(self, message: str, context: Optional[str] = None) -> str:
        """Handle a chat message and return a response."""
        # Add message to conversation
        self._conversation.append(Message(role="user", content=message))
        
        # Determine if this is a task or just a question
        system_prompt = """You are an AI assistant that can control a computer through browser automation and file operations.

If the user is asking you to DO something (navigate, click, extract, read files, etc.), respond with:
ACTION: <brief description of what you'll do>

If the user is just asking a question or chatting, respond normally.

Be concise and helpful."""

        response = await llm_service.complete(
            messages=self._conversation[-10:],  # Keep last 10 messages for context
            system_prompt=system_prompt,
            temperature=0.7,
        )
        
        response_text = response.content
        
        # Check if this requires action
        if response_text.startswith("ACTION:"):
            # Create and execute a task
            async with get_session() as session:
                task = Task(
                    name="User instruction",
                    agent_type="coordinator",
                    payload={"instruction": message, "context": context},
                )
                session.add(task)
                await session.commit()
                await session.refresh(task)
            
            result = await self.execute(task)
            response_text = result.get("summary", response_text)
        
        self._conversation.append(Message(role="assistant", content=response_text))
        return response_text

    async def stream_chat(self, message: str, context: Optional[str] = None):
        """Stream a chat response."""
        self._conversation.append(Message(role="user", content=message))
        
        system_prompt = """You are an AI assistant that can control a computer through browser automation and file operations.
Be concise and helpful. If you need to perform actions, describe what you're doing."""

        full_response = ""
        async for chunk in llm_service.stream(
            messages=self._conversation[-10:],
            system_prompt=system_prompt,
            temperature=0.7,
        ):
            full_response += chunk
            yield chunk
        
        self._conversation.append(Message(role="assistant", content=full_response))

    @property
    def worker_count(self) -> int:
        """Get the number of active workers."""
        return len(self._workers)

    @property
    def conversation_length(self) -> int:
        """Get the conversation history length."""
        return len(self._conversation)

