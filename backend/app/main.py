"""Main FastAPI application."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import init_db, close_db
from .services.scheduler import scheduler
from .services.websocket import ws_manager
from .api import chat_router, tasks_router, agents_router
from .agents import BrowserAgent, FileAgent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("🚀 Starting Ambient Desktop Agent...")
    
    # Initialize database
    await init_db()
    print("✅ Database initialized")
    
    # Register task handlers with scheduler
    async def handle_browser_task(task):
        agent = BrowserAgent()
        await agent.start()
        try:
            return await agent.execute(task)
        finally:
            await agent.stop()
    
    async def handle_file_task(task):
        agent = FileAgent()
        await agent.start()
        try:
            return await agent.execute(task)
        finally:
            await agent.stop()
    
    scheduler.register_handler("browser", handle_browser_task)
    scheduler.register_handler("file", handle_file_task)
    
    # Start scheduler
    await scheduler.start()
    print("✅ Task scheduler started")
    
    print("✅ LangGraph agent workflow ready")
    
    print("🟢 Ambient Desktop Agent is ready!")
    print(f"   API: http://localhost:{settings.port}")
    print(f"   Docs: http://localhost:{settings.port}/docs")
    
    yield
    
    # Shutdown
    print("🔴 Shutting down...")
    
    await scheduler.stop()
    await close_db()
    
    print("👋 Goodbye!")


# Create application
app = FastAPI(
    title="Ambient Desktop Agent",
    description="An always-on multi-agent system for computer automation powered by LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
app.include_router(tasks_router)
app.include_router(agents_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Ambient Desktop Agent",
        "version": "1.0.0",
        "framework": "LangGraph",
        "status": "running",
        "scheduler": scheduler.is_running,
        "connections": ws_manager.connection_count,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "framework": "LangGraph",
        "scheduler_running": scheduler.is_running,
        "active_tasks": scheduler.active_task_count,
        "websocket_connections": ws_manager.connection_count,
    }


@app.get("/config")
async def get_config():
    """Get non-sensitive configuration."""
    return {
        "framework": "LangGraph",
        "openrouter_model": settings.openrouter_model,
        "cohere_model": settings.cohere_model,
        "scheduler_interval": settings.scheduler_interval_seconds,
        "max_concurrent_agents": settings.max_concurrent_agents,
    }
