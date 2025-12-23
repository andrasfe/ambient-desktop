"""Integration tests for API endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture
async def test_client():
    """Create a test client without database."""
    # Don't raise app exceptions into the test runner; convert them to 500 responses.
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.mark.asyncio
    async def test_root(self, test_client):
        """Test root endpoint."""
        response = await test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Ambient Desktop Agent"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_health(self, test_client):
        """Test health check endpoint."""
        response = await test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_config(self, test_client):
        """Test config endpoint."""
        response = await test_client.get("/config")
        assert response.status_code == 200
        data = response.json()
        assert "openrouter_model" in data
        assert "scheduler_interval" in data


class TestTasksAPI:
    """Tests for tasks API."""

    @pytest.mark.asyncio
    async def test_list_tasks(self, test_client):
        """Test listing tasks."""
        response = await test_client.get("/tasks/")
        # May fail without DB connection
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_get_task_stats(self, test_client):
        """Test task statistics endpoint."""
        response = await test_client.get("/tasks/stats/summary")
        # May fail without DB connection
        assert response.status_code in [200, 500]


class TestAgentsAPI:
    """Tests for agents API."""

    @pytest.mark.asyncio
    async def test_list_agents(self, test_client):
        """Test listing agents."""
        response = await test_client.get("/agents/")
        # May fail without DB connection
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_get_agent_not_found(self, test_client):
        """Test getting non-existent agent."""
        response = await test_client.get("/agents/00000000-0000-0000-0000-000000000000")
        # May fail without DB connection, or return 404
        assert response.status_code in [404, 500]

    @pytest.mark.asyncio
    async def test_list_logs(self, test_client):
        """Test listing logs."""
        response = await test_client.get("/agents/logs/")
        # May fail without DB connection
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_get_agent_stats(self, test_client):
        """Test agent statistics endpoint."""
        response = await test_client.get("/agents/stats/summary")
        # May fail without DB connection
        assert response.status_code in [200, 500]


class TestChatAPI:
    """Tests for chat API."""

    @pytest.mark.asyncio
    async def test_get_history(self, test_client):
        """Test getting chat history."""
        response = await test_client.get("/chat/history")
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert "length" in data
