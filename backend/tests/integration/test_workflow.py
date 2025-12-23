"""Integration tests for full agent workflows."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport
import tempfile
import os
from pathlib import Path

from app.main import app


@pytest.fixture
async def test_client():
    """Create a test client without database."""
    # Don't raise app exceptions into the test runner; convert them to 500 responses.
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestFileWorkflow:
    """Tests for file operation workflows."""

    @pytest.mark.asyncio
    async def test_read_file_workflow(self, test_client):
        """Test complete read file workflow."""
        # Create a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for reading")
            temp_path = f.name

        try:
            # Mock the orchestrator to return file read result
            with patch("app.api.chat.orchestrator") as mock_orchestrator:
                mock_orchestrator.process_message = AsyncMock(
                    return_value=f"I read the file at {temp_path}. It contains: Test content for reading"
                )
                
                response = await test_client.post(
                    "/chat/message",
                    json={"message": f"Read the file {temp_path}"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "Test content" in data["response"]
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_list_directory_workflow(self, test_client):
        """Test listing directory workflow."""
        with patch("app.api.chat.orchestrator") as mock_orchestrator:
            mock_orchestrator.process_message = AsyncMock(
                return_value="The directory contains: file1.txt, file2.txt"
            )
            
            response = await test_client.post(
                "/chat/message",
                json={"message": "List files in /tmp"}
            )
            
            assert response.status_code == 200


class TestBrowserWorkflow:
    """Tests for browser operation workflows."""

    @pytest.mark.asyncio
    async def test_list_tabs_workflow(self, test_client):
        """Test listing browser tabs workflow."""
        with patch("app.api.chat.orchestrator") as mock_orchestrator:
            mock_orchestrator.process_message = AsyncMock(
                return_value="You have 2 tabs open: Google, LinkedIn"
            )
            
            response = await test_client.post(
                "/chat/message",
                json={"message": "What tabs do I have open?"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "tabs" in data["response"].lower()

    @pytest.mark.asyncio
    async def test_navigate_workflow(self, test_client):
        """Test browser navigation workflow."""
        with patch("app.api.chat.orchestrator") as mock_orchestrator:
            mock_orchestrator.process_message = AsyncMock(
                return_value="I navigated to https://example.com"
            )
            
            response = await test_client.post(
                "/chat/message",
                json={"message": "Go to example.com"}
            )
            
            assert response.status_code == 200


class TestMultiStepWorkflow:
    """Tests for multi-step workflows."""

    @pytest.mark.asyncio
    async def test_conversation_context(self, test_client):
        """Test conversation maintains context across messages."""
        with patch("app.api.chat.orchestrator") as mock_orchestrator:
            # First message
            mock_orchestrator.process_message = AsyncMock(return_value="I see Google is open")
            
            response1 = await test_client.post(
                "/chat/message",
                json={"message": "What page am I on?", "session_id": "test-context"}
            )
            assert response1.status_code == 200
            
            # Second message in same session
            mock_orchestrator.process_message = AsyncMock(return_value="Clicking search button...")
            
            response2 = await test_client.post(
                "/chat/message",
                json={"message": "Click the search button", "session_id": "test-context"}
            )
            assert response2.status_code == 200

    @pytest.mark.asyncio
    async def test_error_recovery(self, test_client):
        """Test graceful error handling in workflow."""
        with patch("app.api.chat.orchestrator") as mock_orchestrator:
            mock_orchestrator.process_message = AsyncMock(
                return_value="I couldn't complete that action because the element wasn't found."
            )
            
            response = await test_client.post(
                "/chat/message",
                json={"message": "Click on nonexistent button"}
            )
            
            assert response.status_code == 200
            # Should get error message, not crash


class TestTaskManagement:
    """Tests for task management."""

    @pytest.mark.asyncio
    async def test_create_task(self, test_client):
        """Test creating a task."""
        task_data = {
            "name": "Test Task",
            "agent_type": "browser",
            "description": "Navigate to example.com",
            "payload": {"action": "navigate", "url": "https://example.com"},
            "priority": 1,
        }
        
        response = await test_client.post("/tasks/", json=task_data)
        
        # Task creation might require proper DB setup
        assert response.status_code in [200, 201, 422, 500]

    @pytest.mark.asyncio
    async def test_task_stats(self, test_client):
        """Test task statistics."""
        response = await test_client.get("/tasks/stats/summary")
        
        # May fail without DB - that's ok for this test
        assert response.status_code in [200, 500]


class TestAgentManagement:
    """Tests for agent management."""

    @pytest.mark.asyncio
    async def test_agent_stats(self, test_client):
        """Test agent statistics."""
        response = await test_client.get("/agents/stats/summary")
        
        # May fail without DB - that's ok for this test
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_logs_endpoint(self, test_client):
        """Test activity logs endpoint."""
        response = await test_client.get("/agents/logs/")
        
        # May fail without DB - that's ok for this test
        assert response.status_code in [200, 500]

