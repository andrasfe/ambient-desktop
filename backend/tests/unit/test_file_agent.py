"""Unit tests for file agent - file operations only (no database)."""

import pytest
from unittest.mock import AsyncMock, patch
import tempfile
import os
from pathlib import Path

from app.agents.file import FileAgent
from app.models import AgentType


class TestFileAgent:
    """Tests for FileAgent file operations."""

    def test_init(self):
        """Test agent initialization."""
        agent = FileAgent(name="test-file-agent")
        assert agent.name == "test-file-agent"
        assert agent.agent_type == AgentType.FILE

    def test_init_with_base_path(self):
        """Test agent initialization with base path."""
        agent = FileAgent(base_path="/tmp")
        assert agent.base_path == Path("/tmp")

    def test_resolve_path_relative(self):
        """Test relative path resolution."""
        agent = FileAgent(base_path="/tmp")
        resolved = agent._resolve_path("test.txt")
        assert resolved == Path("/tmp/test.txt")

    def test_resolve_path_absolute(self):
        """Test absolute path resolution."""
        agent = FileAgent(base_path="/tmp")
        resolved = agent._resolve_path("/home/test.txt")
        assert resolved == Path("/home/test.txt")

    @pytest.mark.asyncio
    async def test_read_file(self):
        """Test reading a file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, World!")
            temp_path = f.name

        try:
            agent = FileAgent()
            agent.update_status = AsyncMock()  # Mock to avoid DB
            
            result = await agent._action_read({"path": temp_path})
            
            assert result["content"] == "Hello, World!"
            assert result["size"] == 13
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_write_file(self):
        """Test writing a file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = FileAgent(base_path=temp_dir)
            agent.update_status = AsyncMock()  # Mock to avoid DB
            
            result = await agent._action_write({
                "path": "test.txt",
                "content": "Test content",
            })
            
            assert result["written"] is True
            
            # Verify file exists
            file_path = Path(temp_dir) / "test.txt"
            assert file_path.exists()
            assert file_path.read_text() == "Test content"

    @pytest.mark.asyncio
    async def test_list_directory(self):
        """Test listing directory contents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            Path(temp_dir, "file1.txt").touch()
            Path(temp_dir, "file2.txt").touch()
            Path(temp_dir, "subdir").mkdir()
            
            agent = FileAgent(base_path=temp_dir)
            agent.update_status = AsyncMock()  # Mock to avoid DB
            
            result = await agent._action_list({"path": "."})
            
            assert result["count"] == 3
            names = [e["name"] for e in result["entries"]]
            assert "file1.txt" in names
            assert "file2.txt" in names
            assert "subdir" in names

    @pytest.mark.asyncio
    async def test_exists(self):
        """Test checking file existence."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = f.name

        try:
            agent = FileAgent()
            
            result = await agent._action_exists({"path": temp_path})
            assert result["exists"] is True
            assert result["is_file"] is True
            
            result = await agent._action_exists({"path": "/nonexistent/path"})
            assert result["exists"] is False
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_delete_file(self):
        """Test deleting a file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = f.name

        agent = FileAgent()
        agent.update_status = AsyncMock()  # Mock to avoid DB
        
        # File exists before delete
        assert Path(temp_path).exists()
        
        result = await agent._action_delete({"path": temp_path})
        assert result["deleted"] is True
        
        # File no longer exists
        assert not Path(temp_path).exists()

    @pytest.mark.asyncio
    async def test_mkdir(self):
        """Test creating directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = FileAgent(base_path=temp_dir)
            agent.update_status = AsyncMock()  # Mock to avoid DB
            
            result = await agent._action_mkdir({"path": "new_dir/nested"})
            
            assert result["created"] is True
            assert (Path(temp_dir) / "new_dir" / "nested").is_dir()
