"""Unit tests for task scheduler."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.scheduler import TaskScheduler


class TestTaskScheduler:
    """Tests for TaskScheduler."""

    def test_init(self):
        """Test scheduler initialization."""
        scheduler = TaskScheduler()
        assert scheduler._scheduler is not None
        assert scheduler._running is False
        assert scheduler._task_handlers == {}
        assert scheduler._active_tasks == {}

    def test_is_running_false_initially(self):
        """Test is_running is False initially."""
        scheduler = TaskScheduler()
        assert scheduler.is_running is False

    def test_active_task_count_zero_initially(self):
        """Test active_task_count is 0 initially."""
        scheduler = TaskScheduler()
        assert scheduler.active_task_count == 0

    def test_register_handler(self):
        """Test registering a handler."""
        scheduler = TaskScheduler()
        
        async def mock_handler(task):
            return {"done": True}
        
        scheduler.register_handler("browser", mock_handler)
        
        assert "browser" in scheduler._task_handlers
        assert scheduler._task_handlers["browser"] == mock_handler

    def test_register_multiple_handlers(self):
        """Test registering multiple handlers."""
        scheduler = TaskScheduler()
        
        async def browser_handler(task):
            return {}
        
        async def file_handler(task):
            return {}
        
        scheduler.register_handler("browser", browser_handler)
        scheduler.register_handler("file", file_handler)
        
        assert len(scheduler._task_handlers) == 2

    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        """Test start sets _running to True."""
        scheduler = TaskScheduler()
        
        # Mock the scheduler to avoid starting real scheduler
        scheduler._scheduler = MagicMock()
        scheduler._scheduler.add_job = MagicMock()
        scheduler._scheduler.start = MagicMock()
        
        # Mock ws_manager
        with patch("app.services.scheduler.ws_manager") as mock_ws:
            mock_ws.broadcast_log = AsyncMock()
            
            await scheduler.start()
            
            assert scheduler._running is True
            scheduler._scheduler.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """Test start is idempotent."""
        scheduler = TaskScheduler()
        scheduler._running = True
        
        scheduler._scheduler = MagicMock()
        scheduler._scheduler.start = MagicMock()
        
        await scheduler.start()
        
        scheduler._scheduler.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_sets_not_running(self):
        """Test stop sets _running to False."""
        scheduler = TaskScheduler()
        scheduler._running = True
        
        scheduler._scheduler = MagicMock()
        scheduler._scheduler.shutdown = MagicMock()
        
        await scheduler.stop()
        
        assert scheduler._running is False
        scheduler._scheduler.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_idempotent(self):
        """Test stop is idempotent."""
        scheduler = TaskScheduler()
        scheduler._running = False
        
        scheduler._scheduler = MagicMock()
        scheduler._scheduler.shutdown = MagicMock()
        
        await scheduler.stop()
        
        scheduler._scheduler.shutdown.assert_not_called()
