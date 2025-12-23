"""Unit tests for LangGraph agent orchestration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from app.agents.graph import (
    AgentState,
    sanitize_for_llm,
    create_llm,
    route_next_action,
    AgentOrchestrator,
)


@pytest.mark.asyncio
async def test_coordinator_tool_call_markup_parsing(monkeypatch):
    """
    Some models emit raw tool-call markup instead of JSON. Ensure we convert it into subtasks.
    """
    from app.agents import graph as graph_mod
    from langchain_core.messages import HumanMessage

    class _FakeResp:
        def __init__(self, content: str):
            self.content = content

    class _FakeLLM:
        async def ainvoke(self, messages):
            return _FakeResp(
                "I'll switch to LinkedIn.\n"
                "<|tool_calls_section_begin|>\n"
                "<|tool_call_begin|> functions.switch_tab:0 <|tool_call_argument_begin|> "
                '{"url_contains":"linkedin.com"}'
                " <|tool_call_end|>\n"
                "<|tool_calls_section_end|>"
            )

    monkeypatch.setattr(graph_mod, "create_llm", lambda: _FakeLLM())

    state: AgentState = {
        "messages": [HumanMessage(content="give me all the patent titles")],
        "task_id": None,
        "task_type": None,
        "task_payload": None,
        "subtasks": [],
        "current_subtask_index": 0,
        "results": [],
        "agent_id": "test",
        "session_id": "test",
        "next_action": None,
        "error": None,
        "selected_tab_index": None,
    }

    out = await graph_mod.coordinator_node(state)
    assert out["subtasks"], "Expected subtasks to be created from tool-call markup"
    assert out["subtasks"][0]["agent"] == "browser"
    assert out["subtasks"][0]["action"] == "switch_tab"
    assert out["subtasks"][0]["params"]["url_contains"] == "linkedin.com"


class TestSanitizeForLLM:
    """Tests for sanitize_for_llm function."""

    def test_no_sanitization_when_privacy_disabled(self):
        """Test no sanitization when privacy mode is disabled."""
        with patch("app.agents.graph.settings") as mock_settings:
            mock_settings.privacy_mode = False
            mock_settings.is_local_llm = False
            
            data = {"content": "sensitive data", "url": "https://example.com"}
            result = sanitize_for_llm(data)
            
            assert result == data

    def test_no_sanitization_for_local_llm(self):
        """Test no sanitization for local LLM."""
        with patch("app.agents.graph.settings") as mock_settings:
            mock_settings.privacy_mode = True
            mock_settings.is_local_llm = True
            
            data = {"content": "sensitive data"}
            result = sanitize_for_llm(data)
            
            assert result == data

    def test_sanitize_content_field(self):
        """Test content field is sanitized."""
        with patch("app.agents.graph.settings") as mock_settings:
            mock_settings.privacy_mode = True
            mock_settings.is_local_llm = False
            
            data = {"content": "sensitive content here", "url": "https://example.com"}
            result = sanitize_for_llm(data)
            
            assert "[REDACTED" in result["content"]
            assert result["url"] == "https://example.com"

    def test_sanitize_password_field(self):
        """Test password field is sanitized."""
        with patch("app.agents.graph.settings") as mock_settings:
            mock_settings.privacy_mode = True
            mock_settings.is_local_llm = False
            
            data = {"password": "secret123", "username": "user"}
            result = sanitize_for_llm(data)
            
            assert "[REDACTED" in result["password"]
            assert result["username"] == "user"

    def test_sanitize_nested_dict(self):
        """Test nested dictionaries are sanitized."""
        with patch("app.agents.graph.settings") as mock_settings:
            mock_settings.privacy_mode = True
            mock_settings.is_local_llm = False
            
            data = {
                "page": {
                    "content": "page content",
                    "url": "https://example.com"
                }
            }
            result = sanitize_for_llm(data)
            
            assert "[REDACTED" in result["page"]["content"]
            assert result["page"]["url"] == "https://example.com"

    def test_sanitize_list_of_dicts(self):
        """Test lists of dictionaries are sanitized."""
        with patch("app.agents.graph.settings") as mock_settings:
            mock_settings.privacy_mode = True
            mock_settings.is_local_llm = False
            
            data = {
                "items": [
                    {"content": "item 1"},
                    {"content": "item 2"}
                ]
            }
            result = sanitize_for_llm(data)
            
            assert "[REDACTED" in result["items"][0]["content"]
            assert "[REDACTED" in result["items"][1]["content"]


class TestRouteNextAction:
    """Tests for route_next_action function."""

    def test_route_to_browser(self):
        """Test routing to browser node."""
        state: AgentState = {
            "messages": [],
            "next_action": "browser",
            "task_id": None,
            "task_type": None,
            "task_payload": None,
            "subtasks": [],
            "current_subtask_index": 0,
            "results": [],
            "agent_id": "test",
            "session_id": "test",
            "error": None,
        }
        assert route_next_action(state) == "browser"

    def test_route_to_file(self):
        """Test routing to file node."""
        state: AgentState = {
            "messages": [],
            "next_action": "file",
            "task_id": None,
            "task_type": None,
            "task_payload": None,
            "subtasks": [],
            "current_subtask_index": 0,
            "results": [],
            "agent_id": "test",
            "session_id": "test",
            "error": None,
        }
        assert route_next_action(state) == "file"

    def test_route_to_summarize(self):
        """Test routing to summarize node."""
        state: AgentState = {
            "messages": [],
            "next_action": "summarize",
            "task_id": None,
            "task_type": None,
            "task_payload": None,
            "subtasks": [],
            "current_subtask_index": 0,
            "results": [],
            "agent_id": "test",
            "session_id": "test",
            "error": None,
        }
        assert route_next_action(state) == "summarize"

    def test_route_to_end(self):
        """Test routing to end."""
        state: AgentState = {
            "messages": [],
            "next_action": "end",
            "task_id": None,
            "task_type": None,
            "task_payload": None,
            "subtasks": [],
            "current_subtask_index": 0,
            "results": [],
            "agent_id": "test",
            "session_id": "test",
            "error": None,
        }
        assert route_next_action(state) == "end"

    def test_route_default_to_end(self):
        """Test routing defaults to end."""
        state: AgentState = {
            "messages": [],
            "next_action": None,
            "task_id": None,
            "task_type": None,
            "task_payload": None,
            "subtasks": [],
            "current_subtask_index": 0,
            "results": [],
            "agent_id": "test",
            "session_id": "test",
            "error": None,
        }
        assert route_next_action(state) == "end"


class TestAgentOrchestrator:
    """Tests for AgentOrchestrator."""

    def test_init(self):
        """Test orchestrator initialization."""
        orchestrator = AgentOrchestrator()
        assert orchestrator.sessions == {}
        assert orchestrator.workflow is not None
        assert orchestrator.graph is not None

    def test_get_thread_id_creates_new(self):
        """Test thread ID creation for new session."""
        orchestrator = AgentOrchestrator()
        
        thread_id = orchestrator._get_thread_id("session-1")
        
        assert thread_id is not None
        assert "session-1" in orchestrator.sessions

    def test_get_thread_id_returns_existing(self):
        """Test thread ID returns existing for same session."""
        orchestrator = AgentOrchestrator()
        
        thread_id_1 = orchestrator._get_thread_id("session-1")
        thread_id_2 = orchestrator._get_thread_id("session-1")
        
        assert thread_id_1 == thread_id_2

    def test_different_sessions_different_threads(self):
        """Test different sessions get different thread IDs."""
        orchestrator = AgentOrchestrator()
        
        thread_id_1 = orchestrator._get_thread_id("session-1")
        thread_id_2 = orchestrator._get_thread_id("session-2")
        
        assert thread_id_1 != thread_id_2


class TestCreateLLM:
    """Tests for create_llm function."""

    def test_create_cloud_llm(self):
        """Test creating cloud LLM."""
        with patch("app.agents.graph.settings") as mock_settings:
            mock_settings.is_local_llm = False
            mock_settings.openrouter_api_key = "test-key"
            mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
            mock_settings.openrouter_model = "test-model"
            
            llm = create_llm()
            
            assert llm is not None
            assert llm.model_name == "test-model"

    def test_create_local_llm(self):
        """Test creating local LLM (Ollama)."""
        with patch("app.agents.graph.settings") as mock_settings:
            mock_settings.is_local_llm = True
            mock_settings.openrouter_model = "llama3.1:8b"
            mock_settings.openrouter_base_url = "http://localhost:11434/v1"
            
            llm = create_llm()
            
            assert llm is not None
            assert llm.model_name == "llama3.1:8b"

